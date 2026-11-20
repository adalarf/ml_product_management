import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_product_management.config import PROJECT_DIR, PipelineError
from ml_product_management.data.dataset import sha256_hex
from ml_product_management.training.config import (
    ModelConfig,
    RunConfig,
    TrainConfig,
    flatten_params,
)
from ml_product_management.training.data import ImageDataset, download_dataset, stratified_subset
from ml_product_management.training.engine import Evaluation, evaluate, fit
from ml_product_management.training.experiments import (
    container_command,
    expand_grid,
    run_series,
    select_best,
)
from ml_product_management.training.metrics import Metrics, from_confusion
from ml_product_management.training.models import build_model


class FakeStorage:
    bucket = "test"
    endpoint_url = "http://test-s3"

    def __init__(self, root):
        self.root = root
        self.downloads = []

    def uri(self, key):
        return f"s3://{self.bucket}/{key}"

    def download_file(self, key, destination):
        self.downloads.append(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / key, destination)


def record(index, label):
    return {
        "image_id": str(index),
        "file_name": f"images/{index}.png",
        "label": label,
        "label_name": str(label),
        "sha256": "unused",
    }


class DataTests(unittest.TestCase):
    def test_stratified_subset_is_exact_reproducible_and_seeded(self):
        records = [record(index, index % 10) for index in range(100)]
        first = stratified_subset(records, 30, 42)
        self.assertEqual(first, stratified_subset(records, 30, 42))
        self.assertNotEqual(first, stratified_subset(records, 30, 43))
        self.assertEqual(len(first), 30)
        self.assertEqual(
            [sum(item["label"] == label for item in first) for label in range(10)], [3] * 10
        )
        self.assertEqual(len(records), 100)
        with self.assertRaises(PipelineError):
            stratified_subset(records, 5, 42)

    def test_normalization_uses_supplied_channel_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            pixels = np.full((32, 32, 3), [0, 128, 255], dtype=np.uint8)
            Image.fromarray(pixels).save(root / "images/0.png")
            annotation = {"classes": ["a"], "images": [record(0, 0)]}
            dataset = ImageDataset(
                root, annotation, {"mean": [0.1, 0.2, 0.3], "std": [0.5, 0.25, 0.7]}
            )
            tensor, label = dataset[0]
            expected = (
                torch.tensor([0.0, 128 / 255, 1.0]) - torch.tensor([0.1, 0.2, 0.3])
            ) / torch.tensor([0.5, 0.25, 0.7])
            torch.testing.assert_close(tensor[:, 0, 0], expected)
            self.assertEqual(label, 0)

    def test_partial_cache_and_corruption_are_repaired(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            version = root / "remote/v1"
            (version / "images").mkdir(parents=True)
            (version / "annotations").mkdir()
            files = {}
            for index, split in enumerate(("train", "val", "test")):
                image = version / f"images/{index}.png"
                Image.fromarray(np.full((32, 32, 3), index, dtype=np.uint8)).save(image)
                item = record(index, 0)
                item["sha256"] = sha256_hex(image.read_bytes())
                path = version / f"annotations/{split}.json"
                path.write_text(json.dumps({"classes": ["a"], "images": [item]}), encoding="utf-8")
            (version / "stats.json").write_text(
                json.dumps({"source": "train", "mean": [0, 0, 0], "std": [1, 1, 1]}),
                encoding="utf-8",
            )
            for path in [
                *sorted((version / "annotations").glob("*.json")),
                version / "stats.json",
            ]:
                files[path.relative_to(version).as_posix()] = {
                    "size": path.stat().st_size,
                    "sha256": sha256_hex(path.read_bytes()),
                }
            (version / "manifest.json").write_text(
                json.dumps(
                    {
                        "files": files,
                        "num_images": {split: 1 for split in ("train", "val", "test")},
                    }
                ),
                encoding="utf-8",
            )
            storage = FakeStorage(root / "remote")
            cached = download_dataset(storage, "v1", root / "cache", 2)
            self.assertTrue((cached / ".complete").is_file())
            storage.downloads.clear()
            download_dataset(storage, "v1", root / "cache", 2)
            self.assertEqual(storage.downloads, ["v1/manifest.json"])
            (cached / "images/0.png").unlink()
            (cached / "images/1.png").write_bytes(b"broken")
            (cached / "stats.json").write_text("{}", encoding="utf-8")
            storage.downloads.clear()
            download_dataset(storage, "v1", root / "cache", 2)
            self.assertEqual(
                set(storage.downloads),
                {"v1/manifest.json", "v1/stats.json", "v1/images/0.png", "v1/images/1.png"},
            )
            (version / "images/2.png").write_bytes(b"remote corruption")
            (cached / "images/2.png").unlink()
            with self.assertRaises(PipelineError):
                download_dataset(storage, "v1", root / "cache", 2)
            self.assertFalse((cached / ".complete").exists())


class TrainingTests(unittest.TestCase):
    def test_evaluation_weights_uneven_batches_by_image_count(self):
        logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 5.0, 0.0]])
        labels = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        result = evaluate(nn.Identity(), loader, 3)
        expected_loss = float(nn.functional.cross_entropy(logits, labels).item())
        self.assertAlmostEqual(result.metrics.loss, expected_loss, places=6)
        self.assertAlmostEqual(result.metrics.accuracy, 2 / 3)
        np.testing.assert_array_equal(result.confusion, [[1, 1, 0], [0, 1, 0], [0, 0, 0]])

    def test_metrics_include_absent_classes_and_weight_loss_by_samples(self):
        matrix = np.array([[2, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int64)
        metrics = from_confusion(8.0, matrix)
        self.assertAlmostEqual(metrics.loss, 2.0)
        self.assertAlmostEqual(metrics.accuracy, 0.75)
        self.assertAlmostEqual(metrics.macro_f1, (0.8 + 2 / 3) / 3)

    def test_both_models_accept_cifar_images(self):
        for config in (ModelConfig(type="linear", params={}), ModelConfig()):
            self.assertEqual(build_model(config, 10)(torch.zeros(2, 3, 32, 32)).shape, (2, 10))

    def test_best_checkpoint_is_restored_before_single_test_evaluation(self):
        train_loader = DataLoader(
            TensorDataset(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long))
        )
        val_loader = DataLoader(TensorDataset(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)))
        test_loader = DataLoader(
            TensorDataset(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long))
        )
        model = nn.Linear(1, 2)
        epochs = []
        evaluated = []

        def fake_train(model, loader, optimizer):
            epochs.append(len(epochs) + 1)
            with torch.no_grad():
                model.weight.fill_(len(epochs))

        def fake_evaluate(model, loader, num_classes):
            evaluated.append(loader)
            if loader is test_loader or len(evaluated) > 3:
                self.assertEqual(float(model.weight[0, 0].detach()), 1.0)
            loss = [0.2, 0.5, 0.5][min(len(evaluated) - 1, 2)]
            return Evaluation(Metrics(loss, 0.5, 0.5), np.eye(2, dtype=np.int64))

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            with (
                patch("ml_product_management.training.engine.train_epoch", side_effect=fake_train),
                patch("ml_product_management.training.engine.evaluate", side_effect=fake_evaluate),
            ):
                result = fit(
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    TrainConfig(epochs=3),
                    checkpoint,
                    {"classes": ["a", "b"]},
                    lambda epoch, metrics: None,
                )
            self.assertEqual(result.best_epoch, 1)
            self.assertEqual(sum(loader is test_loader for loader in evaluated), 1)
            self.assertIs(evaluated[-1], test_loader)
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved["best_epoch"], 1)

    def test_equal_val_loss_keeps_earliest_checkpoint(self):
        loader = DataLoader(TensorDataset(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)))
        evaluation = Evaluation(Metrics(0.5, 0.5, 0.5), np.eye(2, dtype=np.int64))
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("ml_product_management.training.engine.train_epoch"),
            patch("ml_product_management.training.engine.evaluate", return_value=evaluation),
        ):
            result = fit(
                nn.Linear(1, 2),
                loader,
                loader,
                loader,
                TrainConfig(epochs=2),
                Path(temporary) / "model.pt",
                {"classes": ["a", "b"]},
                lambda epoch, metrics: None,
            )
            self.assertEqual(result.best_epoch, 1)


class OrchestrationTests(unittest.TestCase):
    def test_config_validation_and_flattening(self):
        with self.assertRaises(PipelineError):
            RunConfig.from_dict(
                {"experiment_name": "ok", "run_name": "ok", "data": {"s3_path": "C:/outside"}}
            )
        with self.assertRaises(PipelineError):
            RunConfig.from_dict({"experiment_name": "../bad", "run_name": "ok"})
        with self.assertRaises(PipelineError):
            RunConfig.from_dict({"experiment_name": "ok", "run_name": "ok", "train": {"lr": -1}})
        self.assertEqual(
            flatten_params({"model": {"params": {"channels": [32, 64]}}})["model.params.channels"],
            "[32, 64]",
        )

    def test_grid_and_container_command(self):
        config = RunConfig.load(PROJECT_DIR / "configs/simple_cnn.yaml")
        spec = yaml.safe_load((PROJECT_DIR / "configs/grid.yaml").read_text(encoding="utf-8"))
        combinations = expand_grid(spec, config)
        self.assertEqual(len(combinations), 6)
        self.assertEqual(len({(item.train.lr, item.train.batch_size) for item in combinations}), 6)
        command = container_command(
            PROJECT_DIR / "artifacts/test.yaml", Path("custom.env"), "abc", "false"
        )
        self.assertEqual(
            command[:6], ["docker", "compose", "--env-file", "custom.env", "run", "--rm"]
        )
        self.assertIn("trainer", command)
        self.assertIn("/app/artifacts/test.yaml", command)

    def test_selection_ignores_test_and_keeps_grid_order_on_ties(self):
        results = [
            {"val": {"loss": loss}, "test": {"accuracy": accuracy}}
            for loss, accuracy in [(0.4, 1.0), (0.2, 0.0), (0.2, 0.99)]
        ]
        self.assertIs(select_best(results), results[1])

    def test_series_launches_eight_independent_runs_and_final_uses_winner(self):
        configs = []

        def fake_execute(config, directory, env_file, dry_run):
            configs.append(config)
            return {
                "role": config.role,
                "run_name": config.run_name,
                "config": copy.deepcopy(config.to_dict()),
                "val": {"loss": 0.1 if len(configs) == 3 else 0.9},
                "test": {"accuracy": 1.0 if len(configs) == 2 else 0.0},
            }

        with (
            patch(
                "ml_product_management.training.experiments.execute_run", side_effect=fake_execute
            ),
            patch("ml_product_management.training.experiments.save_summary"),
        ):
            run_series(PROJECT_DIR / "configs/grid.yaml", Path(".env"))
        self.assertEqual(len(configs), 8)
        self.assertEqual(len({item.run_name for item in configs}), 8)
        self.assertEqual({item.experiment_name for item in configs}, {"cifar10-hw3"})
        self.assertEqual(configs[-1].train.lr, configs[2].train.lr)
        self.assertEqual(configs[-1].train.batch_size, configs[2].train.batch_size)
        self.assertIsNone(configs[-1].data.max_train_samples)
        self.assertEqual(configs[-1].train.epochs, 15)

    def test_failure_stops_series(self):
        with patch(
            "ml_product_management.training.experiments.execute_run",
            side_effect=RuntimeError("failed"),
        ) as execute:
            with self.assertRaises(RuntimeError):
                run_series(PROJECT_DIR / "configs/grid.yaml", Path(".env"))
            self.assertEqual(execute.call_count, 1)


if __name__ == "__main__":
    unittest.main()
