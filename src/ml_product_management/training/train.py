import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlflow
import torch
from torch.utils.data import DataLoader

from ml_product_management.config import PipelineError
from ml_product_management.data.dataset import SPLITS, read_annotation
from ml_product_management.s3.storage import S3Storage
from ml_product_management.training.config import RunConfig
from ml_product_management.training.data import (
    ImageDataset,
    download_dataset,
    load_stats,
    stratified_subset,
)
from ml_product_management.training.engine import Loader, fit, seed_everything
from ml_product_management.training.metrics import save_confusion
from ml_product_management.training.models import build_model
from ml_product_management.training.tracking import git_provenance, log_config, log_epoch


def run(config: RunConfig) -> dict[str, Any]:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise PipelineError("Задайте MLFLOW_TRACKING_URI (см. .env.example)")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=config.run_name) as active:
        provenance = git_provenance()
        storage = S3Storage.from_env(config.data.workers)
        model_prefix = f"models/{config.experiment_name}/{config.run_name}/"
        log_config(config, storage.uri(config.data.s3_path), provenance)
        if storage.prefix_exists(model_prefix):
            raise PipelineError(
                f"Результаты {storage.uri(model_prefix)} уже существуют: смените run_name"
            )
        output = (
            Path(os.environ.get("MLPM_OUTPUT_DIR", "artifacts"))
            / config.experiment_name
            / config.run_name
        )
        output.mkdir(parents=True, exist_ok=True)
        root = download_dataset(
            storage,
            config.data.s3_path,
            Path(os.environ.get("MLPM_CACHE_DIR", "data/processed/cache")),
            config.data.workers,
        )
        stats = load_stats(root)
        annotations = {
            split: read_annotation(root / f"annotations/{split}.json", "") for split in SPLITS
        }
        annotations["train"]["images"] = stratified_subset(
            annotations["train"]["images"], config.data.max_train_samples, config.seed
        )
        classes = annotations["train"]["classes"]
        seed_everything(config.seed, config.train.cpu_threads)
        generator = torch.Generator().manual_seed(config.seed)
        loaders: dict[str, Loader] = {
            split: DataLoader(
                ImageDataset(root, annotations[split], stats),
                batch_size=config.train.batch_size,
                shuffle=split == "train",
                num_workers=config.train.num_workers,
                generator=generator,
            )
            for split in SPLITS
        }
        config.save(output / "config.yaml")
        mlflow.log_artifact(str(output / "config.yaml"))
        mlflow.log_artifact(str(root / "manifest.json"))
        mlflow.log_param("data.actual_train_samples", len(annotations["train"]["images"]))
        model = build_model(config.model, len(classes))
        metadata = {
            "model": config.to_dict()["model"],
            "classes": classes,
            "normalization": stats,
            "config": config.to_dict(),
            "git": provenance,
            "data_uri": storage.uri(config.data.s3_path),
        }
        result = fit(
            model,
            loaders["train"],
            loaders["val"],
            loaders["test"],
            config.train,
            output / "model.pt",
            metadata,
            log_epoch,
        )
        metrics: dict[str, Any] = {
            "run_id": active.info.run_id,
            "experiment_name": config.experiment_name,
            "run_name": config.run_name,
            "role": config.role,
            "series_id": config.series_id,
            "git": provenance,
            "best_epoch": result.best_epoch,
            "training_seconds": result.training_seconds,
            "val": result.val.metrics.to_dict(),
            "test": result.test.metrics.to_dict(),
            "model_uri": storage.uri(model_prefix),
            "config": config.to_dict(),
        }
        mlflow.log_metrics(
            {
                "best_epoch": result.best_epoch,
                "training_seconds": result.training_seconds,
                **{f"best_val.{key}": value for key, value in metrics["val"].items()},
                **{f"test.{key}": value for key, value in metrics["test"].items()},
            }
        )
        save_confusion(output / "val_confusion.csv", result.val.confusion, classes)
        mlflow.log_artifact(str(output / "val_confusion.csv"))
        # metrics.json is written last in S3 and serves as the completion marker.
        metrics_path = output / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        for name in ("model.pt", "config.yaml", "metrics.json"):
            mlflow.log_artifact(str(output / name))
            storage.upload_file(output / name, model_prefix + name)
        print(json.dumps(metrics, indent=2), flush=True)
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Один CPU-запуск CIFAR из S3 с трекингом MLflow")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(RunConfig.load(args.config))
    except (PipelineError, OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
