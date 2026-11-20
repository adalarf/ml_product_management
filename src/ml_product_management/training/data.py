import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ml_product_management.config import PipelineError
from ml_product_management.data.dataset import (
    SPLITS,
    Annotation,
    ImageRecord,
    read_annotation,
    sha256_hex,
)
from ml_product_management.s3.storage import S3Storage
from ml_product_management.training.config import safe_relative


def valid_file(path: Path, digest: str, size: int | None = None) -> bool:
    if not path.is_file() or (size is not None and path.stat().st_size != size):
        return False
    return sha256_hex(path.read_bytes()) == digest


def fetch_checked(
    storage: S3Storage,
    prefix: str,
    root: Path,
    name: str,
    digest: str,
    size: int | None = None,
) -> None:
    path = root / safe_relative(name)
    if not valid_file(path, digest, size):
        storage.download_file(prefix + name, path)
        if not valid_file(path, digest, size):
            raise PipelineError(f"Хеш или размер {storage.uri(prefix + name)} не совпал")


def download_dataset(storage: S3Storage, prefix: str, cache: Path, workers: int) -> Path:
    prefix = safe_relative(prefix).rstrip("/") + "/"
    # Include endpoint identity: two independent S3 servers may use the same bucket/version.
    identity = hashlib.sha256(
        f"{storage.endpoint_url}|{storage.uri(prefix)}".encode()
    ).hexdigest()[:16]
    root = cache / identity / prefix
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".complete"
    marker.unlink(missing_ok=True)
    manifest_path = root / "manifest.json"
    storage.download_file(prefix + "manifest.json", manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in [*(f"annotations/{split}.json" for split in SPLITS), "stats.json"]:
        info = manifest["files"][name]
        fetch_checked(storage, prefix, root, name, info["sha256"], info["size"])
    annotations = {
        split: read_annotation(root / f"annotations/{split}.json", "") for split in SPLITS
    }
    classes = annotations["train"]["classes"]
    if not classes:
        raise PipelineError("Пустой список классов")
    records: list[ImageRecord] = []
    for split, annotation in annotations.items():
        if annotation["classes"] != classes:
            raise PipelineError("Классы в разметке выборок различаются")
        if not annotation["images"] or len(annotation["images"]) != manifest["num_images"][split]:
            raise PipelineError(f"Число изображений {split} не соответствует manifest")
        records.extend(annotation["images"])
    ids = [record["image_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise PipelineError("Выборки содержат повторяющиеся image_id")

    def fetch_image(record: ImageRecord) -> None:
        if not 0 <= record["label"] < len(classes):
            raise PipelineError("Метка изображения вне диапазона классов")
        fetch_checked(storage, prefix, root, record["file_name"], record["sha256"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(fetch_image, records):
            pass
    marker.write_text(sha256_hex(manifest_path.read_bytes()), encoding="utf-8")
    return root


def stratified_subset(
    records: list[ImageRecord], limit: int | None, seed: int
) -> list[ImageRecord]:
    if limit is None or limit >= len(records):
        return list(records)
    groups: dict[int, list[ImageRecord]] = {}
    for record in records:
        groups.setdefault(record["label"], []).append(record)
    labels = sorted(groups)
    if limit < len(labels):
        raise PipelineError("max_train_samples должен быть не меньше числа классов")
    # Largest-remainder allocation preserves class proportions and the exact requested size.
    quotas = {label: limit * len(groups[label]) / len(records) for label in labels}
    counts = {label: int(quotas[label]) for label in labels}
    remainder = limit - sum(counts.values())
    for label in sorted(labels, key=lambda item: (-(quotas[item] - counts[item]), item))[
        :remainder
    ]:
        counts[label] += 1
    rng = np.random.default_rng(seed)
    selected: set[str] = set()
    for label in labels:
        indices = rng.permutation(len(groups[label]))[: counts[label]]
        selected.update(groups[label][int(index)]["image_id"] for index in indices)
    return [record for record in records if record["image_id"] in selected]


def load_stats(root: Path) -> dict[str, Any]:
    stats: dict[str, Any] = json.loads((root / "stats.json").read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if (
        stats.get("source") != "train"
        or mean.shape != (3,)
        or std.shape != (3,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or not (std > 0).all()
    ):
        raise PipelineError("stats.json должен содержать конечные mean/std трёх каналов train")
    return stats


class ImageDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, root: Path, annotation: Annotation, stats: dict[str, Any]) -> None:
        self.root = root
        self.records = annotation["images"]
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(stats["std"], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        with Image.open(self.root / safe_relative(record["file_name"])) as image:
            pixels = np.array(image.convert("RGB"), dtype=np.uint8)
        if pixels.shape != (32, 32, 3):
            raise PipelineError(f"Ожидается RGB 32x32: {record['file_name']}")
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
        return (tensor - self.mean) / self.std, record["label"]
