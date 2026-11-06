import io
from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from ml_product_management.config import PipelineError
from ml_product_management.data.dataset import (
    ImageRecord,
    Layout,
    read_annotation,
    sha256_hex,
    write_json,
)

CHANNELS = ("R", "G", "B")
MAX_PIXEL_VALUE = 255


def channel_stats(images: Iterable[npt.NDArray[np.uint8]]) -> tuple[list[float], list[float], int]:
    total = np.zeros(len(CHANNELS), dtype=np.int64)
    total_squares = np.zeros(len(CHANNELS), dtype=np.int64)
    pixels = 0
    count = 0
    for image in images:
        values = image.astype(np.int64).reshape(-1, len(CHANNELS))
        total += values.sum(axis=0)
        total_squares += (values * values).sum(axis=0)
        pixels += len(values)
        count += 1
    if count == 0:
        raise PipelineError("В train нет изображений")

    mean = total / pixels
    variance = total_squares / pixels - mean**2
    return (
        (mean / MAX_PIXEL_VALUE).tolist(),
        (np.sqrt(variance) / MAX_PIXEL_VALUE).tolist(),
        count,
    )


def load_images(layout: Layout, records: list[ImageRecord]) -> Iterable[npt.NDArray[np.uint8]]:
    for record in records:
        png = (layout.dataset_dir / record["file_name"]).read_bytes()
        if sha256_hex(png) != record["sha256"]:
            raise PipelineError(f"Хеш файла {record['file_name']} не совпал с разметкой")
        yield np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint8)


def compute_stats(layout: Layout) -> dict[str, Any]:
    annotation = read_annotation(layout.annotation_path("train"), "сначала выполните шаг convert")
    mean, std, count = channel_stats(load_images(layout, annotation["images"]))

    stats = {
        "source": "train",
        "num_images": count,
        "channels": list(CHANNELS),
        "pixel_range": [0.0, 1.0],
        "mean": mean,
        "std": std,
    }
    write_json(layout.stats_path, stats)
    return stats


def main() -> None:
    compute_stats(Layout.default())
