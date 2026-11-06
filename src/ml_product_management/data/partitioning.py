"""Шаг partition: стратифицированное выделение валидации из train.

Исходный train восстанавливается как объединение train.json и val.json, поэтому запуск с тем
же seed и размером валидации всегда даёт те же файлы, а с другими параметрами разбивает
заново тот же исходный набор. Картинки и test.json не затрагиваются.
"""

import random
from collections import defaultdict

from ml_product_management.config import PipelineError, StepsConfig
from ml_product_management.data.dataset import (
    ImageRecord,
    Layout,
    read_annotation,
    write_annotation,
    write_json,
)

MISSING_TRAIN_HINT = "сначала выполните шаг convert"


def allocate_val_sizes(class_sizes: dict[int, int], val_size: int) -> dict[int, int]:
    total = sum(class_sizes.values())
    quotas = {label: val_size * size / total for label, size in class_sizes.items()}
    allocation = {label: int(quota) for label, quota in quotas.items()}
    leftover = val_size - sum(allocation.values())
    by_remainder = sorted(quotas, key=lambda label: (allocation[label] - quotas[label], label))
    for label in by_remainder[:leftover]:
        allocation[label] += 1
    return allocation


def split(
    images: list[ImageRecord], val_size: int, seed: int
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    by_class: dict[int, list[ImageRecord]] = defaultdict(list)
    for image in sorted(images, key=lambda item: item["image_id"]):
        by_class[image["label"]].append(image)

    allocation = allocate_val_sizes(
        {label: len(items) for label, items in by_class.items()}, val_size
    )
    rng = random.Random(seed)
    train: list[ImageRecord] = []
    val: list[ImageRecord] = []
    for label in sorted(by_class):
        items = by_class[label]
        rng.shuffle(items)
        val.extend(items[: allocation[label]])
        train.extend(items[allocation[label] :])

    train.sort(key=lambda item: item["image_id"])
    val.sort(key=lambda item: item["image_id"])
    return train, val


def partition(layout: Layout, val_size: int, seed: int) -> tuple[int, int]:
    train_path = layout.annotation_path("train")
    val_path = layout.annotation_path("val")

    annotation = read_annotation(train_path, MISSING_TRAIN_HINT)
    classes = annotation["classes"]
    images = annotation["images"]
    if val_path.exists():
        val_annotation = read_annotation(val_path, MISSING_TRAIN_HINT)
        if val_annotation["classes"] != classes:
            raise PipelineError("Списки классов в train.json и val.json различаются")
        images = images + val_annotation["images"]

    if not 0 < val_size < len(images):
        raise PipelineError(
            f"Размер валидации должен быть от 1 до {len(images) - 1}, получено {val_size}"
        )

    train, val = split(images, val_size, seed)
    write_annotation(train_path, {"classes": classes, "images": train})
    write_annotation(val_path, {"classes": classes, "images": val})
    write_json(layout.partition_params_path, {"val_size": val_size, "seed": seed})
    return len(train), len(val)


def main() -> None:
    params = StepsConfig.load().partitioning
    partition(Layout.default(), params.val_size, params.seed)
