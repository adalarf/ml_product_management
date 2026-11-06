import hashlib
import io
import pickle
import shutil
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy._core.multiarray import _reconstruct
from PIL import Image

from ml_product_management.config import ARCHIVE_MD5, ARCHIVE_NAME, PipelineError
from ml_product_management.data.dataset import (
    Annotation,
    ImageRecord,
    Layout,
    sha256_hex,
    write_annotation,
)

ARCHIVE_ROOT = "cifar-10-batches-py"
META_FILE = "batches.meta"
SOURCE_SPLITS = {
    "train": [f"data_batch_{number}" for number in range(1, 6)],
    "test": ["test_batch"],
}
EXPECTED_SIZES = {"train": 50_000, "test": 10_000}

IMAGE_SIDE = 32
IMAGE_CHANNELS = 3
HASH_CHUNK_SIZE = 1024 * 1024

ALLOWED_GLOBALS: dict[tuple[str, str], Any] = {
    ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return ALLOWED_GLOBALS[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"Запрещённый объект в pickle: {module}.{name}") from None


def load_pickle(data: bytes) -> dict[bytes, Any]:
    loaded = RestrictedUnpickler(io.BytesIO(data), encoding="bytes").load()
    if not isinstance(loaded, dict):
        raise pickle.UnpicklingError("Ожидался словарь")
    return loaded


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        while chunk := file.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def read_member(archive: tarfile.TarFile, name: str) -> bytes:
    file = archive.extractfile(f"{ARCHIVE_ROOT}/{name}")
    if file is None:
        raise PipelineError(f"В архиве нет файла {name}")
    with file:
        return file.read()


def encode_png(image: npt.NDArray[np.uint8]) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def convert_split(
    split: str, archive: tarfile.TarFile, classes: list[str], images_dir: Path
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for batch_name in SOURCE_SPLITS[split]:
        batch = load_pickle(read_member(archive, batch_name))
        data, labels = batch[b"data"], batch[b"labels"]
        if (
            not isinstance(data, np.ndarray)
            or data.dtype != np.uint8
            or data.ndim != 2
            or data.shape[1] != IMAGE_CHANNELS * IMAGE_SIDE * IMAGE_SIDE
            or len(labels) != len(data)
        ):
            raise PipelineError(f"Неожиданный формат данных в {batch_name}")

        images = data.reshape(-1, IMAGE_CHANNELS, IMAGE_SIDE, IMAGE_SIDE).transpose(0, 2, 3, 1)

        for image, label in zip(images, labels, strict=True):
            if not 0 <= label < len(classes):
                raise PipelineError(f"Неожиданная метка {label} в {batch_name}")
            image_id = f"{split}_{len(records):05d}"
            png = encode_png(image)
            (images_dir / f"{image_id}.png").write_bytes(png)
            records.append(
                {
                    "image_id": image_id,
                    "file_name": f"images/{image_id}.png",
                    "label": int(label),
                    "label_name": classes[label],
                    "sha256": sha256_hex(png),
                }
            )

    if len(records) != EXPECTED_SIZES[split]:
        raise PipelineError(
            f"В выборке {split} {len(records)} изображений вместо {EXPECTED_SIZES[split]}"
        )
    return records


def convert(archive_path: Path, layout: Layout, expected_md5: str) -> bool:
    output_dir = layout.dataset_dir
    if layout.annotation_path("train").exists():
        return False
    if not archive_path.exists():
        raise PipelineError(f"Нет архива {archive_path}, сначала выполните шаг fetch")

    md5 = file_md5(archive_path)
    if md5 != expected_md5:
        raise PipelineError(
            f"Контрольная сумма архива не совпала: ожидалась {expected_md5}, получена {md5}"
        )

    tmp_dir = output_dir.with_name(output_dir.name + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    (tmp_dir / "images").mkdir(parents=True)
    (tmp_dir / "annotations").mkdir()

    with tarfile.open(archive_path, "r:gz") as archive:
        meta = load_pickle(read_member(archive, META_FILE))
        classes = [name.decode() for name in meta[b"label_names"]]
        for split in SOURCE_SPLITS:
            images = convert_split(split, archive, classes, tmp_dir / "images")
            annotation: Annotation = {"classes": classes, "images": images}
            write_annotation(tmp_dir / "annotations" / f"{split}.json", annotation)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.replace(output_dir)
    return True


def main() -> None:
    layout = Layout.default()
    convert(layout.raw_dir / ARCHIVE_NAME, layout, ARCHIVE_MD5)
