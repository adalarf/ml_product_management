import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from ml_product_management.config import DATA_DIR, PipelineError

SPLITS = ("train", "val", "test")


class ImageRecord(TypedDict):
    image_id: str
    file_name: str
    label: int
    label_name: str
    sha256: str


class Annotation(TypedDict):
    classes: list[str]
    images: list[ImageRecord]


@dataclass(frozen=True)
class Layout:
    data_dir: Path

    @classmethod
    def default(cls) -> "Layout":
        return cls(DATA_DIR)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / "interim" / "cifar10"

    @property
    def images_dir(self) -> Path:
        return self.dataset_dir / "images"

    @property
    def annotations_dir(self) -> Path:
        return self.dataset_dir / "annotations"

    @property
    def stats_path(self) -> Path:
        return self.dataset_dir / "stats.json"

    @property
    def partition_params_path(self) -> Path:
        return self.dataset_dir / "partition_params.json"

    def annotation_path(self, split: str) -> Path:
        return self.annotations_dir / f"{split}.json"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path, hint: str) -> Any:
    if not path.exists():
        raise PipelineError(f"Нет файла {path}, {hint}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_annotation(path: Path, hint: str) -> Annotation:
    annotation: Annotation = read_json(path, hint)
    return annotation


def write_annotation(path: Path, annotation: Annotation) -> None:
    path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
