import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from ml_product_management.config import PipelineError

NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def safe_name(value: str) -> str:
    if not NAME_PATTERN.fullmatch(value):
        raise PipelineError(f"Недопустимое имя: {value!r}")
    return value


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise PipelineError(f"Ожидается относительный S3-путь: {value!r}")
    return str(path)


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise PipelineError(f"Не удалось прочитать конфиг {path}: {error}") from error
    if not isinstance(data, dict):
        raise PipelineError(f"Конфиг {path} должен содержать YAML mapping")
    return {str(key): value for key, value in data.items()}


@dataclass(frozen=True)
class DataConfig:
    s3_path: str = "processed/cifar10/v1"
    max_train_samples: int | None = 10000
    workers: int = 16


@dataclass(frozen=True)
class ModelConfig:
    type: str = "simple_cnn"
    params: dict[str, Any] = field(default_factory=lambda: {"channels": [32, 64], "hidden": 128})


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    batch_size: int = 128
    lr: float = 0.001
    cpu_threads: int = 4
    num_workers: int = 0


@dataclass(frozen=True)
class RunConfig:
    experiment_name: str
    run_name: str
    seed: int = 42
    role: str = "single"
    series_id: str = "standalone"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        try:
            config = cls(
                **{
                    **data,
                    "data": DataConfig(**data.get("data", {})),
                    "model": ModelConfig(**data.get("model", {})),
                    "train": TrainConfig(**data.get("train", {})),
                }
            )
            config.validate()
        except (TypeError, ValueError, AttributeError) as error:
            raise PipelineError(f"Некорректный конфиг запуска: {error}") from error
        return config

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        return cls.from_dict(read_yaml(path))

    def validate(self) -> None:
        safe_name(self.experiment_name)
        safe_name(self.run_name)
        safe_name(self.series_id)
        safe_relative(self.data.s3_path)
        if self.role not in {"single", "baseline", "grid", "final"}:
            raise PipelineError("Неизвестная роль запуска")
        if not 0 <= self.seed < 2**32:
            raise PipelineError("seed должен быть в диапазоне [0, 2**32)")
        positive = [
            self.train.epochs,
            self.train.batch_size,
            self.train.cpu_threads,
            self.data.workers,
        ]
        if self.data.max_train_samples is not None:
            positive.append(self.data.max_train_samples)
        if any(not isinstance(value, int) or value <= 0 for value in positive):
            raise PipelineError(
                "epochs, batch_size, cpu_threads, workers и лимит train должны быть > 0"
            )
        if self.train.num_workers < 0 or not math.isfinite(self.train.lr) or self.train.lr <= 0:
            raise PipelineError("num_workers должен быть >= 0, lr — конечным числом > 0")
        if self.model.type not in {"linear", "simple_cnn"}:
            raise PipelineError("model.type должен быть linear или simple_cnn")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


def flatten_params(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            result.update(flatten_params(value, name))
        else:
            result[name] = (
                yaml.safe_dump(value, default_flow_style=True).removesuffix("...\n").strip()
            )
    return result
