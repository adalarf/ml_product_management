import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]
STEPS_CONFIG = PROJECT_DIR / "configs" / "s3.yaml"
DATA_DIR = PROJECT_DIR / "data"

ARCHIVE_NAME = "cifar-10-python.tar.gz"
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"


class PipelineError(Exception):
    """Ожидаемая ошибка шага пайплайна (неверные данные, версия уже есть в S3 и т. п.)."""


@dataclass(frozen=True)
class IngestConfig:
    url: str
    timeout: float


@dataclass(frozen=True)
class PartitioningConfig:
    val_size: int
    seed: int


@dataclass(frozen=True)
class ExportConfig:
    version: str
    prefix: str
    workers: int


@dataclass(frozen=True)
class StepsConfig:
    raw_object_key: str
    ingest: IngestConfig
    partitioning: PartitioningConfig
    export: ExportConfig

    @classmethod
    def load(cls, path: Path = STEPS_CONFIG) -> "StepsConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        ingest, partitioning, export = data["ingest"], data["partitioning"], data["export"]
        return cls(
            raw_object_key=str(data["raw_object_key"]),
            ingest=IngestConfig(url=str(ingest["url"]), timeout=float(ingest["timeout"])),
            partitioning=PartitioningConfig(
                val_size=int(partitioning["val_size"]), seed=int(partitioning["seed"])
            ),
            export=ExportConfig(
                version=str(export["version"]),
                prefix=str(export["prefix"]),
                workers=int(export["workers"]),
            ),
        )


@dataclass(frozen=True)
class S3Settings:
    endpoint_url: str
    access_key: str
    secret_key: str
    bucket: str
    region: str = "us-east-1"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "S3Settings":
        def require(name: str) -> str:
            value = env.get(name)
            if not value:
                raise PipelineError(f"Переменная окружения {name} не задана (см. .env.example)")
            return value

        return cls(
            endpoint_url=require("S3_ENDPOINT_URL"),
            access_key=require("S3_ACCESS_KEY"),
            secret_key=require("S3_SECRET_KEY"),
            bucket=require("S3_BUCKET"),
            region=env.get("S3_REGION") or "us-east-1",
        )
