from pathlib import Path

from ml_product_management.config import ARCHIVE_NAME, StepsConfig
from ml_product_management.data.dataset import Layout
from ml_product_management.s3.storage import S3Storage


def fetch(storage: S3Storage, object_key: str, destination: Path) -> bool:
    if destination.exists():
        return False

    storage.download_file(object_key, destination)
    return True


def main() -> None:
    destination = Layout.default().raw_dir / ARCHIVE_NAME
    fetch(S3Storage.from_env(), StepsConfig.load().raw_object_key, destination)
