import socket
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ml_product_management.config import ARCHIVE_MD5, PipelineError, StepsConfig
from ml_product_management.s3.storage import S3Storage


@contextmanager
def socket_timeout(seconds: float) -> Iterator[None]:
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def download(url: str, destination: Path, md5: str, timeout: float) -> None:
    # torch подгружается долго, а нужен только этому шагу
    from torchvision.datasets.utils import download_url

    try:
        with socket_timeout(timeout):
            download_url(url, str(destination.parent), filename=destination.name, md5=md5)
    except (RuntimeError, OSError) as error:
        raise PipelineError(
            f"Не удалось скачать {url} или не совпал MD5 {md5}: {error}"
        ) from error


def ingest(storage: S3Storage, url: str, md5: str, object_key: str, timeout: float) -> bool:
    if storage.exists(object_key):
        return False

    with tempfile.TemporaryDirectory() as tmp_dir:
        archive = Path(tmp_dir) / Path(object_key).name
        download(url, archive, md5, timeout)

        storage.upload_file(archive, object_key)
    return True


def main() -> None:
    config = StepsConfig.load()
    ingest(
        S3Storage.from_env(),
        config.ingest.url,
        ARCHIVE_MD5,
        config.raw_object_key,
        config.ingest.timeout,
    )
