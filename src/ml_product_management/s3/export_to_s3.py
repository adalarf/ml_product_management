import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ml_product_management.config import PipelineError, StepsConfig
from ml_product_management.data.dataset import (
    SPLITS,
    Annotation,
    ImageRecord,
    Layout,
    read_annotation,
    read_json,
    sha256_hex,
)
from ml_product_management.s3.storage import S3Storage

VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
PREVIOUS_STEPS_HINT = "сначала выполните предыдущие шаги пайплайна"
CODE_DIR = Path(__file__).resolve().parent


def git_info() -> dict[str, Any]:
    def run_git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True, cwd=CODE_DIR
        ).stdout.strip()

    try:
        return {
            "commit": run_git("rev-parse", "HEAD"),
            "dirty": bool(run_git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        raise PipelineError(f"Не удалось определить git-коммит: {error}") from error


def file_info(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"size": len(data), "sha256": sha256_hex(data)}


def collect_images(layout: Layout, annotations: dict[str, Annotation]) -> list[ImageRecord]:
    images = [image for annotation in annotations.values() for image in annotation["images"]]
    ids = [image["image_id"] for image in images]
    if len(set(ids)) != len(ids):
        raise PipelineError("Один и тот же image_id встречается в нескольких выборках")
    on_disk = len(list(layout.images_dir.glob("*.png")))
    if on_disk != len(images):
        raise PipelineError(f"В images/ {on_disk} файлов, а в разметке {len(images)} изображений")
    return images


def upload_image(storage: S3Storage, layout: Layout, prefix: str, image: ImageRecord) -> None:
    data = (layout.dataset_dir / image["file_name"]).read_bytes()
    if sha256_hex(data) != image["sha256"]:
        raise PipelineError(f"Хеш файла {image['file_name']} не совпал с разметкой")
    storage.put_bytes(prefix + image["file_name"], data, "image/png")


def export(
    storage: S3Storage, layout: Layout, version: str, dataset_prefix: str, workers: int
) -> str:
    if not VERSION_PATTERN.fullmatch(version):
        raise PipelineError(f"Недопустимое имя версии: {version!r}")

    prefix = f"{dataset_prefix.strip('/')}/{version}/"
    if storage.prefix_exists(prefix):
        reason = (
            "уже существует"
            if storage.exists(prefix + "manifest.json")
            else "уже содержит объекты (выгрузка не завершена?)"
        )
        raise PipelineError(
            f"Версия {storage.uri(prefix)} {reason}. Версии неизменяемы: укажите новую версию "
            f"(export.version в configs/s3.yaml, например v2) или удалите префикс вручную."
        )

    annotations = {
        split: read_annotation(layout.annotation_path(split), PREVIOUS_STEPS_HINT)
        for split in SPLITS
    }
    images = collect_images(layout, annotations)
    params = read_json(layout.partition_params_path, PREVIOUS_STEPS_HINT)
    if len(annotations["val"]["images"]) != params["val_size"]:
        raise PipelineError(
            "Размер val.json не соответствует partition_params.json, повторите шаг partition"
        )

    metadata_files = [f"annotations/{split}.json" for split in SPLITS] + ["stats.json"]
    manifest = {
        "dataset": "cifar10",
        "version": version,
        "num_images": {split: len(annotations[split]["images"]) for split in SPLITS},
        "files": {name: file_info(layout.dataset_dir / name) for name in metadata_files},
        "processing": {"val_size": params["val_size"], "seed": params["seed"]},
        "git": git_info(),
    }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(upload_image, storage, layout, prefix, image) for image in images]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    for name in metadata_files:
        data = (layout.dataset_dir / name).read_bytes()
        storage.put_bytes(prefix + name, data, "application/json")
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    storage.put_bytes(prefix + "manifest.json", manifest_bytes, "application/json")
    return prefix


def main() -> None:
    params = StepsConfig.load().export
    try:
        storage = S3Storage.from_env(params.workers)
        export(storage, Layout.default(), params.version, params.prefix, params.workers)
    except PipelineError as error:
        raise SystemExit(str(error)) from None
