import os
import subprocess

import mlflow

from ml_product_management.config import PROJECT_DIR, PipelineError
from ml_product_management.training.config import RunConfig, flatten_params
from ml_product_management.training.metrics import Metrics


def git_provenance() -> dict[str, str]:
    commit = os.environ.get("MLPM_GIT_SHA")
    dirty = os.environ.get("MLPM_GIT_DIRTY")
    if not commit:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True
        ).strip()
    if dirty is None:
        dirty = str(
            bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=PROJECT_DIR, text=True
                ).strip()
            )
        ).lower()
    expected = os.environ.get("MLPM_EXPECTED_GIT_SHA", commit)
    if expected != commit:
        raise PipelineError("Образ собран на другом git-коммите; выполните make infra-up")
    if commit == "unknown":
        raise PipelineError("Не задан git SHA при сборке образа")
    return {"commit": commit, "dirty": dirty}


def log_config(config: RunConfig, data_uri: str, provenance: dict[str, str]) -> None:
    mlflow.log_params(flatten_params(config.to_dict()))
    mlflow.log_param("data.s3_uri", data_uri)
    mlflow.set_tags(
        {
            "mlflow.source.git.commit": provenance["commit"],
            "git.dirty": provenance["dirty"],
            "role": config.role,
            "series_id": config.series_id,
        }
    )


def log_epoch(epoch: int, metrics: Metrics) -> None:
    mlflow.log_metrics(
        {f"val.{key}": value for key, value in metrics.to_dict().items()}, step=epoch
    )
    print(f"epoch={epoch} val={metrics.to_dict()}", flush=True)
