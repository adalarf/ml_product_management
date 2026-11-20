import argparse
import copy
import csv
import itertools
import json
import os
import shlex
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml_product_management.config import PROJECT_DIR, PipelineError
from ml_product_management.training.config import RunConfig, read_yaml


def override(data: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    target = data
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise PipelineError(f"Неизвестный параметр сетки: {key}")
        target = target[part]
    if parts[-1] not in target:
        raise PipelineError(f"Неизвестный параметр сетки: {key}")
    target[parts[-1]] = value


def expand_grid(spec: dict[str, Any], base: RunConfig) -> list[RunConfig]:
    grid = spec.get("grid")
    if not isinstance(grid, dict) or not grid:
        raise PipelineError("grid должен быть непустым mapping")
    keys = list(grid)
    if any(not isinstance(values, list) or not values for values in grid.values()):
        raise PipelineError("Каждый параметр grid должен содержать непустой список")
    combinations = []
    for values in itertools.product(*(grid[key] for key in keys)):
        data = copy.deepcopy(base.to_dict())
        for key, value in zip(keys, values, strict=True):
            override(data, key, value)
        combinations.append(RunConfig.from_dict(data))
    return combinations


def named_run(config: RunConfig, role: str, series: str, index: int = 0) -> RunConfig:
    data = config.to_dict()
    data.update(
        role=role,
        series_id=series,
        run_name=(
            f"{role}-{index:02d}-{config.model.type}-lr{config.train.lr:g}"
            f"-bs{config.train.batch_size}-seed{config.seed}-{series}"
        ),
    )
    return RunConfig.from_dict(data)


def select_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise PipelineError("Нет успешных запусков сетки")
    return min(results, key=lambda result: float(result["val"]["loss"]))


def container_command(config_path: Path, env_file: Path, commit: str, dirty: str) -> list[str]:
    relative = config_path.resolve().relative_to(PROJECT_DIR.resolve()).as_posix()
    return [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "run",
        "--rm",
        "--no-deps",
        "-e",
        f"MLPM_EXPECTED_GIT_SHA={commit}",
        "-e",
        f"MLPM_GIT_DIRTY={dirty}",
        "trainer",
        "--config",
        f"/app/{relative}",
    ]


def execute_run(
    config: RunConfig, directory: Path, env_file: Path, dry_run: bool
) -> dict[str, Any]:
    path = directory / "configs" / f"{config.run_name}.yaml"
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True
    ).strip()
    dirty = str(
        bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_DIR, text=True
            ).strip()
        )
    ).lower()
    command = container_command(path, env_file, commit, dirty)
    print(shlex.join(command), flush=True)
    if dry_run:
        print(json.dumps(config.to_dict(), ensure_ascii=False), flush=True)
        return {}
    config.save(path)
    subprocess.run(command, check=True, cwd=PROJECT_DIR)
    result_path = (
        PROJECT_DIR / "artifacts" / config.experiment_name / config.run_name / "metrics.json"
    )
    result: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    if result["run_name"] != config.run_name:
        raise PipelineError("Контейнер вернул результат другого запуска")
    return result


def save_summary(directory: Path, results: list[dict[str, Any]], selected: str | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(
        json.dumps({"selected_grid_run": selected, "runs": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = ["role", "run_name", "lr", "batch_size", "best_epoch", "training_seconds"]
    fields += [
        f"{split}_{metric}"
        for split in ("val", "test")
        for metric in ("loss", "accuracy", "macro_f1")
    ]
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    **{
                        key: result[key]
                        for key in ("role", "run_name", "best_epoch", "training_seconds")
                    },
                    "lr": result["config"]["train"]["lr"],
                    "batch_size": result["config"]["train"]["batch_size"],
                    **{
                        f"{split}_{metric}": value
                        for split in ("val", "test")
                        for metric, value in result[split].items()
                    },
                }
            )


def run_series(
    config_path: Path, env_file: Path, dry_run: bool = False, single: bool = False
) -> None:
    series = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = PROJECT_DIR / "artifacts" / "series" / series
    if single:
        config = named_run(RunConfig.load(config_path), "single", series)
        execute_run(config, directory, env_file, dry_run)
        return
    spec = read_yaml(config_path)
    base = RunConfig.load(config_path.parent / spec["base"])
    baseline = RunConfig.load(config_path.parent / spec["baseline"])
    if baseline.experiment_name != base.experiment_name:
        raise PipelineError("Baseline и сетка должны иметь одинаковый experiment_name")
    if (baseline.data, baseline.seed, baseline.train.epochs) != (
        base.data,
        base.seed,
        base.train.epochs,
    ):
        raise PipelineError("Baseline и сетка должны иметь одинаковые данные, seed и число эпох")
    grid = expand_grid(spec, base)
    for item in grid:
        if item.experiment_name != base.experiment_name:
            raise PipelineError("Все комбинации должны иметь одинаковый experiment_name")
        if (item.data, item.seed, item.train.epochs) != (base.data, base.seed, base.train.epochs):
            raise PipelineError(
                "Сетка не должна менять данные, seed и число эпох сравниваемых запусков"
            )
    configs = [named_run(baseline, "baseline", series)]
    configs += [named_run(item, "grid", series, index) for index, item in enumerate(grid, start=1)]
    final_overrides = spec.get("final", {"data.max_train_samples": None, "train.epochs": 15})
    # Validate final overrides before starting any containers.
    final_data = base.to_dict()
    for key, value in final_overrides.items():
        override(final_data, key, value)
    RunConfig.from_dict(final_data)
    results = []
    for config in configs:
        result = execute_run(config, directory, env_file, dry_run)
        if not dry_run:
            results.append(result)
            save_summary(directory, results, None)
    if dry_run:
        print("final: победитель по val.loss + " + json.dumps(final_overrides), flush=True)
        return
    selected = select_best([result for result in results if result["role"] == "grid"])
    final_data = copy.deepcopy(selected["config"])
    for key, value in final_overrides.items():
        override(final_data, key, value)
    final_config = named_run(RunConfig.from_dict(final_data), "final", series)
    save_summary(directory, results, selected["run_name"])
    results.append(execute_run(final_config, directory, env_file, False))
    save_summary(directory, results, selected["run_name"])
    print(f"Сводка: {directory / 'summary.csv'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Серия экспериментов: отдельный контейнер на запуск"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(os.environ.get("ENV_FILE", ".env")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    try:
        run_series(args.config, args.env_file, args.dry_run, args.single)
    except (PipelineError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
