.PHONY: setup check s3-up s3-down data ingest download-from-s3 convert-to-safe partitioning stats export-to-s3

ENV_FILE ?= .env
CONFIG ?= configs/grid.yaml
ARGS ?=
GIT_SHA := $(shell git rev-parse HEAD)
export GIT_SHA

.PHONY: infra-up infra-down train experiments test

infra-up:
	docker compose --env-file $(ENV_FILE) build mlflow trainer
	docker compose --env-file $(ENV_FILE) up -d --wait mlflow

infra-down:
	docker compose --env-file $(ENV_FILE) down

train:
	uv run --locked mlpm-experiments --single --config $(CONFIG) --env-file $(ENV_FILE) $(ARGS)

experiments:
	uv run --locked mlpm-experiments --config $(CONFIG) --env-file $(ENV_FILE) $(ARGS)

test:
	uv run --locked python -m unittest discover -s tests -v

s3-up:
	docker compose --env-file $(ENV_FILE) up -d --wait rustfs
	docker compose --env-file $(ENV_FILE) run --rm create-bucket

s3-down:
	docker compose --env-file $(ENV_FILE) stop rustfs

data: download-from-s3 convert-to-safe partitioning stats export-to-s3
	@echo "processing pipeline is done"

ingest:
	uv run --locked --env-file $(ENV_FILE) mlpm-ingest

download-from-s3:
	uv run --locked --env-file $(ENV_FILE) mlpm-download-from-s3

convert-to-safe:
	uv run --locked mlpm-convert-to-safe

partitioning:
	uv run --locked mlpm-partitioning

stats:
	uv run --locked mlpm-stats

export-to-s3:
	uv run --locked --env-file $(ENV_FILE) mlpm-export-to-s3

setup:
	uv sync --locked --all-groups
	uv run pre-commit install

check:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked mypy src
