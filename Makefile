.PHONY: setup check s3-up s3-down data ingest download-from-s3 convert-to-safe partitioning stats export-to-s3

ENV_FILE ?= .env

s3-up:
	docker compose --env-file $(ENV_FILE) up -d --wait rustfs
	docker compose --env-file $(ENV_FILE) run --rm create-bucket

s3-down:
	docker compose down

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
