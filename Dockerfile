FROM ghcr.io/astral-sh/uv:0.11.32 AS uv
FROM python:3.12.13-slim-bookworm
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev
ARG GIT_SHA=unknown
ENV MLPM_GIT_SHA=${GIT_SHA} PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["mlpm-train"]
