# Shadow-mode serving image: one verified model version per image.
# Build context must contain models/serving, written by:
#   uv run python -m src.registry.fetch --release 0.1.0 --output models/serving
ARG PYTHON_IMAGE=python:3.14-slim-bookworm

FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra serving --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --extra serving --no-editable

FROM ${PYTHON_IMAGE}
RUN useradd --system --uid 10001 --no-create-home guardrail
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY configs ./configs
COPY models/serving ./models/serving
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["python", "-m", "src.serving.api"]
