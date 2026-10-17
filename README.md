# LLM Security Guardrail Classifier

A production-oriented open-source MLOps scaffold for detecting prompt injection and jailbreak
attempts in real time.

> [!IMPORTANT]
> This repository is pre-alpha. It contains data ingestion, training, evaluation, model
> versioning, and a shadow-mode serving API. Release 0.1.0 only scores and logs prompts; it must
> not be treated as a security control.

## Goals

- Reproducible dataset and model lifecycle management
- Low-latency ONNX-based classification
- Golden-dataset security benchmarking
- Observable model quality and drift
- Transparent model governance and open-source maintenance

## Repository layout

```text
.
├── .github/              # Community health files and CI workflows
├── configs/              # Hydra configuration
├── data/                 # DVC-tracked raw and processed datasets
├── docs/                 # Architecture, threat model, and model governance
├── src/                  # Data, training, evaluation, serving, and monitoring packages
└── tests/                # Unit, integration, and contract test suites
```

## Development environment

Python 3.11 through 3.14 are supported; 3.14 is recommended for local work. Install
[uv](https://docs.astral.sh/uv/) and synchronize the complete environment:

```bash
uv sync --extra training --extra mlops --group dev
uv run pre-commit install
```

Run quality checks with:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

The training tests in `tests/unit/training` run offline against a tiny local model, and the
registry tests run against a local SQLite MLflow; both need the `training` and `mlops` extras,
which CI installs. The integration and contract suites are still placeholders.

## Data, training, and evaluation commands

```bash
uv run python -m src.data.ingestion --output-dir data   # Rebuild the DVC-tracked datasets
uv run python -m src.training.train training.batch_size=8 # Train and export (Hydra overrides)
uv run python -m src.evaluation.benchmark \
  --export-dir artifacts/training/<run>/export --report artifacts/evaluation/<run>.json
```

The benchmark exits non-zero when a release gate in `configs/evaluation/default.yaml` fails or
the export does not match its manifest. The gates are provisional; see
[model governance](docs/model-governance.md).

Benchmarked exports are released to the MLflow model registry with
`python -m src.registry.release` (see [architecture](docs/architecture.md#model-versioning)).
Pushing a tag `model-v<version>` then publishes a private GHCR image and a GitHub Release with
the model card and checksums (see
[publishing a release](docs/architecture.md#publishing-a-release-on-github)).
Release 0.1.0 is for [shadow mode only](docs/model-cards/guardrail-classifier-0.1.0.md).

## Serving (shadow mode)

```bash
uv run python -m src.registry.fetch --release <release> --output models/serving
docker build -t guardrail-classifier:0.1.0 .
docker run -p 127.0.0.1:8000:8000 -e GUARDRAIL_API_TOKEN=<random 32+ characters> \
  --read-only --tmpfs /tmp guardrail-classifier:0.1.0
```

`POST /v1/classify` with `Authorization: Bearer <token>` and `{"prompt": "..."}` returns a
verdict with `"action": "log_only"`. See [architecture](docs/architecture.md#serving-shadow-mode).

Each published release also deploys automatically to the shadow Container App on Azure; see
[deploying](docs/architecture.md#deploying-azure-container-apps).

## Data and model artifacts

Datasets under `data/raw/` and `data/processed/` are intended to be tracked with DVC rather than
Git. Model weights, ONNX exports, checkpoints, experiment runs, and local caches are excluded
from source control.

Do not add private, licensed, personal, or production data without documented approval and a
compatible data-use policy.

## Documentation

- [Architecture](docs/architecture.md)
- [Threat model](docs/threat-model.md)
- [Model governance](docs/model-governance.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Project governance](GOVERNANCE.md)

## License

Licensed under the [Apache License 2.0](LICENSE).

