# LLM Security Guardrail Classifier

A production-oriented open-source MLOps scaffold for detecting prompt injection and jailbreak
attempts in real time.

> [!IMPORTANT]
> This repository is currently a scaffold. It contains data-ingestion and model-training code but
> no released classifier, trained model, API handler, or serving implementation, and must not be
> treated as a security control.

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

Python 3.11 and 3.12 are supported. Install
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

The training tests in `tests/unit/training` run offline against a tiny local model and need the
`training` extra. The integration and contract suites are still placeholders. CI currently
installs only the `dev` group, so it cannot import the training dependencies.

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

