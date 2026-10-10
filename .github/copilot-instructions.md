# LLM Security Guardrail Classifier Guidelines

## Project status

This repository is a pre-alpha scaffold. Do not describe placeholder modules, empty pipelines, or
unimplemented serving paths as production-ready security controls. When implementing a planned
component, update the relevant architecture, threat-model, governance, and README documentation.

## Repository structure

- `src/data/` owns dataset ingestion and validation.
- `src/training/` owns reproducible training and export workflows.
- `src/evaluation/` owns golden-dataset benchmarks and release metrics.
- `src/serving/` owns the online inference boundary.
- `src/monitoring/` owns telemetry and drift checks.
- `configs/` contains Hydra configuration.
- `data/` contains DVC-managed datasets; dataset contents must not be committed to Git.
- `tests/` is organized into unit, integration, and contract suites.

Keep boundaries explicit and avoid putting training, serving, or monitoring behavior in unrelated
packages. Follow the existing `src` package layout and use absolute imports from `src`.

## Python and quality

- Support Python 3.11 and 3.12.
- Use `uv` for environment and command execution; do not add an alternate dependency manager.
- Format and lint with Ruff using the settings in `pyproject.toml`.
- Keep mypy strict-compatible and preserve useful type information at public boundaries.
- Add or update focused tests for behavior changes. Prefer deterministic tests and small fixtures.
- Before proposing changes, run the smallest relevant checks; for broad changes use:
  `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, and `uv run pytest`.

## Security, data, and model changes

- Never add secrets, personal data, private prompts, licensed datasets, model weights, checkpoints,
  experiment runs, or local caches to source control.
- Treat prompts, datasets, model outputs, logs, and telemetry as potentially sensitive.
- Do not silently swallow failures or create success-shaped fallbacks in a security decision path.
- Document data lineage, model/configuration revisions, threshold changes, evaluation results, and
  rollback implications for training or model changes.
- Keep online and offline preprocessing behavior aligned and covered by tests.
- Do not change fail-open/fail-closed behavior, thresholds, or API contracts without documenting
  the decision and its security and operational impact.

See [CONTRIBUTING.md](../CONTRIBUTING.md), [docs/threat-model.md](../docs/threat-model.md), and
[docs/model-governance.md](../docs/model-governance.md) for the detailed project policies.
