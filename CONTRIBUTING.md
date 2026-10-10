# Contributing

Thank you for helping build the LLM Security Guardrail Classifier.

## Before contributing

- Search existing issues and discussions.
- Open an issue for significant behavior, architecture, data, or dependency changes.
- Never submit secrets, personal data, proprietary datasets, or unlicensed model artifacts.
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md) and [Security Policy](SECURITY.md).

## Development workflow

1. Fork the repository and create a focused branch.
2. Install dependencies with `uv sync --extra training --extra mlops --group dev`.
3. Add tests and documentation with implementation changes.
4. Run Ruff, mypy, and the relevant pytest suites.
5. Open a pull request using the repository template.

Commits should be focused and use an imperative summary. Pull requests should explain user
impact, security implications, data or model changes, and validation evidence.

## Model and data changes

Changes to datasets, labeling policy, features, thresholds, training, or model promotion require
documented lineage and evaluation results. DVC metadata belongs in Git; dataset contents do not.

## Developer Certificate of Origin

By contributing, you certify that you have the right to submit the work under the project's
license. Maintainers may require commits to include a `Signed-off-by` trailer.

