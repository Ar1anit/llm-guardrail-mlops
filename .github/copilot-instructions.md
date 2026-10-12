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

## Explaining results and code

Write explanations so that a human can review and verify the work. Use the lightest format that
makes the point: text first, then a diagram, then an HTML page, then a video.

### Writing

- Write plain, controlled English, about 80% of the way to ASD-STE100 (Simplified Technical
  English). Follow the full standard when the user asks for it.
- Keep sentences short: 20 words or fewer for instructions, 25 or fewer for descriptions.
- Use the active voice, simple words, and one word for one meaning.
- Write one instruction per sentence and one topic per paragraph.
- Simplify the sentence, not the facts. Keep exact terms, numbers, units, and caveats. When you
  report a metric, state the dataset split, the threshold, and the positive class.
- Separate what you ran or read from what you inferred. Name the files, configurations, and runs
  behind each claim.
- Apply these rules to your own explanations only. Do not rewrite existing documentation,
  docstrings, or comments to match them unless asked.

### Diagrams

- Add a diagram when flow, structure, or state is easier to see than to read.
- Write diagrams in Mermaid inside Markdown. Prefer standard notation: UML sequence, class, or
  state diagrams for behavior and structure; flowcharts for pipelines; ER diagrams for data.
- Keep one idea in each diagram. Label every node and edge.
- Show planned and placeholder components differently from implemented ones, for example with a
  "(planned)" label and `classDef planned stroke-dasharray:5 5`. A diagram must never make a
  placeholder look like a working security control.

### HTML pages and videos

- Create an explainer (an HTML page or a video) only when the user asks for one.
- Treat each explainer as discardable, not as product code. Write it to `tmp/explainers/` at the
  repository root, which Git ignores. Do not put it in `src/` or `tests/`. Do not commit it unless
  asked. It needs no tests. Tell the user the path.
- Make an HTML page self-contained. Do not load remote scripts, fonts, or trackers.
- Use synthetic examples. Do not embed real prompts, dataset or golden-set rows, model artifacts,
  or secrets. Render prompt and model text as text, never as HTML.
- Run extra tools in isolation with `uv run --no-project --with <package> ...`. Do not change
  `pyproject.toml` or `uv.lock` for an explainer.
- For narration, use a local text-to-speech tool. Use a hosted service only after explicit user
  approval. Read its API key from an environment variable. Never print or store the key.
