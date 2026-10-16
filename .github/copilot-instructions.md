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

- Support Python 3.11 through 3.14.
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

Optimize explanations for human understanding and oversight. Lead with the result, then explain
what changed, why, validation evidence, limitations, and decisions that need review. Keep simple
answers short.

Choose the simplest format that makes the subject clear:

- Text for concise facts, decisions, and short explanations.
- Diagrams or charts for flows, relationships, state, and quantitative comparisons.
- Self-contained HTML for interactive exploration, controls, or side-by-side comparisons.
- Video for ideas that benefit from motion, a step-by-step visual story, or narration.

Honor the user's requested format. These are options, not a mandatory sequence.
For a complex flow or dense result, offer an HTML page or video once per topic and explain its
benefit. Build one only after the user asks or accepts. Do not produce every format by default.

In every format, show which parts are implemented and which are planned. Never make a placeholder
look like a working security control.

### Writing

- Write plain, controlled English, about 80% of the way to ASD-STE100 (Simplified Technical
  English). Treat this as a readability guide, not a formal compliance claim. Use stricter rules
  when asked.
- Aim for short sentences: 20 words or fewer for instructions, 25 or fewer for descriptions.
  Preserve clarity and technical accuracy instead of forcing a word limit.
- Use the active voice, simple words, and one word for one meaning.
- Write one instruction per sentence and one topic per paragraph. Keep a paragraph to six
  sentences or fewer.
- Keep articles and verbs. Do not write noun strings longer than three words.
- Define unfamiliar terms on first use. Use a small, concrete example to explain a difficult
  abstraction.
- Simplify the sentence, not the facts. Keep exact terms, numbers, units, and caveats. When you
  report a classifier metric, state the dataset split, the threshold, and the positive class.
- Separate what you ran or read from inferences, assumptions, and unknowns. Link to the files,
  configurations, runs, or sources behind your conclusions. Never invent execution or evaluation
  results.
- Apply these rules to your own explanations only. Do not rewrite existing documentation,
  docstrings, or comments to match them unless asked.

### Diagrams and charts

- Add a diagram when flow, structure, or state is easier to see than to read.
- Write diagrams in Mermaid inside Markdown. Prefer standard notation: UML sequence, class, or
  state diagrams for behavior and structure; flowcharts for pipelines; ER diagrams for data.
- Show metrics as a chart when a table is hard to scan. Examples: a confusion matrix, a
  precision-recall curve, or bars per `attack_type`.
- Keep one idea in each diagram or chart. Label every node, edge, and axis.
- Include a short text takeaway so the explanation is useful without rendering the diagram or
  chart.
- Mark planned parts with a "(planned)" label. Use dashed styling where the diagram type supports
  it.

### HTML pages and videos

- Treat each explainer as discardable, not as product code. Write it to `tmp/explainers/` at the
  repository root, which Git ignores, and never to `src/` or `tests/`. Do not commit it unless
  asked. Tell the user the path.
- Make an HTML page self-contained: no remote scripts, fonts, or trackers. Make it interactive
  where that helps, for example a threshold slider that updates a confusion matrix.
- Make videos visual and step-by-step, using original diagrams and animations, for example with
  Manim. Save the narration script next to the video as a transcript that a reviewer can check.
- Use synthetic examples. Label synthetic metrics as illustrative, not measured performance.
  Do not embed real prompts, dataset or golden-set rows, model artifacts, or secrets. Render
  prompt and model text as text, never as HTML.
- Run Python tools in isolation with `uv run --no-project --with <package> ...`. Do not change
  `pyproject.toml` or `uv.lock` for an explainer. Ask before you install system packages (for
  example Cairo, FFmpeg, or LaTeX) or download large files.
- For narration, use a local text-to-speech tool, for example `say` on macOS. Use a hosted service
  only after explicit user approval. A hosted service receives the script, so the script must hold
  nothing sensitive. Read its API key from an environment variable. Never print or store the key.
- Check the result before you report it. For HTML, open the page, exercise its main controls, and
  check the browser console for errors. For video, check the rendered file, timing, and any audio
  against the script. Keep claims and numbers consistent across formats. State any checks you
  could not run.
