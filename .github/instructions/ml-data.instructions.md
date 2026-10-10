---
name: ML and Data Changes
description: "Use when changing training, evaluation, data ingestion, DVC files, Hydra configs, model exports, thresholds, or monitoring. Covers reproducibility, lineage, and security review."
applyTo: ["src/training/**", "src/evaluation/**", "src/data/**", "src/monitoring/**", "configs/**", "data/**/*.dvc"]
---

# ML and data change rules

- Keep raw and processed dataset contents out of Git; commit only intended DVC metadata and
  configuration.
- Make data transformations deterministic and record the input dataset revision, configuration,
  code revision, and random seeds needed to reproduce results.
- Validate schemas, labels, class balance assumptions, and input limits at boundaries.
- Treat evaluation data and golden security examples as protected assets. Do not weaken or remove
  regression cases to make a metric pass.
- Report metrics by relevant attack class and input slice where the change affects model behavior.
  Include threshold changes, false-positive/false-negative implications, and acceptance criteria.
- Keep exported artifacts immutable, checksummed, and traceable to their source revisions.
- Never log raw prompts, sensitive dataset rows, credentials, or model artifacts unless an explicit
  approved redaction policy requires it.
- For behavior that affects a security decision, update tests and the threat-model or governance
  documentation when the documented policy changes.
