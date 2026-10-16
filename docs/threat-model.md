# Threat Model

## Protected assets

- Downstream model instructions and system prompts
- User and tenant data
- Model artifacts, training data, and evaluation sets
- Classification telemetry and security decisions

## Initial threat categories

- Direct and indirect prompt injection
- Jailbreak and instruction-hierarchy manipulation
- Encoded, multilingual, or obfuscated adversarial input
- Data poisoning and golden-dataset contamination
- Model extraction and repeated probing
- Artifact substitution and dependency compromise
- Sensitive data exposure through logs or monitoring

## Trust boundaries

The client, classifier service, model registry, data store, monitoring system, and protected LLM
are separate trust boundaries. Authentication, authorization, input limits, integrity checks,
and audit requirements must be defined before production deployment.

## Security decision policy

The project does not yet define a default allow, block, or escalation policy. A production
release must document fail-open versus fail-closed behavior, uncertainty handling, threshold
ownership, and human-review procedures.

## Training pipeline controls and gaps

Implemented in `src/training/`:

- **Golden-dataset contamination and leakage:** only the `train` and `validation` splits are
  accepted. Every training and validation input is compared, after tokenization, with the
  golden file and with the other split, so renamed copies, case variants, and prompts that differ
  only beyond the truncation limit are rejected. Checkpoints are selected on validation data.
- **Artifact substitution and provenance:** exports are immutable, checked against PyTorch before
  publishing, and accompanied by a manifest with a SHA-256 for every file and for the dataset
  files, taken before parsing and verified again at export. The base model must be pinned to a
  commit SHA, and manifest metrics are attached only to the exact weights they were computed on.
- **Silent unsafe decisions:** non-finite logits, a non-finite training loss, metrics over a
  missing class, and a model that predicts one class for every validation row raise instead of
  producing predictions, numbers, or an artifact.
- **Sensitive data exposure:** validation errors never include prompt text, and no experiment
  tracker is enabled.

Implemented in `src/data/` and `src/evaluation/`:

- **Silent data loss and mislabelling:** ingestion counts every rejected row, raises when a label
  is missing or unknown, when a source rejects more than 5% of its rows, and when one text has
  both labels. Every source is pinned to a commit SHA.
- **Unmeasured attack classes:** ingestion refuses data without every attack type and stratifies
  the splits on `attack_type`, so jailbreak and injection recall are separate golden slices.
- **Evaluating the wrong artifact:** `GoldenEvaluator.from_export` refuses an export whose files
  differ from its manifest or whose inference contract it cannot honour. The benchmark report
  binds every metric to the model, manifest, and golden-file SHA-256 and exits non-zero when a
  gate fails.

Known gaps:

- The manifest checksums detect accidental change but are not signatures: anyone who can write the
  export directory can rewrite a file and its checksum. Signing and a model registry are needed
  before production.
- Exports are immutable only against this pipeline. Concurrent writers to one output directory,
  or someone who can replace the directory between the final check and the rename, are not
  handled.
- `dataset_revision` is recorded but not verified against the DVC-tracked files; the manifest
  records the SHA-256 of the files actually used.
- Prompts longer than `max_sequence_length` are truncated, so a payload placed after the limit is
  not seen.
- Duplicate detection is exact on tokenized input. Paraphrases and other near-duplicates are not
  detected. Ingestion requires every attack type to be present but sets no minimum count or
  balance.
- The export check rejects a model that predicts one class, but there is no quality threshold.
  The benchmark gates are provisional and are not yet a required CI step (see model governance).
- The benchmark verifies tokenizer files and then loads them, so someone who can write the export
  directory between those two steps can swap them. The model file has no such gap.
- The ONNX export uses PyTorch's legacy exporter, which PyTorch has announced it will remove.

