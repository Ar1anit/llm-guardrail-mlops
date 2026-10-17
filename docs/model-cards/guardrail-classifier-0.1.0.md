# Model card: guardrail-classifier 0.1.0

> [!IMPORTANT]
> **Shadow mode only.** Version 0.1.0 may score and log prompts. It must not block, rewrite, or
> route prompts, and it must not be the only control in front of a language model. The evidence
> below does not support enforcement.

## Purpose

The model classifies one prompt as `benign` or `malicious`. "Malicious" covers prompt injection
and jailbreak attempts. Version 0.1.0 is the first release. Its job is to collect real-traffic
evidence in shadow mode, so that the next version can be trained and calibrated on it.

## Release identity

| Field | Value |
| --- | --- |
| Registry model | `guardrail-classifier` (Azure ML workspace MLflow registry) |
| Release | `0.1.0` |
| Training run | `20261017T114940Z-cf1327c1a455` |
| `model.onnx` SHA-256 | `2ab5fbc0eceee5822d5a4aeea6fcbe1654b2dcace488a4410834e5f24ea5e6d2` |
| `manifest.json` SHA-256 | `465483a62a5140279e83b5eaef03e9d3466f98c037d5770fe1aba67d44210170` |
| Code revision | `cf1327c1a4551b0bc89365efb0d46a97191ffee4` |
| Dataset revision | `data/processed.dvc=2ed2a0756756a95df483fbd4ca4313e0.dir`, `data/golden_eval.jsonl.dvc=bdb4da960441462f1671a825d688ca1f` |
| Golden file SHA-256 | `d462748cddf7e0ca0e02d6b3e60848fac1556570981c30dcb71de5909c39ef29` |

The release command checks the export against its manifest and refuses a dirty lineage.

## Model

- **Base model:** `distilbert/distilbert-base-uncased`, Hub commit
  `12040accade4e8a0f71eabdb258fecc2e7e948be`, fine-tuned for two-class sequence classification.
- **Format:** ONNX (opset 17) with the exported tokenizer. Inputs are `input_ids` and
  `attention_mask`; the output is `logits`.
- **Input limit:** 512 tokens. Longer prompts are truncated, so text after the limit is not seen.
- **Decision rule:** `argmax`, the same as a malicious-probability threshold of 0.5. The threshold
  is not calibrated.

## Training

- **Run:** 3 epochs, batch size 32, learning rate 2e-5, seed 42, on one NVIDIA L4 (RunPod) with
  torch 2.14.1+cu126, transformers 5.19.0, and Python 3.14.8. The checkpoint with the best
  validation malicious-class F1 was kept.
- **Data:** 2,431 training rows: 1,868 benign, 8 hard negatives, 161 injections, and 394
  jailbreaks. The sources are pinned to commits:
  - `deepset/prompt-injections` (`4f61ecb0`): injections and benign prompts.
  - `databricks/databricks-dolly-15k` (`bdd27f4d`): general benign prompts.
  - `jackhhao/jailbreak-classification` (`2f2ceeb3`, `balanced/jailbreak_dataset_train_balanced.csv`):
    jailbreaks and benign prompts.
  - Ten curated hard negatives: benign questions about security topics.

## Evaluation

Release evidence comes from the golden set (200 rows, never used for training or selection),
scored on the exported ONNX model by `python -m src.evaluation.benchmark`.

| Measure | Golden set | Validation (development only) |
| --- | --- | --- |
| Attacks caught | 41 / 45, recall 0.911 | 92 / 98, recall 0.939 |
| False alarms | 0 / 155, false-positive rate 0.000 | 1 / 331, false-positive rate 0.003 |
| Precision / F1 | 1.000 / 0.954 | 0.989 / 0.963 |
| Jailbreaks caught | 31 / 32 (0.969) | 68 / 69 |
| Injections caught | 10 / 13 (0.769) | 24 / 29 |

All eight provisional gates in `configs/evaluation/default.yaml` passed. The gates are not
calibrated.

**Approximate 95% intervals (Wilson):** overall recall 0.79 to 0.96; injection recall 0.50 to
0.92; false-positive rate 0 to 0.024.

## Known limitations

- **Short, direct injections are the weak spot.** All four golden misses are short prompts (17 to
  48 tokens), and two of them score below 0.02. Long jailbreak templates score near 1.0. The
  model may partly rely on prompt length or role-play style.
- **False positives on security-related questions are not measured.** The golden set has one hard
  negative. Most benign rows are general questions from Dolly, which are easy to separate.
- **Small slices.** The golden set has 13 injections and 32 jailbreaks. One miss moves injection
  recall by almost eight points.
- **Languages.** Most data is English. Some injections are German; other languages are untested.
- **Truncation.** A payload placed after 512 tokens is not seen.

## Intended and prohibited use

- **Intended:** shadow scoring of inbound prompts, with logged verdicts and scores, to measure
  real-traffic behaviour.
- **Prohibited:** blocking or altering user prompts, use as the only guardrail, and any use that
  logs raw prompts without an approved redaction and retention policy.

## Governance

- **Owner:** repository maintainer.
- **Approvals:** security, privacy, and operations reviews are pending. Shadow mode does not
  change any request outcome.
- **Promotion to enforcement** needs a larger golden set (more injections and hard negatives),
  calibrated thresholds with documented false-positive and false-negative tolerances, a
  fail-open or fail-closed decision, and sign-off as in [model governance](../model-governance.md).
- **Rollback:** point the `shadow` deployment at the previous release with
  `python -m src.registry.release promote`, or stop the shadow scoring. Released versions are
  never changed in place.
