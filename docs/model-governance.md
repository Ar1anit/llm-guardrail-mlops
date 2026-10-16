# Model Governance

Every released model should include:

- A model card describing purpose, scope, limitations, and prohibited uses
- Versioned training and evaluation data lineage
- Reproducible configuration and experiment references
- Performance sliced by attack class, language, and relevant input characteristics
- Documented decision thresholds and regression budgets
- Security, privacy, fairness, and operational review approvals
- A rollback plan and a named model owner

Model promotion should progress through development, staging, and production gates. Production
artifacts must not be replaced in place.

## Training decisions awaiting approval

The training pipeline (`src/training/distilbert_pipeline.py`) makes these choices. Each affects the
false-positive/false-negative balance or what the model can see, so each needs reviewer sign-off
before a model is promoted:

- **Checkpoint selection** uses malicious-class F1 on the validation split. Recall alone would
  reward flagging everything; changing the metric changes which checkpoint is exported.
- **Decision rule** is `argmax` over the logits, equivalent to a 0.5 malicious-probability
  threshold. No threshold has been calibrated against documented false-positive and
  false-negative tolerances.
- **The loss is unweighted.** In the current data about 10% of training rows are malicious;
  class weighting or resampling would raise recall at the cost of precision.
- **Inputs are truncated** at `max_sequence_length` (512 tokens by default). Text beyond the limit
  is invisible to the classifier in training and in serving.
- **Release evidence** must come from the benchmark report written by
  `python -m src.evaluation.benchmark`. It loads the model only through
  `GoldenEvaluator.from_export`, which verifies the export against its manifest, and the report
  names the model, manifest, and golden-file SHA-256. The validation metrics stored in an
  export's `manifest.json` are for development only.
- **The golden set is still too small to calibrate thresholds on.** After the jailbreak source
  was added, `data/golden_eval.jsonl` has 200 rows: 153 benign, 33 jailbreak, 13 injection, and
  one hard negative. Stratifying on `attack_type` lowered the injection count from 20 to 13, so
  one missed injection moves its slice recall by almost eight points. Hard negatives cannot be
  measured. The golden set must be expanded before acceptance thresholds are calibrated.
- **Release gates are provisional.** `configs/evaluation/default.yaml` sets floors for recall,
  precision, the false-positive rate, the injection and jailbreak flag rates, and the minimum
  support for each slice. The benchmark exits non-zero when one fails or its metric is missing.
  The values catch a broken model; they are not calibrated against documented tolerances.
  Changing a gate needs reviewer approval. Export itself still refuses only a model that
  predicts one class for every validation row.
- **Base model pinning.** `base_model_revision` must be a commit SHA; a branch, tag, or missing
  revision is rejected because it can change between runs. The shipped `configs/model/classifier.yaml`
  sets `revision: null` and must be updated before the pipeline is wired to Hydra.
- **Validation coverage** is limited: the validation split has 432 rows with 29 injections,
  72 jailbreaks, and one hard negative, so false positives on benign security-education prompts
  are barely measured.
- **Jailbreak data** comes from one public source, `jackhhao/jailbreak-classification`, mostly
  role-play prompts from 2023. Recall on other jailbreak styles is unknown.

Each export is written once to a fresh directory and never overwritten; its `manifest.json`
records the dataset files, base-model revision, seed, settings, and library versions needed to
reproduce it.

