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
- **Release evidence** must come from `GoldenEvaluator.evaluate_file` in `src/evaluation`, run on
  the exported ONNX model. The validation metrics stored in an export's `manifest.json` are for
  development only.
- **The golden set is too small to set thresholds on.** `data/golden_eval.jsonl` has 200 rows, of
  which only 20 are injections and one is a hard negative, and it has no jailbreaks. One missed
  injection moves recall by five points. It must be expanded before acceptance thresholds are
  set; until then, golden results are provisional.
- **No quality gate yet.** Export refuses only a model that predicts a single class on every
  validation row. It applies no recall, precision, or false-positive threshold, so a weak model
  still exports, and its weakness is visible in `manifest.json`. Regression budgets must be set
  and enforced per release before promotion.
- **Base model pinning.** `base_model_revision` must be a commit SHA; a branch, tag, or missing
  revision is rejected because it can change between runs. The shipped `configs/model/classifier.yaml`
  sets `revision: null` and must be updated before the pipeline is wired to Hydra.
- **Validation coverage** is limited: the current validation split has few malicious rows and no
  `hard_negative` rows, so false positives on benign security-education prompts are not measured.

Each export is written once to a fresh directory and never overwritten; its `manifest.json`
records the dataset files, base-model revision, seed, settings, and library versions needed to
reproduce it.

