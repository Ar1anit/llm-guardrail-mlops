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
- **The loss is unweighted.** In the current data about 48% of training rows are malicious, far
  above any realistic traffic prevalence; class weighting or resampling would move the
  precision-recall balance.
- **Inputs are truncated** at `max_sequence_length` (512 tokens by default). Text beyond the limit
  is invisible to the classifier in training and in serving.
- **Release evidence** must come from the benchmark report written by
  `python -m src.evaluation.benchmark`. It loads the model only through
  `GoldenEvaluator.from_export`, which verifies the export against its manifest, and the report
  names the model, manifest, and golden-file SHA-256. The validation metrics stored in an
  export's `manifest.json` are for development only.
- **The golden set has a fixed quota per attack type.** `data/golden_eval.jsonl` has 900 rows:
  300 benign, 200 hard negatives, 200 injections, and 200 jailbreaks, so one error moves a slice
  rate by half a point. Before this change it had 200 rows with 13 injections and one hard
  negative. The class mix is set by the quotas, not by traffic, so golden precision and the
  overall false-positive rate (which now includes the hard negatives) are not comparable with
  reports on the earlier golden set. Thresholds can now be calibrated, starting from a baseline
  run on this set; the per-slice flag rates are the numbers to calibrate against.
- **Near-duplicate leakage.** The first 900-row golden set was split row by row and had 291
  near-duplicate pairs across splits (similarity >= 0.5), 153 of them between golden and
  train, mostly jailbreak templates that appear in both jailbreak sources. Its jailbreak flag
  rate overstated generalisation. Ingestion now splits by near-duplicate cluster and refuses
  splits with any cross-split pair. Benchmark reports from before this change are not
  comparable with later ones.
- **Release gates are provisional.** `configs/evaluation/default.yaml` sets floors for recall,
  precision, the false-positive rate, the injection and jailbreak flag rates, and the minimum
  support for each slice. The benchmark exits non-zero when one fails or its metric is missing.
  The values catch a broken model; they are not calibrated against documented tolerances.
  Changing a gate needs reviewer approval. Export itself still refuses only a model that
  predicts one class for every validation row.
- **Base model pinning.** `base_model_revision` must be a commit SHA; a branch, tag, or missing
  revision is rejected because it can change between runs. `configs/model/classifier.yaml` pins
  `distilbert/distilbert-base-uncased` to commit `12040accade4e8a0f71eabdb258fecc2e7e948be`
  (2024-05-06). Changing the base model or its revision needs review.
- **Run provenance.** `src/training/train.py` records the git commit as `code_revision` and the
  verified DVC hashes as `dataset_revision`. It refuses a dirty working tree, or data that
  differs from its DVC pointers. `training.allow_dirty=true` overrides this for experiments and
  appends `-dirty` to the revision. A model with a `-dirty` revision must not be released.
- **Training device.** `use_cpu: false` lets the trainer use CUDA or Apple MPS. Results are
  reproducible only on the same device and library versions, which the manifest records.
- **Validation coverage:** the validation split has 998 rows: 316 benign, 207 hard negatives,
  305 injections, and 170 jailbreaks.
- **Jailbreak data** comes from `jackhhao/jailbreak-classification` and
  `TrustAIRLab/in-the-wild-jailbreak-prompts`, which overlap heavily and are mostly role-play
  prompts from 2023. Recall on newer or optimized jailbreak styles is unknown.
- **Injection data** is dominated by direct "ignore your instructions" and prompt-leaking
  prompts (Gandalf, deepset, and a multilingual set). Indirect injection through retrieved
  documents or tool output is not represented. The multilingual rows are scored by an
  English-only base model.
- **Hard negatives** come mostly from OR-Bench-Hard-1K, whose benign labels were assigned by an
  LLM-judge ensemble, and from XSTest's human-written safe prompts. Some OR-Bench prompts are
  borderline requests; they are benign here because they do not attack the model's
  instructions. A guardrail for harmful content would label them differently.

## Releases

- **Release record.** Every release has a model card in `docs/model-cards/`, a registry version
  with checksum and lineage tags, and the benchmark report in the same MLflow run. Release
  0.1.0 is [shadow mode only](model-cards/guardrail-classifier-0.1.0.md).
- **Deployments** are named pointers (`deployment.<name>` tags on the registered model). Moving a
  pointer is a promotion or rollback and needs the same review as the first release. Moving
  `shadow` does not change request outcomes; any enforcing deployment needs calibrated
  thresholds and a documented fail-open or fail-closed decision first.
- **Publication.** A `model-v<version>` tag publishes the release: a private GHCR image with a
  build-provenance attestation, and a public GitHub Release with the model card, benchmark
  report, `release.json`, `manifest.json`, and checksums. Weights are never public, because a
  public copy lets an attacker search offline for prompts the guardrail misses. The release
  workflow fails if the image package is not private.
- **Deployment.** Every published release deploys to the `production-shadow` Container App
  automatically, after its build provenance is verified. This is acceptable only because shadow
  mode cannot block a request. Any enforcing deployment must be a separate, manually approved
  step.
- **Immutability.** Registered versions are never edited. A second copy of each export also
  sits in the `models` Blob container under a 365-day retention policy.

Each export is written once to a fresh directory and never overwritten; its `manifest.json`
records the dataset files, base-model revision, seed, settings, and library versions needed to
reproduce it.

