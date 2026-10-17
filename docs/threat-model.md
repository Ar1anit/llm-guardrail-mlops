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

The only serving mode today is **shadow**: every verdict carries `"action": "log_only"`, so no
request outcome depends on the model, and an outage or wrong verdict cannot block a user. This
is fail-open by construction and is acceptable only because nothing is enforced.

## Serving boundary controls and gaps

Implemented in `src/serving/`:

- **Model substitution:** the service starts only with an export that verifies against its
  manifest and matches the `model_sha256` in `release.json`. The image holds one model version.
- **Unauthorized use:** classification and model routes need a bearer token of at least 32
  characters, compared in constant time. Disabling authentication is an explicit setting.
- **Prompt leakage:** logs and error responses never contain prompt text; validation errors do
  not echo the request body; API documentation routes are disabled.
- **Resource abuse:** prompts are limited to 10,000 characters.
- **Telemetry:** ONNX Runtime telemetry events are disabled before a model is loaded.

Implemented in `infra/azure/serving.bicep` and `.github/workflows/deploy.yml`:

- **Image substitution in deployment:** the deploy workflow refuses an image unless its build
  provenance shows this repository's `release.yml` built it, deploys only by digest, and checks
  that the live service reports the release's model checksum.
- **Credential exposure:** GitHub Actions signs in to Azure through OIDC; the release and deploy
  identities are separate, each trusted by one GitHub environment, with access to one resource
  group or workspace. The app pulls with a managed identity that has only `AcrPull`. The API
  token lives only in the Container App's secret store.

Known gaps:

- The endpoint is reachable from the internet; only the bearer token protects it, unless the
  source-IP allowlist is set.
- One shared static token; no per-caller identity, rotation, or rate limiting yet.
- The deploy identity has Contributor on the serving resource group, which is broader than
  rolling out an image needs. A custom role would narrow it.
- Every successful release deploys to shadow automatically. That is acceptable only because
  shadow mode cannot change a request outcome.
- ONNX Runtime initializes its telemetry provider when it is imported, before it can be
  disabled. Whether that step makes a network call has not been verified; restrict outbound
  traffic from the serving environment.
- Shadow verdicts are only logged as metadata. Collecting prompts for evaluation needs an
  approved redaction and retention policy first.

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
- **Template leakage across splits:** ingestion deduplicates on the tokenized, truncated model
  input, so variants of one long jailbreak template that differ only after the truncation point
  cannot land in different splits.
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
- `dataset_revision` is verified against the DVC pointers only when training runs through
  `src/training/train.py`. A caller that builds `DistilBertPipeline` directly can pass any
  string. The manifest always records the SHA-256 of the files actually used.
- Prompts longer than `max_sequence_length` are truncated, so a payload placed after the limit is
  not seen.
- Duplicate detection is exact on the tokenized, truncated input. Paraphrases and templates that
  differ within the first `max_sequence_length` tokens are not detected, so near-duplicate
  jailbreak variants can still appear in more than one split. Ingestion requires every attack type to be present but sets no minimum count or
  balance.
- The export check rejects a model that predicts one class, but there is no quality threshold.
  The benchmark gates are provisional and are not yet a required CI step (see model governance).
- The benchmark verifies tokenizer files and then loads them, so someone who can write the export
  directory between those two steps can swap them. The model file has no such gap.
- The ONNX export uses PyTorch's legacy exporter, which PyTorch has announced it will remove.

