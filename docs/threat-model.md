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

