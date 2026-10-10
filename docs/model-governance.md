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

