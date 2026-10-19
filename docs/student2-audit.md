# Student-2 Audit and Initial Security Layer

## Baseline

Audit target: origin/main at 511f3c3 on 9 October 2026. Student-2 work is on
student2/security-layer and is additive. Student 1's preprocessing, models,
experiments, EDA notebook, and error-analysis files are unmodified.

## What the existing work establishes

The Asteria.ipynb EDA records 175,341 UNSW-NB15 training rows and 45 columns.
It found no missing values or exact duplicates. The binary label is attack-heavy
(119,341 attacks; 56,000 normal records), and the data contains both numerical
features and the categorical proto, service, and state fields. The notebook
correctly treats id as an identifier and label and attack_cat as target
information.

Saved experiment output currently reports:

| Model | Precision | Recall | F1 | False-positive rate |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9144 | 0.9873 | 0.9495 | 0.1970 |
| Isolation Forest, contamination 0.30 | 0.5347 | 0.2383 | 0.3296 | 0.4419 |

The current Isolation Forest experiments are useful as anomaly-detection
research, but their saved results show that they are not a substitute for
validated attack-path evidence.

## Existing boundaries and observations

detection_engine.py emits supervised and anomaly outputs and creates
DetectionEvidence with evidence_builder.py. main.py is a separate
agent-telemetry heuristic flow using loader.py, features.py, detections.py,
and investigation.py.

The following observations are documented only; Student 1 files remain
unchanged:

- preprocessing.py declares a categorical pipeline but its current
  ColumnTransformer registers only numerical features.
- detection_engine.py builds evidence in a loop but prints only the final
  item after it completes.
- main.py expects a telemetry.csv input that is not tracked in this
  repository.
- The existing model scores and heuristic scores are detection signals, not
  capability reachability proofs or authorization decisions.

## Student-2 initial component

security_layer.py now provides a pure-Python, deterministic boundary for
future synthetic agent actions. DetectionEvidenceV1 accepts only the exact
plain built-in dictionary emitted by Student 1's current to_dict contract.
It validates required fields and scalar types, copies feature data into an
immutable normalized tuple, and emits its own explicit contract version.

The evidence adapter accepts only DetectionEvidenceV1. It does not accept
arbitrary mappings, provider objects, or custom to_dict implementations. A
positive supervised or anomaly model signal requests human investigation; it
does not assert that an attack path exists, does not validate an attack, and
does not authorize a response.

CapabilityPolicy accepts only a tuple of validated CapabilityRule values.
Identifiers and CapabilityAction values are checked strictly, rules are exact
matches, duplicate rules are rejected, and non-read actions require human
approval. An ungranted capability remains blocked by default.

The module is dry-run only. It does not call tools, access networks, change
permissions, or execute remediation.

## Next safe integration step

Define a synthetic agent telemetry fixture and an explicit policy file, then
pass only review-triggered records into an investigation workflow. Keep
machine-learning detection, policy evaluation, controlled validation, and
remediation execution as separate stages.

