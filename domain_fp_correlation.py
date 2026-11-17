
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr


# --------------------------------------------------
# 1. LOAD DATA AND SAVED MODELS
# --------------------------------------------------

train_df = pd.read_csv(
    "data/UNSW_NB15_training-set.csv"
)

test_df = pd.read_csv(
    "data/UNSW_NB15_testing-set.csv"
)

domain_model = joblib.load(
    "analysis/domain_classifier/domain_classifier.pkl"
)

xgb_model = joblib.load(
    "models/xgboost_asteria_pipeline.pkl"
)

OUTPUT_DIR = Path("analysis/domain_fp_correlation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.50

print("=" * 65)
print("ASTERIA - DOMAIN SHIFT / FALSE POSITIVE CORRELATION")
print("=" * 65)


# --------------------------------------------------
# 2. RECREATE THE DOMAIN CLASSIFIER DATASET
# --------------------------------------------------

excluded = ["id", "attack_cat", "label"]

train_normal = train_df[
    train_df["label"] == 0
].copy()

test_normal = test_df[
    test_df["label"] == 0
].copy()

X_train_domain = train_normal.drop(
    columns=excluded
)

X_test_domain = test_normal.drop(
    columns=excluded
)

X_domain = pd.concat(
    [X_train_domain, X_test_domain],
    ignore_index=True
)

y_domain = np.concatenate([
    np.zeros(len(X_train_domain), dtype=int),
    np.ones(len(X_test_domain), dtype=int)
])


# --------------------------------------------------
# 3. RECREATE THE EXACT VALIDATION SPLIT
# --------------------------------------------------

all_indices = np.arange(len(y_domain))

_, validation_indices = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=42,
    stratify=y_domain
)

# Keep only held-out testing-domain records.
test_domain_start = len(train_normal)

heldout_test_positions = validation_indices[
    validation_indices >= test_domain_start
]

# Convert positions into original official-test row indices.
test_normal_positions = (
    heldout_test_positions - test_domain_start
)

original_test_indices = test_normal.index[
    test_normal_positions
]

X_heldout = X_domain.iloc[
    heldout_test_positions
]

print("\nHeld-out testing-domain normal records:",
      len(X_heldout))


# --------------------------------------------------
# 4. GET DOMAIN CLASSIFIER PROBABILITIES
# --------------------------------------------------

domain_probabilities = domain_model.predict_proba(
    X_heldout
)[:, 1]


# --------------------------------------------------
# 5. GET XGBOOST PREDICTIONS ON SAME RECORDS
# --------------------------------------------------

XGB_FEATURES = [
    col for col in train_df.columns
    if col not in excluded
]

X_exact = test_df.loc[
    original_test_indices,
    XGB_FEATURES
]

xgb_probabilities = xgb_model.predict_proba(
    X_exact
)[:, 1]

xgb_predictions = (
    xgb_probabilities >= THRESHOLD
).astype(int)


# --------------------------------------------------
# 6. BUILD MATCHED ANALYSIS DATASET
# --------------------------------------------------

matched = test_df.loc[
    original_test_indices,
    ["proto", "state", "service", "label"]
].copy()

matched["domain_probability"] = domain_probabilities
matched["attack_probability"] = xgb_probabilities
matched["xgb_prediction"] = xgb_predictions

matched["result"] = np.where(
    matched["xgb_prediction"] == 1,
    "FALSE_POSITIVE",
    "TRUE_NEGATIVE"
)

matched.to_csv(
    OUTPUT_DIR / "matched_domain_xgb_predictions.csv",
    index=False
)


# --------------------------------------------------
# 7. CORRELATION ANALYSIS
# --------------------------------------------------

correlation, p_value = spearmanr(
    matched["domain_probability"],
    matched["attack_probability"]
)

print("\nCORRELATION RESULTS")

print(
    "Spearman correlation:",
    round(correlation, 4)
)

print(
    "P-value:",
    p_value
)


# --------------------------------------------------
# 8. DOMAIN PROBABILITY BANDS
# --------------------------------------------------

matched["domain_probability_band"] = pd.qcut(
    matched["domain_probability"],
    q=5,
    labels=[
        "Q1 - Lowest",
        "Q2",
        "Q3",
        "Q4",
        "Q5 - Highest"
    ],
    duplicates="drop"
)

band_summary = (
    matched.groupby(
        "domain_probability_band",
        observed=False
    )
    .agg(
        records=("label", "size"),
        mean_domain_probability=("domain_probability", "mean"),
        mean_attack_probability=("attack_probability", "mean"),
        false_positives=("xgb_prediction", "sum")
    )
    .reset_index()
)

band_summary["false_positive_rate_pct"] = (
    band_summary["false_positives"]
    / band_summary["records"] * 100
)

band_summary.to_csv(
    OUTPUT_DIR / "domain_probability_band_analysis.csv",
    index=False
)

print("\nFALSE POSITIVE RATE BY DOMAIN PROBABILITY")

print(band_summary.to_string(index=False))


# --------------------------------------------------
# 9. ANALYSE THE FOUR PROBLEMATIC COMBINATIONS
# --------------------------------------------------

PATTERNS = [
    ("tcp", "FIN", "-"),
    ("udp", "INT", "-"),
    ("tcp", "FIN", "http"),
    ("tcp", "FIN", "ftp")
]

pattern_results = []

for proto, state, service in PATTERNS:

    subset = matched[
        (matched["proto"].astype(str) == proto) &
        (matched["state"].astype(str) == state) &
        (matched["service"].astype(str) == service)
    ]

    fp = int(subset["xgb_prediction"].sum())
    total = len(subset)

    pattern_results.append({
        "proto": proto,
        "state": state,
        "service": service,
        "records": total,
        "mean_domain_probability": (
            subset["domain_probability"].mean()
        ),
        "mean_attack_probability": (
            subset["attack_probability"].mean()
        ),
        "false_positives": fp,
        "false_positive_rate_pct": (
            fp / total * 100 if total else np.nan
        )
    })

pattern_report = pd.DataFrame(pattern_results)

pattern_report.to_csv(
    OUTPUT_DIR / "problematic_pattern_domain_analysis.csv",
    index=False
)

print("\nPROBLEMATIC PATTERN ANALYSIS")

print(pattern_report.to_string(index=False))


# --------------------------------------------------
# 10. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 65)
print("DOMAIN / FALSE POSITIVE ANALYSIS COMPLETED")
print("=" * 65)

print("\nReports saved inside:", OUTPUT_DIR)
