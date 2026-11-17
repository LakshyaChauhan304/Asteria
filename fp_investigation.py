
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.stats import ks_2samp


# --------------------------------------------------
# 1. LOAD MODEL AND OFFICIAL TEST DATA
# --------------------------------------------------

MODEL_PATH = Path("models/xgboost_asteria_pipeline.pkl")
TEST_PATH = Path("data/UNSW_NB15_testing-set.csv")

OUTPUT_DIR = Path("analysis/false_positive_investigation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("ASTERIA - FALSE POSITIVE INVESTIGATION")
print("=" * 65)

pipeline = joblib.load(MODEL_PATH)
test_df = pd.read_csv(TEST_PATH)

excluded_columns = ["id", "attack_cat", "label"]

X_test = test_df.drop(
    columns=excluded_columns,
    errors="ignore"
)

y_test = test_df["label"]

# Use the same provisional threshold as the saved evaluation.
threshold = 0.50

probabilities = pipeline.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

results = test_df.copy()

results["attack_probability"] = probabilities
results["predicted_label"] = predictions

# False positives: actual normal, predicted attack.
false_positives = results[
    (results["label"] == 0) &
    (results["predicted_label"] == 1)
].copy()

# True negatives: actual normal, predicted normal.
true_negatives = results[
    (results["label"] == 0) &
    (results["predicted_label"] == 0)
].copy()

print("\nTOTAL NORMAL RECORDS:", len(false_positives) + len(true_negatives))
print("FALSE POSITIVES:", len(false_positives))
print("TRUE NEGATIVES:", len(true_negatives))
print(
    "FALSE POSITIVE RATE:",
    round(
        len(false_positives) /
        (len(false_positives) + len(true_negatives)) * 100,
        2
    ),
    "%"
)


# --------------------------------------------------
# 2. SAVE FALSE POSITIVE RECORDS
# --------------------------------------------------

false_positives.to_csv(
    OUTPUT_DIR / "false_positive_records.csv",
    index=False
)

print("\nSaved false_positive_records.csv")


# --------------------------------------------------
# 3. ANALYSE PROTOCOL, STATE AND SERVICE
# --------------------------------------------------

def analyse_category(column):

    if column not in results.columns:
        return

    grouped = results[results["label"] == 0].copy()

    grouped[column] = grouped[column].fillna(
        "<MISSING>"
    ).astype(str)

    summary = grouped.groupby(column).agg(
        total_normal_records=("label", "size"),
        false_positives=("predicted_label", "sum"),
        average_attack_probability=("attack_probability", "mean")
    ).reset_index()

    summary["true_negatives"] = (
        summary["total_normal_records"] -
        summary["false_positives"]
    )

    summary["false_positive_rate_pct"] = (
        summary["false_positives"] /
        summary["total_normal_records"] * 100
    )

    summary = summary.sort_values(
        "false_positives",
        ascending=False
    )

    summary.to_csv(
        OUTPUT_DIR / f"false_positives_by_{column}.csv",
        index=False
    )

    print(f"\nFALSE POSITIVES BY {column.upper()}")
    print(summary.to_string(index=False))


for column in ["proto", "state", "service"]:
    analyse_category(column)


# --------------------------------------------------
# 4. COMPARE FALSE POSITIVES AGAINST TRUE NEGATIVES
# --------------------------------------------------

excluded = [
    "id", "attack_cat", "label",
    "attack_probability", "predicted_label"
]

numerical_features = [
    col for col in X_test.columns
    if col not in ["proto", "service", "state"]
]

feature_results = []

for col in numerical_features:

    fp_values = pd.to_numeric(
        false_positives[col], errors="coerce"
    ).dropna()

    tn_values = pd.to_numeric(
        true_negatives[col], errors="coerce"
    ).dropna()

    if fp_values.empty or tn_values.empty:
        continue

    ks_stat, ks_pvalue = ks_2samp(
        fp_values,
        tn_values
    )

    feature_results.append({
        "feature": col,
        "fp_mean": fp_values.mean(),
        "tn_mean": tn_values.mean(),
        "fp_median": fp_values.median(),
        "tn_median": tn_values.median(),
        "fp_std": fp_values.std(),
        "tn_std": tn_values.std(),
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue
    })

feature_report = pd.DataFrame(feature_results)

feature_report = feature_report.sort_values(
    "ks_statistic",
    ascending=False
)

feature_report.to_csv(
    OUTPUT_DIR / "false_positive_feature_comparison.csv",
    index=False
)

print("\nTOP 15 FEATURES DIFFERENTIATING FP FROM TN")

print(
    feature_report[
        [
            "feature",
            "fp_median",
            "tn_median",
            "ks_statistic",
            "ks_pvalue"
        ]
    ].head(15).to_string(index=False)
)


# --------------------------------------------------
# 5. FALSE POSITIVE PROBABILITY DISTRIBUTION
# --------------------------------------------------

print("\nFALSE POSITIVE ATTACK PROBABILITIES")

print(
    false_positives["attack_probability"].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

print("\nTRUE NEGATIVE ATTACK PROBABILITIES")

print(
    true_negatives["attack_probability"].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)


# --------------------------------------------------
# 6. SAVE PROBABILITY SUMMARY
# --------------------------------------------------

probability_summary = pd.DataFrame({
    "group": ["False Positives", "True Negatives"],
    "count": [
        len(false_positives),
        len(true_negatives)
    ],
    "mean_probability": [
        false_positives["attack_probability"].mean(),
        true_negatives["attack_probability"].mean()
    ],
    "median_probability": [
        false_positives["attack_probability"].median(),
        true_negatives["attack_probability"].median()
    ],
    "p90_probability": [
        false_positives["attack_probability"].quantile(0.90),
        true_negatives["attack_probability"].quantile(0.90)
    ]
})

probability_summary.to_csv(
    OUTPUT_DIR / "probability_comparison.csv",
    index=False
)


# --------------------------------------------------
# 7. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 65)
print("FALSE POSITIVE INVESTIGATION COMPLETED")
print("=" * 65)

print("\nReports saved in:", OUTPUT_DIR)
