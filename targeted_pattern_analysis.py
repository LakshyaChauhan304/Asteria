
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.stats import ks_2samp


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

train_df = pd.read_csv(
    "data/UNSW_NB15_training-set.csv"
)

test_df = pd.read_csv(
    "data/UNSW_NB15_testing-set.csv"
)

pipeline = joblib.load(
    "models/xgboost_asteria_pipeline.pkl"
)

OUTPUT_DIR = Path("analysis/targeted_patterns")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED = ["id", "attack_cat", "label"]

FEATURES = [
    col for col in train_df.columns
    if col not in EXCLUDED
]

# Keep the saved model's provisional threshold.
THRESHOLD = 0.50

# Obtain predictions only for the official test data.
X_test = test_df[FEATURES].copy()

test_probabilities = pipeline.predict_proba(X_test)[:, 1]

test_df = test_df.copy()

test_df["attack_probability"] = test_probabilities
test_df["prediction"] = (
    test_probabilities >= THRESHOLD
).astype(int)


# --------------------------------------------------
# 2. DEFINE THE FOUR PROBLEMATIC COMBINATIONS
# --------------------------------------------------

PATTERNS = [
    ("tcp", "FIN", "-"),
    ("udp", "INT", "-"),
    ("tcp", "FIN", "http"),
    ("tcp", "FIN", "ftp")
]

print("=" * 65)
print("ASTERIA - TARGETED PATTERN INVESTIGATION")
print("=" * 65)


# --------------------------------------------------
# 3. COMPARE TRAINING AND TESTING COUNTS
# --------------------------------------------------

count_results = []

for proto, state, service in PATTERNS:

    for dataset_name, df in [
        ("TRAIN", train_df),
        ("TEST", test_df)
    ]:

        subset = df[
            (df["proto"].astype(str) == proto) &
            (df["state"].astype(str) == state) &
            (df["service"].astype(str) == service)
        ]

        normal_count = int((subset["label"] == 0).sum())
        attack_count = int((subset["label"] == 1).sum())

        row = {
            "dataset": dataset_name,
            "proto": proto,
            "state": state,
            "service": service,
            "total_records": len(subset),
            "normal_records": normal_count,
            "attack_records": attack_count
        }

        if dataset_name == "TEST":

            fp = int(
                (
                    (subset["label"] == 0) &
                    (subset["prediction"] == 1)
                ).sum()
            )

            tn = int(
                (
                    (subset["label"] == 0) &
                    (subset["prediction"] == 0)
                ).sum()
            )

            tp = int(
                (
                    (subset["label"] == 1) &
                    (subset["prediction"] == 1)
                ).sum()
            )

            fn = int(
                (
                    (subset["label"] == 1) &
                    (subset["prediction"] == 0)
                ).sum()
            )

            row.update({
                "false_positives": fp,
                "true_negatives": tn,
                "true_positives": tp,
                "false_negatives": fn,
                "normal_fpr_pct": (
                    fp / normal_count * 100
                    if normal_count else np.nan
                ),
                "attack_recall_pct": (
                    tp / attack_count * 100
                    if attack_count else np.nan
                )
            })

        count_results.append(row)

count_report = pd.DataFrame(count_results)

count_report.to_csv(
    OUTPUT_DIR / "targeted_pattern_counts.csv",
    index=False
)

print("\nTRAINING VS TESTING COUNTS")
print(count_report.to_string(index=False))


# --------------------------------------------------
# 4. COMPARE NORMAL FEATURE DISTRIBUTIONS
# --------------------------------------------------

categorical_features = [
    "proto", "service", "state"
]

numerical_features = [
    col for col in FEATURES
    if col not in categorical_features
]

feature_results = []

for proto, state, service in PATTERNS:

    train_subset = train_df[
        (train_df["proto"].astype(str) == proto) &
        (train_df["state"].astype(str) == state) &
        (train_df["service"].astype(str) == service) &
        (train_df["label"] == 0)
    ]

    test_subset = test_df[
        (test_df["proto"].astype(str) == proto) &
        (test_df["state"].astype(str) == state) &
        (test_df["service"].astype(str) == service) &
        (test_df["label"] == 0)
    ]

    print(
        f"\nAnalysing {proto} | {state} | {service}"
    )

    print("Training normal records:", len(train_subset))
    print("Testing normal records:", len(test_subset))

    if len(train_subset) < 2 or len(test_subset) < 2:
        print("Insufficient records for distribution comparison.")
        continue

    for col in numerical_features:

        train_values = pd.to_numeric(
            train_subset[col], errors="coerce"
        ).dropna()

        test_values = pd.to_numeric(
            test_subset[col], errors="coerce"
        ).dropna()

        if train_values.empty or test_values.empty:
            continue

        ks_stat, ks_pvalue = ks_2samp(
            train_values,
            test_values
        )

        feature_results.append({
            "proto": proto,
            "state": state,
            "service": service,
            "feature": col,
            "train_normal_median": train_values.median(),
            "test_normal_median": test_values.median(),
            "train_normal_mean": train_values.mean(),
            "test_normal_mean": test_values.mean(),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue
        })


feature_report = pd.DataFrame(feature_results)

feature_report.to_csv(
    OUTPUT_DIR / "targeted_normal_feature_comparison.csv",
    index=False
)


# --------------------------------------------------
# 5. PRINT TOP DIFFERENCES FOR EACH PATTERN
# --------------------------------------------------

for proto, state, service in PATTERNS:

    subset = feature_report[
        (feature_report["proto"] == proto) &
        (feature_report["state"] == state) &
        (feature_report["service"] == service)
    ].copy()

    subset = subset.sort_values(
        "ks_statistic",
        ascending=False
    )

    print(
        f"\nTOP FEATURE DIFFERENCES: "
        f"{proto} | {state} | {service}"
    )

    if subset.empty:
        print("No comparable features.")
    else:
        print(
            subset[
                [
                    "feature",
                    "train_normal_median",
                    "test_normal_median",
                    "ks_statistic"
                ]
            ].head(12).to_string(index=False)
        )


# --------------------------------------------------
# 6. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 65)
print("TARGETED PATTERN ANALYSIS COMPLETED")
print("=" * 65)

print("\nReports saved in:", OUTPUT_DIR)
print("- targeted_pattern_counts.csv")
print("- targeted_normal_feature_comparison.csv")
