
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.stats import ks_2samp


# --------------------------------------------------
# 1. LOAD DATA AND MODEL
# --------------------------------------------------

pipeline = joblib.load(
    "models/xgboost_asteria_pipeline.pkl"
)

test_df = pd.read_csv(
    "data/UNSW_NB15_testing-set.csv"
)

output_dir = Path("analysis/false_positive_patterns")
output_dir.mkdir(parents=True, exist_ok=True)

X = test_df.drop(
    columns=["id", "attack_cat", "label"]
)

y = test_df["label"]

threshold = 0.50

probabilities = pipeline.predict_proba(X)[:, 1]
predictions = (probabilities >= threshold).astype(int)

results = test_df.copy()

results["attack_probability"] = probabilities
results["prediction"] = predictions

# Restrict this investigation to legitimate traffic.
normal = results[results["label"] == 0].copy()

normal["result"] = np.where(
    normal["prediction"] == 1,
    "FALSE_POSITIVE",
    "TRUE_NEGATIVE"
)

print("=" * 65)
print("ASTERIA - FALSE POSITIVE PATTERN ANALYSIS")
print("=" * 65)


# --------------------------------------------------
# 2. PROTOCOL + STATE + SERVICE COMBINATIONS
# --------------------------------------------------

pattern_summary = (
    normal.groupby(
        ["proto", "state", "service"],
        dropna=False
    )
    .agg(
        total_normal=("label", "size"),
        false_positives=("prediction", "sum"),
        mean_probability_all=("attack_probability", "mean")
    )
    .reset_index()
)

pattern_summary["true_negatives"] = (
    pattern_summary["total_normal"]
    - pattern_summary["false_positives"]
)

pattern_summary["false_positive_rate_pct"] = (
    pattern_summary["false_positives"]
    / pattern_summary["total_normal"] * 100
)

# Calculate probability statistics separately.
fp_means = (
    normal[normal["prediction"] == 1]
    .groupby(["proto", "state", "service"])["attack_probability"]
    .mean()
    .rename("mean_fp_probability")
)

tn_means = (
    normal[normal["prediction"] == 0]
    .groupby(["proto", "state", "service"])["attack_probability"]
    .mean()
    .rename("mean_tn_probability")
)

pattern_summary = pattern_summary.merge(
    fp_means,
    on=["proto", "state", "service"],
    how="left"
)

pattern_summary = pattern_summary.merge(
    tn_means,
    on=["proto", "state", "service"],
    how="left"
)

pattern_summary = pattern_summary.sort_values(
    ["false_positives", "false_positive_rate_pct"],
    ascending=False
)

pattern_summary.to_csv(
    output_dir / "protocol_state_service_patterns.csv",
    index=False
)

print("\nTOP 25 PROBLEMATIC COMBINATIONS")

print(
    pattern_summary.head(25).to_string(index=False)
)


# --------------------------------------------------
# 3. STATE-LEVEL PROBABILITY COMPARISON
# --------------------------------------------------

state_summary = (
    normal.groupby(["state", "result"])
    .agg(
        records=("attack_probability", "size"),
        mean_probability=("attack_probability", "mean"),
        median_probability=("attack_probability", "median"),
        p90_probability=(
            "attack_probability",
            lambda x: x.quantile(0.90)
        )
    )
    .reset_index()
)

state_summary.to_csv(
    output_dir / "state_probability_comparison.csv",
    index=False
)

print("\nSTATE PROBABILITY COMPARISON")
print(state_summary.to_string(index=False))


# --------------------------------------------------
# 4. FEATURE COMPARISON WITHIN INT AND FIN
# --------------------------------------------------

categorical = ["proto", "service", "state"]

numerical_features = [
    col for col in X.columns
    if col not in categorical
]

for selected_state in ["INT", "FIN"]:

    state_data = normal[
        normal["state"] == selected_state
    ]

    fp = state_data[
        state_data["prediction"] == 1
    ]

    tn = state_data[
        state_data["prediction"] == 0
    ]

    print(f"\n{'=' * 65}")
    print(f"FEATURE COMPARISON: {selected_state}")
    print("=" * 65)

    print("False positives:", len(fp))
    print("True negatives:", len(tn))

    feature_results = []

    for col in numerical_features:

        fp_values = pd.to_numeric(
            fp[col], errors="coerce"
        ).dropna()

        tn_values = pd.to_numeric(
            tn[col], errors="coerce"
        ).dropna()

        if fp_values.empty or tn_values.empty:
            continue

        ks_stat, ks_pvalue = ks_2samp(
            fp_values,
            tn_values
        )

        feature_results.append({
            "feature": col,
            "fp_median": fp_values.median(),
            "tn_median": tn_values.median(),
            "fp_mean": fp_values.mean(),
            "tn_mean": tn_values.mean(),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue
        })

    feature_report = pd.DataFrame(feature_results)

    feature_report = feature_report.sort_values(
        "ks_statistic",
        ascending=False
    )

    feature_report.to_csv(
        output_dir /
        f"{selected_state.lower()}_feature_comparison.csv",
        index=False
    )

    print("\nTOP 15 DIFFERENTIATING FEATURES")

    print(
        feature_report.head(15).to_string(index=False)
    )


# --------------------------------------------------
# 5. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 65)
print("PATTERN ANALYSIS COMPLETED")
print("=" * 65)

print("\nReports saved inside:", output_dir)
