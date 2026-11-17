
import pandas as pd
import numpy as np

from scipy.stats import ks_2samp
from pathlib import Path


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

train_df = pd.read_csv("data/UNSW_NB15_training-set.csv")
test_df = pd.read_csv("data/UNSW_NB15_testing-set.csv")

output_dir = Path("analysis")
output_dir.mkdir(exist_ok=True)

# Keep only legitimate traffic.
train_normal = train_df[train_df["label"] == 0].copy()
test_normal = test_df[test_df["label"] == 0].copy()

print("=" * 60)
print("ASTERIA - NORMAL TRAFFIC DISTRIBUTION ANALYSIS")
print("=" * 60)

print("Training normal records:", len(train_normal))
print("Testing normal records:", len(test_normal))


# --------------------------------------------------
# 2. SELECT MODEL FEATURES
# --------------------------------------------------

excluded = ["id", "attack_cat", "label"]

features = [
    col for col in train_df.columns
    if col not in excluded
]

categorical_features = [
    col for col in ["proto", "service", "state"]
    if col in features
]

numerical_features = [
    col for col in features
    if col not in categorical_features
]


# --------------------------------------------------
# 3. PSI FUNCTION
# --------------------------------------------------

def calculate_psi(expected, actual, bins=10):

    expected = pd.to_numeric(
        expected, errors="coerce"
    ).dropna()

    actual = pd.to_numeric(
        actual, errors="coerce"
    ).dropna()

    if expected.empty or actual.empty:
        return np.nan

    edges = np.unique(
        expected.quantile(
            np.linspace(0, 1, bins + 1)
        ).values
    )

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_counts = np.histogram(
        expected, bins=edges
    )[0]

    actual_counts = np.histogram(
        actual, bins=edges
    )[0]

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    epsilon = 1e-6

    expected_pct = np.clip(expected_pct, epsilon, None)
    actual_pct = np.clip(actual_pct, epsilon, None)

    return float(
        np.sum(
            (actual_pct - expected_pct)
            * np.log(actual_pct / expected_pct)
        )
    )


# --------------------------------------------------
# 4. NUMERICAL COMPARISON
# --------------------------------------------------

results = []

for col in numerical_features:

    train_values = pd.to_numeric(
        train_normal[col], errors="coerce"
    ).dropna()

    test_values = pd.to_numeric(
        test_normal[col], errors="coerce"
    ).dropna()

    ks_stat, ks_pvalue = ks_2samp(
        train_values,
        test_values
    )

    psi = calculate_psi(
        train_values,
        test_values
    )

    results.append({
        "feature": col,
        "train_mean": train_values.mean(),
        "test_mean": test_values.mean(),
        "train_median": train_values.median(),
        "test_median": test_values.median(),
        "train_std": train_values.std(),
        "test_std": test_values.std(),
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
        "psi": psi
    })

numerical_report = pd.DataFrame(results)

numerical_report = numerical_report.sort_values(
    "psi", ascending=False
)

numerical_report.to_csv(
    output_dir / "normal_traffic_numerical_report.csv",
    index=False
)

print("\nTOP 15 NUMERICAL FEATURES")
print(
    numerical_report[
        ["feature", "psi", "ks_statistic", "ks_pvalue"]
    ].head(15).to_string(index=False)
)


# --------------------------------------------------
# 5. CATEGORICAL COMPARISON
# --------------------------------------------------

categorical_results = []

for col in categorical_features:

    train_values = train_normal[col].fillna(
        "<MISSING>"
    ).astype(str)

    test_values = test_normal[col].fillna(
        "<MISSING>"
    ).astype(str)

    train_dist = train_values.value_counts(normalize=True)
    test_dist = test_values.value_counts(normalize=True)

    categories = set(train_dist.index) | set(test_dist.index)

    train_dist = train_dist.reindex(
        categories, fill_value=0
    )

    test_dist = test_dist.reindex(
        categories, fill_value=0
    )

    distance = 0.5 * np.abs(
        train_dist - test_dist
    ).sum()

    unseen = set(test_values.unique()) - set(
        train_values.unique()
    )

    categorical_results.append({
        "feature": col,
        "train_unique_values": train_values.nunique(),
        "test_unique_values": test_values.nunique(),
        "unseen_test_categories": len(unseen),
        "unseen_category_names": ", ".join(sorted(unseen)),
        "total_variation_distance": distance
    })

categorical_report = pd.DataFrame(categorical_results)

categorical_report = categorical_report.sort_values(
    "total_variation_distance",
    ascending=False
)

categorical_report.to_csv(
    output_dir / "normal_traffic_categorical_report.csv",
    index=False
)

print("\nCATEGORICAL COMPARISON")
print(categorical_report.to_string(index=False))


# --------------------------------------------------
# 6. PROTOCOL AND STATE BREAKDOWN
# --------------------------------------------------

for col in ["proto", "state", "service"]:

    print(f"\n{col.upper()} DISTRIBUTION")

    comparison = pd.concat(
        [
            train_normal[col].value_counts(
                normalize=True
            ).rename("training"),
            test_normal[col].value_counts(
                normalize=True
            ).rename("testing")
        ],
        axis=1
    ).fillna(0)

    comparison["difference"] = (
        comparison["testing"] - comparison["training"]
    )

    comparison = comparison.sort_values(
        "difference",
        key=lambda x: x.abs(),
        ascending=False
    )

    print(comparison.head(15).to_string())


# --------------------------------------------------
# 7. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 60)
print("NORMAL TRAFFIC ANALYSIS COMPLETED")
print("=" * 60)

print("\nReports saved:")
print("- analysis/normal_traffic_numerical_report.csv")
print("- analysis/normal_traffic_categorical_report.csv")
