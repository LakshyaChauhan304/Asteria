
import pandas as pd
import numpy as np

from scipy.stats import ks_2samp
from pathlib import Path


# --------------------------------------------------
# 1. LOAD DATASETS
# --------------------------------------------------

TRAIN_PATH = Path("data/UNSW_NB15_training-set.csv")
TEST_PATH = Path("data/UNSW_NB15_testing-set.csv")

OUTPUT_DIR = Path("analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("=" * 60)
print("ASTERIA - DISTRIBUTION SHIFT ANALYSIS")
print("=" * 60)

print("\nTRAINING DATA:", train_df.shape)
print("TESTING DATA:", test_df.shape)


# --------------------------------------------------
# 2. IDENTIFY MODEL INPUT FEATURES
# --------------------------------------------------

excluded_columns = ["id", "attack_cat", "label"]

feature_columns = [
    col for col in train_df.columns
    if col not in excluded_columns
]

train_features = train_df[feature_columns].copy()
test_features = test_df[feature_columns].copy()

# Ensure both datasets have exactly the same input columns.
test_features = test_features[feature_columns]

categorical_features = [
    col for col in ["proto", "service", "state"]
    if col in feature_columns
]

numerical_features = [
    col for col in feature_columns
    if col not in categorical_features
]

print("\nNumber of model input features:", len(feature_columns))
print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))


# --------------------------------------------------
# 3. DATA QUALITY CHECK
# --------------------------------------------------

quality_results = []

for dataset_name, df in [
    ("TRAIN", train_df),
    ("TEST", test_df)
]:
    for col in feature_columns:
        quality_results.append({
            "dataset": dataset_name,
            "feature": col,
            "missing_values": df[col].isna().sum(),
            "missing_percentage": df[col].isna().mean() * 100
        })

quality_df = pd.DataFrame(quality_results)

quality_df.to_csv(
    OUTPUT_DIR / "data_quality_report.csv",
    index=False
)

print("\nDATA QUALITY CHECK")
print("Training missing values:", train_features.isna().sum().sum())
print("Testing missing values:", test_features.isna().sum().sum())


# --------------------------------------------------
# 4. NUMERICAL DISTRIBUTION COMPARISON
# --------------------------------------------------

def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index.
    Bins are defined using training data quantiles.
    """

    expected = pd.to_numeric(
        expected, errors="coerce"
    ).dropna()

    actual = pd.to_numeric(
        actual, errors="coerce"
    ).dropna()

    if expected.empty or actual.empty:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(expected.quantile(quantiles).values)

    # Constant or insufficiently varied feature.
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

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return float(psi)


numerical_results = []

for col in numerical_features:

    train_values = pd.to_numeric(
        train_features[col], errors="coerce"
    ).dropna()

    test_values = pd.to_numeric(
        test_features[col], errors="coerce"
    ).dropna()

    ks_stat, ks_pvalue = ks_2samp(
        train_values,
        test_values
    )

    psi = calculate_psi(
        train_values,
        test_values
    )

    numerical_results.append({
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

numerical_df = pd.DataFrame(numerical_results)

numerical_df = numerical_df.sort_values(
    by="psi",
    ascending=False
)

numerical_df.to_csv(
    OUTPUT_DIR / "numerical_distribution_report.csv",
    index=False
)

print("\nTOP 15 NUMERICAL FEATURES WITH HIGHEST PSI")
print(
    numerical_df[
        ["feature", "psi", "ks_statistic", "ks_pvalue"]
    ].head(15).to_string(index=False)
)


# --------------------------------------------------
# 5. CATEGORICAL DISTRIBUTION COMPARISON
# --------------------------------------------------

categorical_results = []

for col in categorical_features:

    train_values = train_features[col].fillna(
        "<MISSING>"
    ).astype(str)

    test_values = test_features[col].fillna(
        "<MISSING>"
    ).astype(str)

    all_categories = set(train_values.unique()) | set(
        test_values.unique()
    )

    train_distribution = train_values.value_counts(
        normalize=True
    )

    test_distribution = test_values.value_counts(
        normalize=True
    )

    train_distribution = train_distribution.reindex(
        all_categories, fill_value=0
    )

    test_distribution = test_distribution.reindex(
        all_categories, fill_value=0
    )

    total_variation_distance = (
        0.5 * np.abs(
            train_distribution - test_distribution
        ).sum()
    )

    unseen_test_categories = (
        set(test_values.unique())
        - set(train_values.unique())
    )

    categorical_results.append({
        "feature": col,
        "train_unique_values": train_values.nunique(),
        "test_unique_values": test_values.nunique(),
        "unseen_test_categories": len(
            unseen_test_categories
        ),
        "unseen_category_names": ", ".join(
            sorted(unseen_test_categories)
        ),
        "total_variation_distance": total_variation_distance
    })

categorical_df = pd.DataFrame(categorical_results)

categorical_df = categorical_df.sort_values(
    by="total_variation_distance",
    ascending=False
)

categorical_df.to_csv(
    OUTPUT_DIR / "categorical_distribution_report.csv",
    index=False
)

print("\nCATEGORICAL DISTRIBUTION COMPARISON")
print(categorical_df.to_string(index=False))


# --------------------------------------------------
# 6. TARGET DISTRIBUTION COMPARISON
# --------------------------------------------------

print("\nTARGET DISTRIBUTION")

for name, df in [
    ("TRAINING", train_df),
    ("TESTING", test_df)
]:

    print(f"\n{name}")

    print(
        df["label"].value_counts().sort_index()
    )

    print(
        df["label"].value_counts(
            normalize=True
        ).sort_index() * 100
    )


# --------------------------------------------------
# 7. SAVE COMPLETE SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)

print("\nReports saved inside the 'analysis' folder:")

for file in OUTPUT_DIR.glob("*.csv"):
    print("-", file)

print("\nDo not modify or retrain the saved model yet.")
print("Review these reports before choosing the next experiment.")
