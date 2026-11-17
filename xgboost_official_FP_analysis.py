
import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix

from preprocessing import (
    prepare_features,
    create_preprocessor
)


TRAIN_PATH = "data/UNSW_NB15_training-set.csv"
TEST_PATH = "data/UNSW_NB15_testing-set.csv"

THRESHOLD = 0.50


def main():

    print("=" * 65)
    print("ASTERIA - OFFICIAL TEST FALSE POSITIVE ANALYSIS")
    print("=" * 65)

    # 1. Load datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train = prepare_features(train_df)

    X_test = test_df.drop(
        columns=["id", "attack_cat", "label"]
    )

    y_test = test_df["label"]

    # 2. Preprocessing
    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 3. Train model
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    print("\n[+] Training XGBoost...")

    model.fit(X_train_processed, y_train)

    # 4. Predict
    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # 5. Identify false positives
    false_positive_mask = (
        (y_test.to_numpy() == 0) &
        (predictions == 1)
    )

    false_positive_indices = test_df.index[
        false_positive_mask
    ]

    false_positives = test_df.loc[
        false_positive_indices
    ].copy()

    false_positives["model_probability"] = (
        probabilities[false_positive_mask]
    )

    print("\n" + "=" * 65)
    print("FALSE POSITIVE SUMMARY")
    print("=" * 65)

    print("Total normal records:", int((y_test == 0).sum()))
    print("False positives:", len(false_positives))

    print("\nFalse positives by protocol:")

    print(
        false_positives["proto"]
        .value_counts()
        .to_string()
    )

    print("\nFalse positives by service:")

    print(
        false_positives["service"]
        .value_counts()
        .head(15)
        .to_string()
    )

    print("\nFalse positives by connection state:")

    print(
        false_positives["state"]
        .value_counts()
        .to_string()
    )

    # 6. Feature distribution comparison
    normal_records = test_df.loc[y_test == 0]

    numerical_features = X_train.select_dtypes(
        include="number"
    ).columns

    comparison = pd.DataFrame({
        "Normal_Mean": normal_records[
            numerical_features
        ].mean(),

        "FP_Mean": false_positives[
            numerical_features
        ].mean(),

        "Normal_Median": normal_records[
            numerical_features
        ].median(),

        "FP_Median": false_positives[
            numerical_features
        ].median()
    })

    comparison["Mean_Difference"] = (
        comparison["FP_Mean"] -
        comparison["Normal_Mean"]
    ).abs()

    comparison = comparison.sort_values(
        "Mean_Difference",
        ascending=False
    )

    print("\n" + "=" * 65)
    print("TOP 20 FALSE POSITIVE FEATURE DIFFERENCES")
    print("=" * 65)

    print(comparison.head(20).to_string())

    # 7. Save results
    Path("experiments").mkdir(exist_ok=True)

    false_positives.to_csv(
        "experiments/xgboost_official_false_positives.csv",
        index=False
    )

    comparison.to_csv(
        "experiments/xgboost_official_fp_feature_analysis.csv"
    )

    print("\n[+] False-positive analysis saved.")


if __name__ == "__main__":
    main()
