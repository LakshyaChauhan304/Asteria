
import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier

from preprocessing import (
    prepare_features,
    create_preprocessor
)


def group_fp_analysis(normal_data, fp_data, column):

    total = normal_data[column].value_counts()
    false_positives = fp_data[column].value_counts()

    result = pd.DataFrame({
        "Total_Normal": total,
        "False_Positives": false_positives
    }).fillna(0)

    result["Total_Normal"] = result["Total_Normal"].astype(int)
    result["False_Positives"] = (
        result["False_Positives"].astype(int)
    )

    result["FP_Rate"] = (
        result["False_Positives"] /
        result["Total_Normal"]
    )

    return result.sort_values(
        "False_Positives",
        ascending=False
    )


def main():

    print("=" * 65)
    print("ASTERIA - DEEP FALSE POSITIVE ANALYSIS")
    print("=" * 65)

    # Load datasets
    train_df = pd.read_csv(
        "data/UNSW_NB15_training-set.csv"
    )

    test_df = pd.read_csv(
        "data/UNSW_NB15_testing-set.csv"
    )

    X_train, y_train = prepare_features(train_df)

    X_test = test_df.drop(
        columns=["id", "attack_cat", "label"]
    )

    y_test = test_df["label"]

    # Preprocessing
    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    # Train XGBoost
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

    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    # Separate normal records
    normal_data = test_df.loc[
        y_test == 0
    ].copy()

    normal_predictions = predictions[
        y_test.to_numpy() == 0
    ]

    normal_data["prediction"] = normal_predictions

    # TN and FP groups
    fp_data = normal_data[
        normal_data["prediction"] == 1
    ]

    tn_data = normal_data[
        normal_data["prediction"] == 0
    ]

    print("\nNormal records:", len(normal_data))
    print("True negatives:", len(tn_data))
    print("False positives:", len(fp_data))

    # Analyse categorical features
    for column in ["proto", "service", "state"]:

        print("\n" + "=" * 65)
        print(f"FALSE POSITIVE RATE BY {column.upper()}")
        print("=" * 65)

        result = group_fp_analysis(
            normal_data,
            fp_data,
            column
        )

        print(result.to_string())

        Path("experiments").mkdir(
            exist_ok=True
        )

        result.to_csv(
            f"experiments/fp_rate_by_{column}.csv"
        )

    # Compare TN and FP numerical distributions
    numerical_features = X_train.select_dtypes(
        include="number"
    ).columns

    comparison = pd.DataFrame({
        "TN_Median": tn_data[numerical_features].median(),
        "FP_Median": fp_data[numerical_features].median(),

        "TN_Q25": tn_data[numerical_features].quantile(0.25),
        "FP_Q25": fp_data[numerical_features].quantile(0.25),

        "TN_Q75": tn_data[numerical_features].quantile(0.75),
        "FP_Q75": fp_data[numerical_features].quantile(0.75)
    })

    comparison["Median_Difference"] = (
        comparison["TN_Median"] -
        comparison["FP_Median"]
    ).abs()

    comparison = comparison.sort_values(
        "Median_Difference",
        ascending=False
    )

    print("\n" + "=" * 65)
    print("TRUE NEGATIVE VS FALSE POSITIVE DISTRIBUTIONS")
    print("=" * 65)

    print(comparison.head(25).to_string())

    comparison.to_csv(
        "experiments/xgboost_fp_vs_tn_distributions.csv"
    )

    print("\n[+] Deep false-positive analysis completed.")


if __name__ == "__main__":
    main()
