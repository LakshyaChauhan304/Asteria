
import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier

from preprocessing import (
    prepare_features,
    create_preprocessor
)


def main():

    print("=" * 65)
    print("ASTERIA - UDP INT DEEP ANALYSIS")
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

    # Train model
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

    # Build results dataframe
    results = test_df.copy()

    results["prediction"] = predictions
    results["model_probability"] = probabilities

    # Isolate UDP + INT
    udp_int = results[
        (results["proto"] == "udp") &
        (results["state"] == "INT")
    ].copy()

    # Separate four prediction groups
    normal_correct = udp_int[
        (udp_int["label"] == 0) &
        (udp_int["prediction"] == 0)
    ]

    normal_false_positive = udp_int[
        (udp_int["label"] == 0) &
        (udp_int["prediction"] == 1)
    ]

    attack_detected = udp_int[
        (udp_int["label"] == 1) &
        (udp_int["prediction"] == 1)
    ]

    attack_missed = udp_int[
        (udp_int["label"] == 1) &
        (udp_int["prediction"] == 0)
    ]

    # Summary
    print("\nUDP + INT SUMMARY")
    print("-" * 45)

    print("Total UDP + INT records:", len(udp_int))
    print("Normal correctly classified:", len(normal_correct))
    print("Normal false positives:", len(normal_false_positive))
    print("Attacks detected:", len(attack_detected))
    print("Attacks missed:", len(attack_missed))

    if len(udp_int) > 0:
        print(
            "\nOverall UDP + INT attack prevalence:",
            round(udp_int["label"].mean() * 100, 2),
            "%"
        )

    # Attack categories within UDP + INT
    print("\nATTACK CATEGORIES IN UDP + INT")

    print(
        udp_int[udp_int["label"] == 1][
            "attack_cat"
        ].value_counts().to_string()
    )

    # Compare normal TN vs normal FP
    numerical_features = X_train.select_dtypes(
        include="number"
    ).columns

    comparison = pd.DataFrame({
        "TN_Median": normal_correct[
            numerical_features
        ].median(),

        "FP_Median": normal_false_positive[
            numerical_features
        ].median(),

        "TN_Mean": normal_correct[
            numerical_features
        ].mean(),

        "FP_Mean": normal_false_positive[
            numerical_features
        ].mean()
    })

    comparison["Median_Difference"] = (
        comparison["TN_Median"] -
        comparison["FP_Median"]
    ).abs()

    comparison = comparison.sort_values(
        "Median_Difference",
        ascending=False
    )

    print("\nNORMAL UDP-INT: TN VS FP FEATURE COMPARISON")

    print(comparison.head(25).to_string())

    # Save results
    Path("experiments").mkdir(exist_ok=True)

    comparison.to_csv(
        "experiments/xgboost_udp_int_feature_analysis.csv"
    )

    udp_int.to_csv(
        "experiments/xgboost_udp_int_records.csv",
        index=False
    )

    print("\n[+] UDP INT analysis saved.")


if __name__ == "__main__":
    main()
