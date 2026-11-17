
import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier

from preprocessing import (
    prepare_features,
    create_preprocessor
)


def main():

    print("=" * 65)
    print("ASTERIA - TCP FIN DEEP ANALYSIS")
    print("=" * 65)

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

    # Build results
    results = test_df.copy()

    results["prediction"] = predictions
    results["model_probability"] = probabilities

    # Isolate TCP + FIN
    tcp_fin = results[
        (results["proto"] == "tcp") &
        (results["state"] == "FIN")
    ].copy()

    # Four groups
    normal_correct = tcp_fin[
        (tcp_fin["label"] == 0) &
        (tcp_fin["prediction"] == 0)
    ]

    normal_false_positive = tcp_fin[
        (tcp_fin["label"] == 0) &
        (tcp_fin["prediction"] == 1)
    ]

    attack_detected = tcp_fin[
        (tcp_fin["label"] == 1) &
        (tcp_fin["prediction"] == 1)
    ]

    attack_missed = tcp_fin[
        (tcp_fin["label"] == 1) &
        (tcp_fin["prediction"] == 0)
    ]

    # Summary
    print("\nTCP + FIN SUMMARY")
    print("-" * 45)

    print("Total records:", len(tcp_fin))
    print("Normal correctly classified:", len(normal_correct))
    print("Normal false positives:", len(normal_false_positive))
    print("Attacks detected:", len(attack_detected))
    print("Attacks missed:", len(attack_missed))

    print(
        "\nAttack prevalence:",
        round(tcp_fin["label"].mean() * 100, 2),
        "%"
    )

    # Attack categories
    print("\nATTACK CATEGORIES IN TCP + FIN")

    print(
        tcp_fin[tcp_fin["label"] == 1][
            "attack_cat"
        ].value_counts().to_string()
    )

    # Numerical feature comparison
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

    print("\nTCP-FIN: NORMAL TN VS FP FEATURE COMPARISON")

    print(comparison.head(25).to_string())

    # Save
    Path("experiments").mkdir(exist_ok=True)

    comparison.to_csv(
        "experiments/xgboost_tcp_fin_feature_analysis.csv"
    )

    tcp_fin.to_csv(
        "experiments/xgboost_tcp_fin_records.csv",
        index=False
    )

    print("\n[+] TCP FIN analysis saved.")


if __name__ == "__main__":
    main()
