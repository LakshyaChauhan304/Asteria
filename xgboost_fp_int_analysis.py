
import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier

from preprocessing import (
    prepare_features,
    create_preprocessor
)


def main():

    print("=" * 65)
    print("ASTERIA - PROTOCOL AND STATE INTERACTION ANALYSIS")
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

    # Build evaluation dataframe
    results = test_df[
        ["proto", "service", "state", "label"]
    ].copy()

    results["prediction"] = predictions

    # Restrict analysis to normal records
    normal_results = results[
        results["label"] == 0
    ].copy()

    normal_results["is_false_positive"] = (
        normal_results["prediction"] == 1
    )

    # Group by protocol and state
    interaction = (
        normal_results
        .groupby(["proto", "state"])
        .agg(
            Total_Normal=("label", "size"),
            False_Positives=("is_false_positive", "sum")
        )
        .reset_index()
    )

    interaction["FP_Rate"] = (
        interaction["False_Positives"] /
        interaction["Total_Normal"]
    )

    interaction = interaction.sort_values(
        by="False_Positives",
        ascending=False
    )

    print("\n" + "=" * 75)
    print("FALSE POSITIVES BY PROTOCOL AND STATE")
    print("=" * 75)

    print(interaction.to_string(index=False))

    # Save results
    Path("experiments").mkdir(exist_ok=True)

    interaction.to_csv(
        "experiments/xgboost_fp_protocol_state.csv",
        index=False
    )

    print("\n[+] Interaction analysis saved.")


if __name__ == "__main__":
    main()
