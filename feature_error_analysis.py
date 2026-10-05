import pandas as pd

from sklearn.ensemble import IsolationForest

from preprocessing import (
    load_dataset,
    prepare_features,
    split_data,
    create_preprocessor
)


CONTAMINATION = 0.30


def main():

    print("=" * 60)
    print("ASTERIA - FEATURE ERROR ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    df = load_dataset()

    X, y = prepare_features(df)

    # --------------------------------------------------
    # 2. Split data
    # --------------------------------------------------

    X_train, X_val, y_train, y_val = split_data(
        X,
        y
    )

    # --------------------------------------------------
    # 3. Preprocessing
    # --------------------------------------------------

    preprocessor = create_preprocessor(
        X_train
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_val_processed = preprocessor.transform(
        X_val
    )

    print("\n[+] Preprocessing completed.")

    # --------------------------------------------------
    # 4. Train Isolation Forest
    # --------------------------------------------------

    print(
        f"\n[+] Training Isolation Forest "
        f"(contamination={CONTAMINATION})..."
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_processed)

    print("[+] Model training completed.")

    # --------------------------------------------------
    # 5. Predictions
    # --------------------------------------------------

    raw_predictions = model.predict(
        X_val_processed
    )

    y_pred = (
        raw_predictions == -1
    ).astype(int)

    # --------------------------------------------------
    # 6. Reconstruct validation dataframe
    # --------------------------------------------------

    validation_df = df.loc[X_val.index].copy()

    validation_df["prediction"] = y_pred

    # --------------------------------------------------
    # 7. Define error groups
    # --------------------------------------------------

    true_positive = validation_df[
        (validation_df["label"] == 1) &
        (validation_df["prediction"] == 1)
    ]

    false_negative = validation_df[
        (validation_df["label"] == 1) &
        (validation_df["prediction"] == 0)
    ]

    true_negative = validation_df[
        (validation_df["label"] == 0) &
        (validation_df["prediction"] == 0)
    ]

    false_positive = validation_df[
        (validation_df["label"] == 0) &
        (validation_df["prediction"] == 1)
    ]

    print("\n" + "=" * 60)
    print("ERROR GROUP SIZES")
    print("=" * 60)

    print(f"\nTrue Positives : {len(true_positive):,}")
    print(f"False Negatives: {len(false_negative):,}")
    print(f"True Negatives : {len(true_negative):,}")
    print(f"False Positives: {len(false_positive):,}")

    # --------------------------------------------------
    # 8. Numerical feature comparison
    # --------------------------------------------------

    numerical_features = [
        column
        for column in X.columns
        if column not in [
            "proto",
            "service",
            "state"
        ]
    ]

    print("\n" + "=" * 60)
    print("TRUE POSITIVE vs FALSE NEGATIVE")
    print("=" * 60)

    comparison = []

    for feature in numerical_features:

        tp_mean = true_positive[feature].mean()
        fn_mean = false_negative[feature].mean()

        comparison.append({
            "feature": feature,
            "TP_mean": tp_mean,
            "FN_mean": fn_mean,
            "absolute_difference": abs(
                tp_mean - fn_mean
            )
        })

    comparison_df = pd.DataFrame(
        comparison
    )

    comparison_df = comparison_df.sort_values(
        "absolute_difference",
        ascending=False
    )

    print(
        comparison_df.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # 9. Normal traffic comparison
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TRUE NEGATIVE vs FALSE POSITIVE")
    print("=" * 60)

    normal_comparison = []

    for feature in numerical_features:

        tn_mean = true_negative[feature].mean()
        fp_mean = false_positive[feature].mean()

        normal_comparison.append({
            "feature": feature,
            "TN_mean": tn_mean,
            "FP_mean": fp_mean,
            "absolute_difference": abs(
                tn_mean - fp_mean
            )
        })

    normal_comparison_df = pd.DataFrame(
        normal_comparison
    )

    normal_comparison_df = (
        normal_comparison_df
        .sort_values(
            "absolute_difference",
            ascending=False
        )
    )

    print(
        normal_comparison_df.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # 10. Save results
    # --------------------------------------------------

    comparison_df.to_csv(
        "experiments/tp_vs_fn_features.csv",
        index=False
    )

    normal_comparison_df.to_csv(
        "experiments/tn_vs_fp_features.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("FEATURE ERROR ANALYSIS COMPLETED")
    print("=" * 60)

    print(
        "\nSaved:"
        "\n  experiments/tp_vs_fn_features.csv"
        "\n  experiments/tn_vs_fp_features.csv"
    )


if __name__ == "__main__":
    main()