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
    print("ASTERIA - CATEGORICAL ERROR ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------
    # Load and prepare data
    # --------------------------------------------------

    df = load_dataset()

    X, y = prepare_features(df)

    X_train, X_val, y_train, y_val = split_data(
        X,
        y
    )

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_val_processed = preprocessor.transform(
        X_val
    )

    # --------------------------------------------------
    # Train Isolation Forest
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

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    predictions = model.predict(
        X_val_processed
    )

    y_pred = (
        predictions == -1
    ).astype(int)

    # --------------------------------------------------
    # Reconstruct validation data
    # --------------------------------------------------

    validation_df = df.loc[X_val.index].copy()

    validation_df["prediction"] = y_pred

    validation_df["error_type"] = "TN"

    validation_df.loc[
        (validation_df["label"] == 0) &
        (validation_df["prediction"] == 1),
        "error_type"
    ] = "FP"

    validation_df.loc[
        (validation_df["label"] == 1) &
        (validation_df["prediction"] == 0),
        "error_type"
    ] = "FN"

    validation_df.loc[
        (validation_df["label"] == 1) &
        (validation_df["prediction"] == 1),
        "error_type"
    ] = "TP"

    # --------------------------------------------------
    # Protocol analysis
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("PROTOCOL ERROR ANALYSIS")
    print("=" * 60)

    protocol_analysis = (
        validation_df
        .groupby(["proto", "error_type"])
        .size()
        .unstack(fill_value=0)
    )

    print(
        protocol_analysis.to_string()
    )

    # --------------------------------------------------
    # Service analysis
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("SERVICE ERROR ANALYSIS")
    print("=" * 60)

    service_analysis = (
        validation_df
        .groupby(["service", "error_type"])
        .size()
        .unstack(fill_value=0)
    )

    print(
        service_analysis
        .sort_values(
            "FP",
            ascending=False
        )
        .head(20)
        .to_string()
    )

    # --------------------------------------------------
    # State analysis
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STATE ERROR ANALYSIS")
    print("=" * 60)

    state_analysis = (
        validation_df
        .groupby(["state", "error_type"])
        .size()
        .unstack(fill_value=0)
    )

    print(
        state_analysis
        .sort_values(
            "FP",
            ascending=False
        )
        .to_string()
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    protocol_analysis.to_csv(
        "experiments/protocol_error_analysis.csv"
    )

    service_analysis.to_csv(
        "experiments/service_error_analysis.csv"
    )

    state_analysis.to_csv(
        "experiments/state_error_analysis.csv"
    )

    print("\n" + "=" * 60)
    print("CATEGORICAL ERROR ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()