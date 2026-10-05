import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

from preprocessing import (
    load_dataset,
    prepare_features,
    split_data,
    create_preprocessor
)


CONTAMINATION = 0.30


def main():

    print("=" * 60)
    print("ASTERIA - ISOLATION FOREST ERROR ANALYSIS")
    print("=" * 60)

   

    df = load_dataset()

    

    X, y = prepare_features(df)

    

    X_train, X_val, y_train, y_val = split_data(
        X,
        y
    )



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

    print("[+] Isolation Forest training completed.")

  

    raw_predictions = model.predict(
        X_val_processed
    )



    y_pred = (raw_predictions == -1).astype(int)

  
    tn, fp, fn, tp = confusion_matrix(
        y_val,
        y_pred
    ).ravel()

    print("\n" + "=" * 60)
    print("ERROR BREAKDOWN")
    print("=" * 60)

    print(f"\nTrue Negatives  : {tn}")
    print(f"False Positives : {fp}")
    print(f"False Negatives : {fn}")
    print(f"True Positives  : {tp}")

    print("\nInterpretation:")

    print(
        f"\nFalse Positives:"
        f" {fp:,} normal network flows "
        f"were classified as anomalies."
    )

    print(
        f"\nFalse Negatives:"
        f" {fn:,} attack network flows "
        f"were classified as normal."
    )

    print(
        f"\nTrue Positives:"
        f" {tp:,} attack network flows "
        f"were correctly detected."
    )

    print(
        f"\nTrue Negatives:"
        f" {tn:,} normal network flows "
        f"were correctly classified."
    )

   

    validation_df = df.loc[X_val.index].copy()

    validation_df["prediction"] = y_pred

    

    false_positives = validation_df[
        (validation_df["label"] == 0) &
        (validation_df["prediction"] == 1)
    ].copy()

    print("\n" + "=" * 60)
    print("FALSE POSITIVE ANALYSIS")
    print("=" * 60)

    print(
        f"\nFalse positives: "
        f"{len(false_positives):,}"
    )

    print("\nMost common protocols:")

    print(
        false_positives["proto"]
        .value_counts()
        .head(10)
        .to_string()
    )

    print("\nMost common services:")

    print(
        false_positives["service"]
        .value_counts()
        .head(10)
        .to_string()
    )

    

    false_negatives = validation_df[
        (validation_df["label"] == 1) &
        (validation_df["prediction"] == 0)
    ].copy()

    print("\n" + "=" * 60)
    print("FALSE NEGATIVE ANALYSIS")
    print("=" * 60)

    print(
        f"\nFalse negatives: "
        f"{len(false_negatives):,}"
    )

    print("\nMissed attack categories:")

    print(
        false_negatives["attack_cat"]
        .value_counts()
        .to_string()
    )

    

    attack_df = validation_df[
        validation_df["label"] == 1
    ].copy()

    category_summary = (
        attack_df
        .groupby("attack_cat")
        .agg(
            total_attacks=("label", "size"),
            detected=("prediction", "sum")
        )
        .reset_index()
    )

    category_summary["missed"] = (
        category_summary["total_attacks"]
        - category_summary["detected"]
    )

    category_summary["recall"] = (
        category_summary["detected"]
        / category_summary["total_attacks"]
    )

    category_summary = category_summary.sort_values(
        "recall"
    )

    print("\n" + "=" * 60)
    print("ATTACK CATEGORY ANALYSIS")
    print("=" * 60)

    print(
        category_summary.to_string(
            index=False
        )
    )

   

    category_summary.to_csv(
        "experiments/attack_category_analysis.csv",
        index=False
    )

    false_positives.to_csv(
        "experiments/false_positives.csv",
        index=False
    )

    false_negatives.to_csv(
        "experiments/false_negatives.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS COMPLETED")
    print("=" * 60)

    print("\nFiles saved:")

    print(
        "  experiments/attack_category_analysis.csv"
    )

    print(
        "  experiments/false_positives.csv"
    )

    print(
        "  experiments/false_negatives.csv"
    )


if __name__ == "__main__":
    main()