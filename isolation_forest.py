import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing import (
    load_dataset,
    prepare_features,
    split_data,
    create_preprocessor
)


def build_isolation_forest():

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    return model


def main():

    print("=" * 60)
    print("ASTERIA - ISOLATION FOREST")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------
    # 2. Prepare features
    # --------------------------------------------------

    X, y = prepare_features(df)

    # --------------------------------------------------
    # 3. Split data
    # --------------------------------------------------

    X_train, X_val, y_train, y_val = split_data(
        X,
        y
    )

    # --------------------------------------------------
    # 4. Create preprocessing pipeline
    # --------------------------------------------------

    preprocessor = create_preprocessor(
        X_train
    )

    print("\n[+] Fitting preprocessing pipeline...")

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_val_processed = preprocessor.transform(
        X_val
    )

    print("[+] Preprocessing completed.")

    print(
        f"[+] Training matrix: "
        f"{X_train_processed.shape}"
    )

    print(
        f"[+] Validation matrix: "
        f"{X_val_processed.shape}"
    )

    # --------------------------------------------------
    # 5. Build Isolation Forest
    # --------------------------------------------------

    model = build_isolation_forest()

    # --------------------------------------------------
    # 6. Train
    # --------------------------------------------------

    print("\n[+] Training Isolation Forest...")

    model.fit(
        X_train_processed
    )

    print("[+] Isolation Forest training completed.")

    # --------------------------------------------------
    # 7. Predict
    # --------------------------------------------------

    print("\n[+] Detecting anomalies...")

    predictions = model.predict(
        X_val_processed
    )

    # Isolation Forest:
    #
    #  1  = normal
    # -1  = anomaly
    #
    # Convert to Asteria convention:
    #
    #  0 = normal
    #  1 = attack

    y_pred = (
        predictions == -1
    ).astype(int)

    # --------------------------------------------------
    # 8. Evaluation
    # --------------------------------------------------

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        y_pred
    ).ravel()

    false_positive_rate = (
        fp / (fp + tn)
    )

    

    print("\n" + "=" * 60)
    print("ISOLATION FOREST RESULTS")
    print("=" * 60)

    print(
        f"\nPrecision           : {precision:.4f}"
    )

    print(
        f"Recall              : {recall:.4f}"
    )

    print(
        f"F1 Score            : {f1:.4f}"
    )

    print(
        f"False Positive Rate : {false_positive_rate:.4f}"
    )

    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)

    print(
        f"\nTrue Negatives  : {tn}"
    )

    print(
        f"False Positives : {fp}"
    )

    print(
        f"False Negatives : {fn}"
    )

    print(
        f"True Positives  : {tp}"
    )

    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)

    print(
        classification_report(
            y_val,
            y_pred,
            target_names=[
                "Normal",
                "Attack"
            ],
            zero_division=0
        )
    )

    print("\nRaw Isolation Forest predictions:")
    print(pd.Series(predictions).value_counts())


    print("\nAsteria predictions:")
    print(pd.Series(y_pred).value_counts())


if __name__ == "__main__":
    main()