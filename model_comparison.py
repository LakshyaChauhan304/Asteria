import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocessing import (
    load_dataset,
    prepare_features,
    split_data,
    create_preprocessor
)


ISOLATION_CONTAMINATION = 0.30


def evaluate_model(y_true, y_pred):
    """
    Calculate common classification metrics.
    """

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    false_positive_rate = fp / (fp + tn)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": false_positive_rate,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


def train_logistic_regression(
    X_train_processed,
    y_train
):
    """
    Train supervised Logistic Regression.
    """

    print("\n[+] Training Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_processed,
        y_train
    )

    print("[+] Logistic Regression completed.")

    return model


def train_isolation_forest(
    X_train_processed
):
    """
    Train unsupervised Isolation Forest.
    """

    print(
        f"\n[+] Training Isolation Forest "
        f"(contamination={ISOLATION_CONTAMINATION})..."
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=ISOLATION_CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_processed
    )

    print("[+] Isolation Forest completed.")

    return model


def main():

    print("=" * 60)
    print("ASTERIA - MODEL COMPARISON")
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
    # 3. Same train/validation split for both models
    # --------------------------------------------------

    X_train, X_val, y_train, y_val = split_data(
        X,
        y
    )

    # --------------------------------------------------
    # 4. Same preprocessing for both models
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

    # ==================================================
    # 5. LOGISTIC REGRESSION
    # ==================================================

    logistic_model = train_logistic_regression(
        X_train_processed,
        y_train
    )

    logistic_predictions = logistic_model.predict(
        X_val_processed
    )

    logistic_metrics = evaluate_model(
        y_val,
        logistic_predictions
    )

    logistic_metrics["Model"] = (
        "Logistic Regression"
    )

    # ==================================================
    # 6. ISOLATION FOREST
    # ==================================================

    isolation_model = train_isolation_forest(
        X_train_processed
    )

    isolation_raw_predictions = (
        isolation_model.predict(
            X_val_processed
        )
    )

    # Isolation Forest:
    #     1  = normal
    #    -1  = anomaly
    #
    # Convert to Asteria convention:
    #     0  = normal
    #     1  = attack/anomaly

    isolation_predictions = (
        isolation_raw_predictions == -1
    ).astype(int)

    isolation_metrics = evaluate_model(
        y_val,
        isolation_predictions
    )

    isolation_metrics["Model"] = (
        "Isolation Forest"
    )

    # --------------------------------------------------
    # 7. Comparison table
    # --------------------------------------------------

    results = [
        logistic_metrics,
        isolation_metrics
    ]

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "Precision",
            "Recall",
            "F1",
            "FPR",
            "TN",
            "FP",
            "FN",
            "TP"
        ]
    ]

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # 8. Save comparison
    # --------------------------------------------------

    results_df.to_csv(
        "experiments/model_comparison.csv",
        index=False
    )

    print(
        "\n[+] Results saved to "
        "experiments/model_comparison.csv"
    )


if __name__ == "__main__":
    main()