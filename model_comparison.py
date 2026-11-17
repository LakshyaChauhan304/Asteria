
from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

from preprocessing import (
    load_dataset,
    prepare_features,
    split_data,
    create_preprocessor
)


ISOLATION_CONTAMINATION = 0.30


def evaluate_model(y_true, y_pred):

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred
    ).ravel()

    fpr = fp / (fp + tn)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


def main():

    print("=" * 60)
    print("ASTERIA - THREE MODEL COMPARISON")
    print("=" * 60)

    # 1. Load dataset
    df = load_dataset()

    # 2. Prepare features
    X, y = prepare_features(df)

    # 3. Split data
    X_train, X_val, y_train, y_val = split_data(
        X, y
    )

    # 4. Preprocessing
    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_val_processed = preprocessor.transform(
        X_val
    )

    print("\n[+] Preprocessing completed.")

    results = []

    # ==================================================
    # MODEL 1: LOGISTIC REGRESSION
    # ==================================================

    print("\n[+] Training Logistic Regression...")

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    logistic_model.fit(
        X_train_processed,
        y_train
    )

    logistic_pred = logistic_model.predict(
        X_val_processed
    )

    logistic_metrics = evaluate_model(
        y_val, logistic_pred
    )

    logistic_metrics["Model"] = "Logistic Regression"

    results.append(logistic_metrics)

    # ==================================================
    # MODEL 2: ISOLATION FOREST
    # ==================================================

    print("\n[+] Training Isolation Forest...")

    isolation_model = IsolationForest(
        n_estimators=200,
        contamination=ISOLATION_CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )

    isolation_model.fit(X_train_processed)

    isolation_raw = isolation_model.predict(
        X_val_processed
    )

    isolation_pred = (
        isolation_raw == -1
    ).astype(int)

    isolation_metrics = evaluate_model(
        y_val, isolation_pred
    )

    isolation_metrics["Model"] = "Isolation Forest"

    results.append(isolation_metrics)

    # ==================================================
    # MODEL 3: XGBOOST
    # ==================================================

    print("\n[+] Training XGBoost...")

    xgb_model = XGBClassifier(
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

    xgb_model.fit(
        X_train_processed,
        y_train
    )

    print("[+] XGBoost training completed.")

    xgb_pred = xgb_model.predict(
        X_val_processed
    )

    xgb_metrics = evaluate_model(
        y_val, xgb_pred
    )

    xgb_metrics["Model"] = "XGBoost"

    results.append(xgb_metrics)

    # ==================================================
    # XGBOOST CLASSIFICATION REPORT
    # ==================================================

    print("\n" + "=" * 60)
    print("XGBOOST CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_val,
            xgb_pred,
            target_names=["Normal", "Attack"],
            zero_division=0
        )
    )

    # ==================================================
    # MODEL COMPARISON
    # ==================================================

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
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(index=False)
    )

    # ==================================================
    # SAVE RESULTS AND MODEL
    # ==================================================

    Path("experiments").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    results_df.to_csv(
        "experiments/three_model_comparison.csv",
        index=False
    )

    xgb_model.save_model(
        "models/xgboost_model.json"
    )

    print("\n[+] Comparison saved.")
    print("[+] XGBoost model saved.")

    print(
        "\nResults: experiments/three_model_comparison.csv"
    )

    print(
        "Model: models/xgboost_model.json"
    )


if __name__ == "__main__":
    main()
