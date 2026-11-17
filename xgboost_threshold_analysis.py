
import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier
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


def evaluate_threshold(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_true, predictions, zero_division=0
    )

    recall = recall_score(
        y_true, predictions, zero_division=0
    )

    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions
    ).ravel()

    fpr = fp / (fp + tn)

    return {
        "Threshold": threshold,
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
    print("ASTERIA - XGBOOST THRESHOLD ANALYSIS")
    print("=" * 60)

    df = load_dataset()

    X, y = prepare_features(df)

    X_train, X_val, y_train, y_val = split_data(X, y)

    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

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

    print("[+] Training completed.")

    # Get probabilities instead of fixed predictions
    probabilities = model.predict_proba(
        X_val_processed
    )[:, 1]

    thresholds = [
        0.50, 0.45, 0.40, 0.35,
        0.30, 0.25, 0.20
    ]

    results = []

    for threshold in thresholds:

        metrics = evaluate_threshold(
            y_val,
            probabilities,
            threshold
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("XGBOOST THRESHOLD COMPARISON")
    print("=" * 70)

    print(results_df.to_string(index=False))

    # Evaluate Fuzzers recall at each threshold
    fuzzers_mask = (
        (df.loc[X_val.index, "attack_cat"] == "Fuzzers") &
        (y_val == 1)
    )

    fuzzers_indices = X_val.index[fuzzers_mask]

    fuzzers_positions = X_val.index.get_indexer(
        fuzzers_indices
    )

    fuzzers_actual = y_val.loc[fuzzers_indices]

    fuzzers_probabilities = probabilities[
        fuzzers_positions
    ]

    print("\n" + "=" * 70)
    print("FUZZERS RECALL AT DIFFERENT THRESHOLDS")
    print("=" * 70)

    fuzzers_results = []

    for threshold in thresholds:

        fuzzers_predictions = (
            fuzzers_probabilities >= threshold
        ).astype(int)

        fuzzers_recall = recall_score(
            fuzzers_actual,
            fuzzers_predictions,
            zero_division=0
        )

        fuzzers_results.append({
            "Threshold": threshold,
            "Fuzzers_Recall": fuzzers_recall,
            "Detected": int(fuzzers_predictions.sum()),
            "Missed": int(
                len(fuzzers_predictions) -
                fuzzers_predictions.sum()
            )
        })

    fuzzers_df = pd.DataFrame(fuzzers_results)

    print(fuzzers_df.to_string(index=False))

    # Save results
    Path("experiments").mkdir(exist_ok=True)

    results_df.to_csv(
        "experiments/xgboost_threshold_comparison.csv",
        index=False
    )

    fuzzers_df.to_csv(
        "experiments/xgboost_fuzzers_threshold_analysis.csv",
        index=False
    )

    print("\n[+] Threshold experiments saved.")


if __name__ == "__main__":
    main()
