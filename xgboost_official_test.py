
import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocessing import (
    load_dataset,
    prepare_features,
    create_preprocessor
)


TRAIN_PATH = "data/UNSW_NB15_training-set.csv"
TEST_PATH = "data/UNSW_NB15_testing-set.csv"

THRESHOLDS = [0.50, 0.45, 0.35]


def evaluate_model(y_true, probabilities, threshold):

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
        y_true, predictions, labels=[0, 1]
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

    print("=" * 65)
    print("ASTERIA - OFFICIAL UNSW-NB15 TEST EVALUATION")
    print("=" * 65)

    # 1. Load both datasets
    print("\n[+] Loading datasets...")

    train_df = load_dataset(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("\nTraining shape:", train_df.shape)
    print("Testing shape :", test_df.shape)

    # 2. Prepare training features
    X_train, y_train = prepare_features(train_df)

    # 3. Prepare official testing features
    y_test = test_df["label"]

    X_test = test_df.drop(
        columns=["id", "attack_cat", "label"]
    )

    # Verify feature consistency
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing feature columns do not match!"
        )

    print("\n[+] Feature columns verified.")

    # 4. Fit preprocessing ONLY on training data
    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    print("\n[+] Preprocessing completed.")
    print("Training processed:", X_train_processed.shape)
    print("Testing processed :", X_test_processed.shape)

    # 5. Train XGBoost
    print("\n[+] Training XGBoost on complete training dataset...")

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

    model.fit(X_train_processed, y_train)

    print("[+] Training completed.")

    # 6. Obtain attack probabilities
    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    # 7. Evaluate overall metrics
    overall_results = []

    for threshold in THRESHOLDS:

        metrics = evaluate_model(
            y_test,
            probabilities,
            threshold
        )

        overall_results.append(metrics)

    overall_df = pd.DataFrame(overall_results)

    print("\n" + "=" * 65)
    print("OFFICIAL TEST - OVERALL MODEL PERFORMANCE")
    print("=" * 65)

    print(overall_df.to_string(index=False))

    # 8. Classification report at threshold 0.45
    selected_threshold = 0.45

    selected_predictions = (
        probabilities >= selected_threshold
    ).astype(int)

    print("\n" + "=" * 65)
    print("CLASSIFICATION REPORT - THRESHOLD 0.45")
    print("=" * 65)

    print(
        classification_report(
            y_test,
            selected_predictions,
            target_names=["Normal", "Attack"],
            zero_division=0
        )
    )

    # 9. Attack-category recall
    category_results = []

    attack_categories = test_df["attack_cat"]

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        for category in sorted(attack_categories.unique()):

            mask = (
                (attack_categories == category) &
                (y_test == 1)
            ).to_numpy()

            if mask.sum() == 0:
                continue

            actual = y_test.to_numpy()[mask]
            predicted = predictions[mask]

            detected = int(predicted.sum())
            total = int(mask.sum())

            category_results.append({
                "Threshold": threshold,
                "Attack_Category": category,
                "Total": total,
                "Detected": detected,
                "Missed": total - detected,
                "Recall": recall_score(
                    actual,
                    predicted,
                    zero_division=0
                )
            })

    category_df = pd.DataFrame(category_results)

    print("\n" + "=" * 75)
    print("OFFICIAL TEST - ATTACK CATEGORY RECALL")
    print("=" * 75)

    print(category_df.to_string(index=False))

    # 10. Save results
    Path("experiments").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    overall_df.to_csv(
        "experiments/xgboost_official_test_metrics.csv",
        index=False
    )

    category_df.to_csv(
        "experiments/xgboost_official_test_categories.csv",
        index=False
    )

    model.save_model(
        "models/xgboost_official_test_model.json"
    )

    print("\n[+] Official test evaluation completed.")
    print("[+] Overall metrics saved.")
    print("[+] Category metrics saved.")
    print("[+] Model saved.")


if __name__ == "__main__":
    main()
