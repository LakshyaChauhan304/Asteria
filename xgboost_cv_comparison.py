
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocessing import (
    load_dataset,
    prepare_features,
    create_preprocessor
)

from feature_engineering import engineer_features


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = load_dataset()

X_original, y = prepare_features(df)

# Create engineered version of the same dataset
X_engineered = engineer_features(X_original)

print("\nOriginal feature shape:", X_original.shape)
print("Engineered feature shape:", X_engineered.shape)


# ============================================================
# 2. DEFINE EXPERIMENTS
# ============================================================

candidates = {
    "Original 42 features": X_original,
    "Engineered 48 features": X_engineered
}


# ============================================================
# 3. STRATIFIED 5-FOLD CROSS VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []


# ============================================================
# 4. TRAIN AND EVALUATE EACH MODEL
# ============================================================

for name, candidate_X in candidates.items():

    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)

    fold = 1

    for train_idx, val_idx in skf.split(candidate_X, y):

        X_train = candidate_X.iloc[train_idx]
        X_val = candidate_X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Preprocessing
        preprocessor = create_preprocessor(X_train)

        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)

        # XGBoost model
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

        print(f"\nTraining fold {fold}...")

        model.fit(X_train_processed, y_train)

        # Predictions
        probabilities = model.predict_proba(
            X_val_processed
        )[:, 1]

        predictions = (probabilities >= 0.50).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1]
        ).ravel()

        # Evaluation metrics
        precision = precision_score(
            y_val, predictions, zero_division=0
        )

        recall = recall_score(
            y_val, predictions, zero_division=0
        )

        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        fpr = fp / (fp + tn) if (fp + tn) else 0

        # Store results
        results.append({
            "Model": name,
            "Fold": fold,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "FPR": fpr,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

        # Print fold results
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"False Positive Rate: {fpr:.4f}")

        print(f"TN: {tn} | FP: {fp}")
        print(f"FN: {fn} | TP: {tp}")

        fold += 1


# ============================================================
# 5. DISPLAY FOLD-BY-FOLD RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 90)
print("FOLD-BY-FOLD COMPARISON")
print("=" * 90)

print(results_df.round(4).to_string(index=False))


# ============================================================
# 6. CALCULATE MEAN AND STANDARD DEVIATION
# ============================================================

metrics = [
    "Precision",
    "Recall",
    "F1",
    "FPR"
]

summary = results_df.groupby("Model")[metrics].agg(
    ["mean", "std"]
)

print("\n" + "=" * 90)
print("FIVE-FOLD CROSS-VALIDATION SUMMARY")
print("=" * 90)

print(summary.round(4).to_string())


# ============================================================
# 7. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "xgboost_cv_results.csv",
    index=False
)

summary.to_csv(
    "xgboost_cv_summary.csv"
)

print("\nResults saved successfully:")
print("- xgboost_cv_results.csv")
print("- xgboost_cv_summary.csv")
