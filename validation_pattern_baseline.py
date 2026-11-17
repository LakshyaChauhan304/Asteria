
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from xgboost import XGBClassifier

from preprocessing import create_preprocessor


# --------------------------------------------------
# 1. LOAD TRAINING DATASET
# --------------------------------------------------

df = pd.read_csv(
    "data/UNSW_NB15_training-set.csv"
)

OUTPUT_DIR = Path("analysis/validation_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("ASTERIA - VALIDATION PATTERN BASELINE")
print("=" * 65)


# --------------------------------------------------
# 2. PREPARE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(
    columns=["id", "attack_cat", "label"]
)

y = df["label"]

# Preserve the original row information for pattern analysis.
metadata = df[
    ["proto", "state", "service"]
].copy()

X_train, X_val, y_train, y_val, meta_train, meta_val = (
    train_test_split(
        X,
        y,
        metadata,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))


# --------------------------------------------------
# 3. BUILD MODEL
# --------------------------------------------------

preprocessor = create_preprocessor(X_train)

classifier = XGBClassifier(
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

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])


# --------------------------------------------------
# 4. TRAIN MODEL
# --------------------------------------------------

print("\nTraining XGBoost baseline...")

model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 5. VALIDATION PREDICTIONS
# --------------------------------------------------

THRESHOLD = 0.50

probabilities = model.predict_proba(X_val)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)

precision = precision_score(
    y_val, predictions, zero_division=0
)

recall = recall_score(
    y_val, predictions, zero_division=0
)

f1 = f1_score(
    y_val, predictions, zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_val,
    predictions,
    labels=[0, 1]
).ravel()

fpr = fp / (fp + tn)

print("\nOVERALL VALIDATION RESULTS")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"FPR:       {fpr:.4f}")

print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")


# --------------------------------------------------
# 6. BUILD VALIDATION RESULTS
# --------------------------------------------------

validation_results = meta_val.copy()

validation_results["actual"] = y_val.values
validation_results["prediction"] = predictions
validation_results["attack_probability"] = probabilities

validation_results.to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index=False
)


# --------------------------------------------------
# 7. ANALYSE THE FOUR PROBLEMATIC COMBINATIONS
# --------------------------------------------------

PATTERNS = [
    ("tcp", "FIN", "-"),
    ("udp", "INT", "-"),
    ("tcp", "FIN", "http"),
    ("tcp", "FIN", "ftp")
]

pattern_results = []

for proto, state, service in PATTERNS:

    subset = validation_results[
        (validation_results["proto"].astype(str) == proto) &
        (validation_results["state"].astype(str) == state) &
        (validation_results["service"].astype(str) == service)
    ]

    normal = subset[subset["actual"] == 0]
    attacks = subset[subset["actual"] == 1]

    fp = int((normal["prediction"] == 1).sum())
    tn_count = int((normal["prediction"] == 0).sum())

    tp_count = int((attacks["prediction"] == 1).sum())
    fn_count = int((attacks["prediction"] == 0).sum())

    normal_fpr = (
        fp / len(normal) if len(normal) else np.nan
    )

    attack_recall = (
        tp_count / len(attacks) if len(attacks) else np.nan
    )

    pattern_results.append({
        "proto": proto,
        "state": state,
        "service": service,
        "normal_records": len(normal),
        "attack_records": len(attacks),
        "false_positives": fp,
        "true_negatives": tn_count,
        "true_positives": tp_count,
        "false_negatives": fn_count,
        "normal_fpr_pct": normal_fpr * 100,
        "attack_recall_pct": attack_recall * 100
    })


pattern_df = pd.DataFrame(pattern_results)

pattern_df.to_csv(
    OUTPUT_DIR / "validation_pattern_metrics.csv",
    index=False
)

print("\nVALIDATION RESULTS BY PROBLEMATIC COMBINATION")

print(pattern_df.to_string(index=False))


# --------------------------------------------------
# 8. SAVE SUMMARY
# --------------------------------------------------

summary = pd.DataFrame([{
    "threshold": THRESHOLD,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "fpr": fpr,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp
}])

summary.to_csv(
    OUTPUT_DIR / "validation_overall_metrics.csv",
    index=False
)

print("\n" + "=" * 65)
print("VALIDATION BASELINE COMPLETED")
print("=" * 65)

print("\nResults saved in:", OUTPUT_DIR)
