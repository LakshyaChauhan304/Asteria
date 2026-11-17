
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. LOAD SAVED MODEL AND METADATA
# ============================================================

MODEL_PATH = "models/xgboost_asteria_pipeline.pkl"
METADATA_PATH = "models/model_metadata.json"
TEST_DATA_PATH = "data/UNSW_NB15_testing-set.csv"

print("=" * 70)
print("ASTERIA - SAVED MODEL TESTING")
print("=" * 70)

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

threshold = metadata["threshold"]

print("\nModel loaded successfully.")
print("Threshold:", threshold)


# ============================================================
# 2. LOAD OFFICIAL TEST DATASET
# ============================================================

df = pd.read_csv(TEST_DATA_PATH)

print("\nTesting dataset shape:", df.shape)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

y_test = df["label"]

X_test = df.drop(
    columns=["id", "attack_cat", "label"]
)

# Ensure feature order matches training metadata
expected_features = metadata["input_features"]

X_test = X_test[expected_features]

print("Testing feature shape:", X_test.shape)


# ============================================================
# 4. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)


# ============================================================
# 5. EVALUATE MODEL
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

fpr = fp / (fp + tn)

print("\n" + "=" * 70)
print("OFFICIAL TEST RESULTS")
print("=" * 70)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"False Positive Rate: {fpr:.4f}")

print("\nConfusion Matrix:")
print(f"TN: {tn} | FP: {fp}")
print(f"FN: {fn} | TP: {tp}")


# ============================================================
# 6. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Attack"],
        zero_division=0
    )
)


# ============================================================
# 7. SAVE PREDICTIONS
# ============================================================

results = df[["id", "proto", "service", "state", "label"]].copy()

results["attack_probability"] = probabilities
results["predicted_label"] = predictions

results["prediction_status"] = results[
    "predicted_label"
].map({
    0: "NORMAL",
    1: "ATTACK"
})

results.to_csv(
    "models/official_test_predictions.csv",
    index=False
)

print("\nPredictions saved to:")
print("models/official_test_predictions.csv")

print("\nASTERIA SAVED MODEL TESTING COMPLETE.")
