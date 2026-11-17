
import os
import json
import joblib
import pandas as pd

from datetime import datetime, timezone
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import load_dataset, prepare_features, create_preprocessor


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_asteria_pipeline.pkl"
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "model_metadata.json"
)

THRESHOLD = 0.50  # Provisional operating threshold


# ============================================================
# 2. LOAD TRAINING DATASET
# ============================================================

print("\n" + "=" * 70)
print("ASTERIA - FINAL XGBOOST MODEL TRAINING")
print("=" * 70)

df = load_dataset()

X, y = prepare_features(df)

print("\nTraining feature shape:", X.shape)
print("Training target shape:", y.shape)


# ============================================================
# 3. CREATE PREPROCESSING PIPELINE
# ============================================================

preprocessor = create_preprocessor(X)

print("\nPreprocessing configuration created.")

print("Numerical and categorical transformations ready.")


# ============================================================
# 4. INITIALIZE XGBOOST
# ============================================================

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


# ============================================================
# 5. CREATE COMPLETE MODEL PIPELINE
# ============================================================

final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])


# ============================================================
# 6. TRAIN ON COMPLETE TRAINING DATASET
# ============================================================

print("\nTraining final XGBoost model...")
print("This may take some time.")

final_pipeline.fit(X, y)

print("\nModel training completed successfully.")


# ============================================================
# 7. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 8. SAVE COMPLETE PIPELINE
# ============================================================

joblib.dump(final_pipeline, MODEL_PATH)

print("\nComplete model pipeline saved:")
print(MODEL_PATH)


# ============================================================
# 9. SAVE MODEL METADATA
# ============================================================

metadata = {
    "model_name": "Asteria XGBoost Network Intrusion Detector",
    "model_type": "XGBClassifier",
    "dataset": "UNSW-NB15 Training Dataset",
    "training_samples": int(len(X)),
    "input_features": list(X.columns),
    "number_of_input_features": int(X.shape[1]),
    "threshold": THRESHOLD,
    "threshold_status": "Provisional",
    "random_state": 42,
    "parameters": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1
    },
    "training_date_utc": datetime.now(
        timezone.utc
    ).isoformat()
}

with open(METADATA_PATH, "w") as file:
    json.dump(metadata, file, indent=4)

print("\nModel metadata saved:")
print(METADATA_PATH)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ASTERIA MODEL TRAINING SUMMARY")
print("=" * 70)

print("Model: XGBoost")
print("Training records:", len(X))
print("Input features:", X.shape[1])
print("Classification threshold:", THRESHOLD)
print("Model file:", MODEL_PATH)
print("Metadata file:", METADATA_PATH)

print("\nASTERIA FINAL MODEL READY.")
