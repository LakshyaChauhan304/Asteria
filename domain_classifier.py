
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# --------------------------------------------------
# 1. LOAD DATASETS
# --------------------------------------------------

train_df = pd.read_csv(
    "data/UNSW_NB15_training-set.csv"
)

test_df = pd.read_csv(
    "data/UNSW_NB15_testing-set.csv"
)

OUTPUT_DIR = Path("analysis/domain_classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("ASTERIA - DATASET DOMAIN CLASSIFIER")
print("=" * 65)


# --------------------------------------------------
# 2. KEEP ONLY NORMAL TRAFFIC
# --------------------------------------------------

train_normal = train_df[
    train_df["label"] == 0
].copy()

test_normal = test_df[
    test_df["label"] == 0
].copy()

print("Training normal records:", len(train_normal))
print("Testing normal records:", len(test_normal))


# --------------------------------------------------
# 3. CREATE DOMAIN LABELS
# --------------------------------------------------

excluded = ["id", "attack_cat", "label"]

X_train_domain = train_normal.drop(
    columns=excluded
)

X_test_domain = test_normal.drop(
    columns=excluded
)

X_domain = pd.concat(
    [X_train_domain, X_test_domain],
    ignore_index=True
)

# 0 = training dataset
# 1 = testing dataset

y_domain = np.concatenate([
    np.zeros(len(X_train_domain), dtype=int),
    np.ones(len(X_test_domain), dtype=int)
])


# --------------------------------------------------
# 4. IDENTIFY FEATURE TYPES
# --------------------------------------------------

categorical_features = [
    col for col in ["proto", "service", "state"]
    if col in X_domain.columns
]

numerical_features = [
    col for col in X_domain.columns
    if col not in categorical_features
]


# --------------------------------------------------
# 5. BUILD PREPROCESSING PIPELINE
# --------------------------------------------------

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# --------------------------------------------------
# 6. SPLIT DATA
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_domain,
    y_domain,
    test_size=0.20,
    random_state=42,
    stratify=y_domain
)


# --------------------------------------------------
# 7. BUILD DOMAIN CLASSIFIER
# --------------------------------------------------

domain_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])


# --------------------------------------------------
# 8. TRAIN
# --------------------------------------------------

print("\nTraining domain classifier...")

domain_model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 9. EVALUATE
# --------------------------------------------------

predictions = domain_model.predict(X_val)

probabilities = domain_model.predict_proba(X_val)[:, 1]

accuracy = accuracy_score(y_val, predictions)

auc = roc_auc_score(y_val, probabilities)

print("\nDOMAIN CLASSIFIER RESULTS")

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {auc:.4f}")

print("\nClassification Report")

print(
    classification_report(
        y_val,
        predictions,
        target_names=[
            "Training domain",
            "Testing domain"
        ]
    )
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_val,
        predictions
    )
)


# --------------------------------------------------
# 10. FEATURE IMPORTANCE
# --------------------------------------------------

fitted_preprocessor = domain_model.named_steps["preprocessor"]

feature_names = fitted_preprocessor.get_feature_names_out()

importances = domain_model.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

importance_df.to_csv(
    OUTPUT_DIR / "domain_feature_importance.csv",
    index=False
)

print("\nTOP 20 DOMAIN-DISTINGUISHING FEATURES")

print(
    importance_df.head(20).to_string(index=False)
)


# --------------------------------------------------
# 11. SAVE RESULTS
# --------------------------------------------------

joblib.dump(
    domain_model,
    OUTPUT_DIR / "domain_classifier.pkl"
)

print("\nDomain classifier saved.")

print("\n" + "=" * 65)
print("DOMAIN ANALYSIS COMPLETED")
print("=" * 65)
