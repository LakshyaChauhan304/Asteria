
import pandas as pd

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
    create_preprocessor,
    split_data
)

from feature_engineering import engineer_features


# 1. Load dataset
df = load_dataset()

X, y = prepare_features(df)

# 2. Apply feature engineering
X = engineer_features(X)

# 3. Split once - same split for every experiment
X_train, X_val, y_train, y_val = split_data(X, y)

# 4. Preserve original validation metadata
validation_metadata = df.loc[X_val.index].copy()

# 5. Define experiments
experiments = {
    "Full engineered model": [],
    "Without TTL features": ["sttl", "dttl"],
    "Without state TTL": ["ct_state_ttl"],
    "Without all TTL features": [
        "sttl", "dttl", "ct_state_ttl"
    ]
}

results = []

# 6. Train and evaluate each experiment
for experiment_name, removed_features in experiments.items():

    print("\n" + "=" * 75)
    print(experiment_name)
    print("=" * 75)

    train_data = X_train.drop(
        columns=removed_features,
        errors="ignore"
    )

    val_data = X_val.drop(
        columns=removed_features,
        errors="ignore"
    )

    # Create a separate preprocessor for each experiment
    preprocessor = create_preprocessor(train_data)

    train_processed = preprocessor.fit_transform(train_data)
    val_processed = preprocessor.transform(val_data)

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

    model.fit(train_processed, y_train)

    # Evaluate at the same threshold
    probabilities = model.predict_proba(val_processed)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    fpr = fp / (fp + tn)

    # Analyze the two major false-positive groups
    metadata = validation_metadata.copy()
    metadata["actual"] = y_val
    metadata["predicted"] = predictions

    tcp_fin_fp = len(metadata[
        (metadata["proto"] == "tcp") &
        (metadata["state"] == "FIN") &
        (metadata["actual"] == 0) &
        (metadata["predicted"] == 1)
    ])

    udp_int_fp = len(metadata[
        (metadata["proto"] == "udp") &
        (metadata["state"] == "INT") &
        (metadata["actual"] == 0) &
        (metadata["predicted"] == 1)
    ])

    results.append({
        "Experiment": experiment_name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "TCP_FIN_FP": tcp_fin_fp,
        "UDP_INT_FP": udp_int_fp
    })

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"FPR: {fpr:.4f}")
    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")
    print(f"TCP-FIN false positives: {tcp_fin_fp}")
    print(f"UDP-INT false positives: {udp_int_fp}")


# 7. Final comparison
results_df = pd.DataFrame(results)

print("\n" + "=" * 100)
print("FEATURE ABLATION COMPARISON")
print("=" * 100)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "feature_ablation_results.csv",
    index=False
)

print("\nResults saved to feature_ablation_results.csv")
