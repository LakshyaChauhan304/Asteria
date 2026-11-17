
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

from preprocessing import (
    load_dataset,
    prepare_features,
    split_data,
    create_preprocessor
)


def main():

    print("=" * 60)
    print("ASTERIA - XGBOOST ATTACK CATEGORY ANALYSIS")
    print("=" * 60)

    # 1. Load dataset
    df = load_dataset()

    # 2. Prepare features
    X, y = prepare_features(df)

    # 3. Split dataset
    X_train, X_val, y_train, y_val = split_data(X, y)

    # 4. Preprocess
    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # 5. Train XGBoost
    print("\n[+] Training XGBoost...")

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

    # 6. Predict validation data
    predictions = model.predict(X_val_processed)

    # 7. Attach actual attack categories
    results = pd.DataFrame({
        "attack_cat": df.loc[X_val.index, "attack_cat"],
        "actual_label": y_val,
        "predicted_label": predictions
    })

    # 8. Analyse each attack category
    attack_results = results[
        (results["actual_label"] == 1) &
        (results["attack_cat"] != "Normal")
    ]

    summary = []

    for category, group in attack_results.groupby("attack_cat"):

        total = len(group)

        detected = int(
            (group["predicted_label"] == 1).sum()
        )

        missed = total - detected

        recall = detected / total if total else 0

        summary.append({
            "Attack Category": category,
            "Total Attacks": total,
            "Detected": detected,
            "Missed": missed,
            "Recall": recall
        })

    summary_df = pd.DataFrame(summary)

    summary_df = summary_df.sort_values(
        by="Recall",
        ascending=False
    )

    # 9. Display results
    print("\n" + "=" * 70)
    print("XGBOOST ATTACK CATEGORY DETECTION RESULTS")
    print("=" * 70)

    print(summary_df.to_string(index=False))

    # 10. Save results
    Path("experiments").mkdir(exist_ok=True)

    summary_df.to_csv(
        "experiments/xgboost_attack_category_analysis.csv",
        index=False
    )

    print("\n[+] Results saved.")
    print(
        "File: experiments/xgboost_attack_category_analysis.csv"
    )


if __name__ == "__main__":
    main()
