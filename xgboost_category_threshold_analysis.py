
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

from sklearn.metrics import recall_score

from preprocessing import (
    load_dataset,
    prepare_features,
    split_data,
    create_preprocessor
)


def main():

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

    probabilities = model.predict_proba(
        X_val_processed
    )[:, 1]

    thresholds = [0.50, 0.45, 0.40, 0.35]

    validation_categories = df.loc[
        X_val.index, "attack_cat"
    ]

    results = []

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        for category in sorted(validation_categories.unique()):

            mask = (
                (validation_categories == category) &
                (y_val == 1)
            ).to_numpy()

            if mask.sum() == 0:
                continue

            category_recall = recall_score(
                y_val.to_numpy()[mask],
                predictions[mask],
                zero_division=0
            )

            results.append({
                "Threshold": threshold,
                "Attack_Category": category,
                "Total": int(mask.sum()),
                "Detected": int(predictions[mask].sum()),
                "Missed": int(mask.sum() - predictions[mask].sum()),
                "Recall": category_recall
            })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 75)
    print("ATTACK CATEGORY RECALL BY THRESHOLD")
    print("=" * 75)

    print(results_df.to_string(index=False))

    Path("experiments").mkdir(exist_ok=True)

    results_df.to_csv(
        "experiments/xgboost_category_threshold_analysis.csv",
        index=False
    )

    print("\n[+] Results saved.")


if __name__ == "__main__":
    main()
