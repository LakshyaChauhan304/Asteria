
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
    print("ASTERIA - FUZZERS ERROR ANALYSIS")
    print("=" * 60)

    # 1. Load dataset
    df = load_dataset()

    # 2. Prepare features
    X, y = prepare_features(df)

    # 3. Split dataset
    X_train, X_val, y_train, y_val = split_data(X, y)

    # 4. Preprocess data
    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # 5. Train XGBoost
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

    # 6. Predict validation data
    predictions = model.predict(X_val_processed)

    # 7. Identify Fuzzers attacks
    fuzzers_mask = (
        (df.loc[X_val.index, "attack_cat"] == "Fuzzers") &
        (y_val == 1)
    )

    fuzzers_indices = X_val.index[fuzzers_mask]

    fuzzers_results = pd.DataFrame({
        "actual": y_val.loc[fuzzers_indices],
        "predicted": predictions[
            X_val.index.get_indexer(fuzzers_indices)
        ]
    }, index=fuzzers_indices)

    # 8. Separate detected and missed attacks
    detected_indices = fuzzers_results[
        fuzzers_results["predicted"] == 1
    ].index

    missed_indices = fuzzers_results[
        fuzzers_results["predicted"] == 0
    ].index

    # 9. Compare numerical feature distributions
    numerical_features = X_train.select_dtypes(
        include="number"
    ).columns

    detected_data = df.loc[
        detected_indices, numerical_features
    ]

    missed_data = df.loc[
        missed_indices, numerical_features
    ]

    comparison = pd.DataFrame({
        "Detected_Mean": detected_data.mean(),
        "Missed_Mean": missed_data.mean(),

        "Detected_Median": detected_data.median(),
        "Missed_Median": missed_data.median(),

        "Detected_Q25": detected_data.quantile(0.25),
        "Missed_Q25": missed_data.quantile(0.25),

        "Detected_Q75": detected_data.quantile(0.75),
        "Missed_Q75": missed_data.quantile(0.75),

        "Detected_Std": detected_data.std(),
        "Missed_Std": missed_data.std()
    })

    comparison["Absolute_Mean_Difference"] = (
        comparison["Detected_Mean"] -
        comparison["Missed_Mean"]
    ).abs()

    comparison = comparison.sort_values(
        by="Absolute_Mean_Difference",
        ascending=False
    )

    # 10. Display results
    print("\n" + "=" * 60)
    print("FUZZERS ERROR ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"Total Fuzzers attacks: {len(fuzzers_results)}")
    print(f"Detected: {len(detected_indices)}")
    print(f"Missed: {len(missed_indices)}")

    print("\nTOP 20 FUZZERS FEATURE DISTRIBUTIONS")

    print(
        comparison.head(20).to_string()
    )

    # 11. Save results
    Path("experiments").mkdir(exist_ok=True)

    comparison.to_csv(
        "experiments/xgboost_fuzzers_feature_analysis.csv"
    )

    print("\n[+] Updated feature analysis saved.")
    print(
        "File: experiments/xgboost_fuzzers_feature_analysis.csv"
    )


if __name__ == "__main__":
    main()
