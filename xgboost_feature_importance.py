
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
    print("ASTERIA - XGBOOST FEATURE IMPORTANCE")
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

    print("\n[+] Preprocessing completed.")

    # 5. Train XGBoost
    print("\n[+] Training XGBoost...")

    xgb_model = XGBClassifier(
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

    xgb_model.fit(X_train_processed, y_train)

    print("[+] Training completed.")

    # 6. Extract feature names
    feature_names = preprocessor.get_feature_names_out()

    # 7. Extract feature importance
    importance_values = xgb_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })

    # 8. Sort by importance
    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # 9. Display top 20 features
    print("\n" + "=" * 60)
    print("TOP 20 IMPORTANT FEATURES")
    print("=" * 60)

    print(
        importance_df.head(20).to_string(index=False)
    )

    # 10. Save results
    Path("experiments").mkdir(exist_ok=True)

    importance_df.to_csv(
        "experiments/xgboost_feature_importance.csv",
        index=False
    )

    print("\n[+] Feature importance saved.")
    print("File: experiments/xgboost_feature_importance.csv")


if __name__ == "__main__":
    main()
