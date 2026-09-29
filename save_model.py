import joblib

from preprocessing import (
    load_dataset,
    prepare_features,
    split_data,
    create_preprocessor
)

from model import build_model, train_model


MODEL_PATH = "models/logistic_regression.pkl"


def main():

    print("=" * 60)
    print("ASTERIA - MODEL TRAINING")
    print("=" * 60)

    

    df = load_dataset()



    X, y = prepare_features(df)

  

    X_train, X_val, y_train, y_val = split_data(
        X,
        y
    )


    preprocessor = create_preprocessor(
        X_train
    )

    

    model = build_model(
        preprocessor
    )

    

    model = train_model(
        model,
        X_train,
        y_train
    )



    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\n[+] Model saved successfully.")
    print(f"[+] Location: {MODEL_PATH}")


if __name__ == "__main__":
    main()