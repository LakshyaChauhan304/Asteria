import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


from preprocessing import (
    load_dataset,
    prepare_features,
    split_data,
    create_preprocessor
)



def build_model(preprocessor):
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    return model



def train_model(model,X_train,y_train):

    print("\n [+] Training Logistic Regression...")

    model.fit(
        X_train,
        y_train
    )

    return model


def main():

    print("=" * 60)
    print("ASTERIA - BASELINE DETECTION MODEL")
    print("=" * 60)



    df = load_dataset()

    X,y = prepare_features(df)


    X_train,X_val,y_train,y_val = split_data(X,y)


    preprocessor = create_preprocessor(
        X_train
    )

    model = build_model(preprocessor)

    model = train_model(model,X_train,y_train)

    print("\n[+] Baseline model is ready.")

    print("\nModel:")
    print(model)

if __name__ == "__main__":
    main()