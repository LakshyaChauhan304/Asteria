from sklearn.metrics import confusion_matrix

from preprocessing import (
    load_dataset,
    prepare_features,
    split_data,
    create_preprocessor
)

from model import build_model, train_model


def main():

    print("=" * 60)
    print("ASTERIA - ERROR ANALYSIS")
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

    

    y_pred = model.predict(X_val)

    

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        y_pred
    ).ravel()

    print("\n" + "=" * 60)
    print("ERROR BREAKDOWN")
    print("=" * 60)

    print(f"\nTrue Negatives  : {tn}")
    print(f"False Positives : {fp}")
    print(f"False Negatives : {fn}")
    print(f"True Positives  : {tp}")

    print("\nInterpretation:")

    print(
        f"\nFalse Positives:"
        f" {fp:,} normal network flows "
        f"were classified as attacks."
    )

    print(
        f"\nFalse Negatives:"
        f" {fn:,} attack network flows "
        f"were classified as normal."
    )


if __name__ == "__main__":
    main()