from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocessing import (
    load_dataset,
    prepare_features,
    split_data,
    create_preprocessor
)

from model import build_model, train_model


def evaluate_model(model, X_val, y_val):

    print("\n[+] Generating predictions...")

    y_pred = model.predict(X_val)



    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred
    )

    recall = recall_score(
        y_val,
        y_pred
    )

    f1 = f1_score(
        y_val,
        y_pred
    )



    tn, fp, fn, tp = confusion_matrix(
        y_val,
        y_pred
    ).ravel()

 

    false_positive_rate = fp / (fp + tn)


    print("\n" + "=" * 60)
    print("ASTERIA - MODEL EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy           : {accuracy:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1 Score           : {f1:.4f}")
    print(
        f"False Positive Rate: "
        f"{false_positive_rate:.4f}"
    )

    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)

    print(f"\nTrue Negatives  : {tn}")
    print(f"False Positives : {fp}")
    print(f"False Negatives : {fn}")
    print(f"True Positives  : {tp}")

    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)

    print(
        classification_report(
            y_val,
            y_pred,
            target_names=[
                "Normal",
                "Attack"
            ]
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp
    }


def main():

    print("=" * 60)
    print("ASTERIA - DETECTION MODEL EVALUATION")
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


    evaluate_model(
        model,
        X_val,
        y_val
    )


if __name__ == "__main__":
    main()