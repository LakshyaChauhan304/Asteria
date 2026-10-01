import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)



from preprocessing import (
    load_dataset,
    prepare_features,
    split_data,
    create_preprocessor
)


CONTAMINATION_VALUES = [
    "auto",
    0.01,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]


def evaluate_predictions(y_true,y_pred):

    precision = precision_score(y_true,y_pred,zero_division=0)

    recall = recall_score(y_true,y_pred,zero_division=0)

    f1 = f1_score(y_true,y_pred,zero_division=0)

    tn,fp,fn,tp = confusion_matrix(y_true,y_pred).ravel()

    false_positive_rate = (
        fp / (fp + tn)
    )

    return {
        "Precision": precision,
        "Recall":recall,
        "F1": f1,
        "False_Positive_Rate": false_positive_rate,
        "TN":tn,
        "FP":fp, 
        "FN":fn,
        "TP":tp
    }


def run_experiment(
        X_train_processed,
        X_val_processed,
        y_val,
        contamination
):

    print(f"\n[+] Testing contamination = "
          f"{contamination}"
        )


    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_processed)

    predictions = model.predict(X_val_processed)


    y_pred = (
        predictions == -1
    ).astype(int)

    metrics = evaluate_predictions(y_val,y_pred)


    metrics["contamination"] = contamination

    metrics["predicted_anomalies"] = (
        y_pred == 1
    ).sum()

    return metrics


def main():

    print("=" * 60)
    print("ASTERIA - ISOLATION FOREST EXPERIMENT")
    print("=" * 60)

    df = load_dataset()

    X,y = prepare_features(df)

    X_train,X_val,y_train,y_val = split_data(X,y)


    preprocessor = create_preprocessor(X_train)

    print("\n[+] Preprocessing training data...")

    X_train_processed = preprocessor.fit_transform(X_train)

    X_val_processed = preprocessor.transform(X_val)


    print("[+] Preprocessing completed.")


    results = []

    for contamination in CONTAMINATION_VALUES:

        metrics = run_experiment(
            X_train_processed,
            X_val_processed,
            y_val,
            contamination
        )

        results.append(metrics)



    results_df = pd.DataFrame(results)


    results_df = results_df[
        [
            "contamination",
            "predicted_anomalies",
            "Precision",
            "Recall",
            "F1",
            "False_Positive_Rate",
            "TN",
            "FP",
            "FN",
            "TP"
        ]
    ]

    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    print(results_df.to_string(index=False))




    results_df.to_csv("experiments/Isolation_forest_experiments.csv",index=False)

    print("\n[+] Results saved to "
              "experiments/Isolation_forest_experiments.csv"
              )



if __name__ == "__main__":
            main()
