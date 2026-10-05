import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from evidence_builder import build_detection_evidence

from preprocessing import (
    load_dataset,
    prepare_features,
    split_data,
    create_preprocessor
)


ISOLATION_CONTAMINATION = 0.30


class AsteriaDetectionEngine:

    def __init__(self):

        self.logistic_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        self.isolation_model = IsolationForest(
            n_estimators=200,
            contamination=ISOLATION_CONTAMINATION,
            random_state=42,
            n_jobs=-1
        )

        self.preprocessor = None

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    def train(
        self,
        X_train,
        y_train
    ):

        print("\n[+] Training detection engine...")

        self.preprocessor = create_preprocessor(
            X_train
        )

        X_train_processed = (
            self.preprocessor.fit_transform(
                X_train
            )
        )

        # Primary supervised detector
        print(
            "[+] Training Logistic Regression..."
        )

        self.logistic_model.fit(
            X_train_processed,
            y_train
        )

        # Secondary anomaly detector
        print(
            "[+] Training Isolation Forest..."
        )

        self.isolation_model.fit(
            X_train_processed
        )

        print(
            "[+] Detection engine training completed."
        )

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------

    def predict(
        self,
        X
    ):

        if self.preprocessor is None:
            raise RuntimeError(
                "Detection engine has not been trained."
            )

        X_processed = (
            self.preprocessor.transform(X)
        )

        # Logistic Regression
        logistic_prediction = (
            self.logistic_model.predict(
                X_processed
            )
        )

        logistic_probability = (
            self.logistic_model.predict_proba(
                X_processed
            )[:, 1]
        )

        # Isolation Forest
        isolation_raw = (
            self.isolation_model.predict(
                X_processed
            )
        )

        isolation_prediction = (
            isolation_raw == -1
        ).astype(int)

        # Isolation Forest anomaly score
        isolation_score = (
            -self.isolation_model.score_samples(
                X_processed
            )
        )

        return {
            "logistic_prediction": logistic_prediction,
            "logistic_probability": logistic_probability,
            "isolation_prediction": isolation_prediction,
            "isolation_score": isolation_score
        }


def main():

    print("=" * 60)
    print("ASTERIA - DETECTION ENGINE")
    print("=" * 60)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    df = load_dataset()

    X, y = prepare_features(df)

    X_train, X_val, y_train, y_val = split_data(
        X,
        y
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    engine = AsteriaDetectionEngine()

    engine.train(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # Test predictions
    # --------------------------------------------------

    print("\n[+] Generating detection signals...")

    results = engine.predict(
        X_val.head(10)
    )



    print("\n" + "=" * 60)
    print("DETECTION SIGNALS")
    print("=" * 60)

    for i in range(10):

       row = X_val.iloc[i]


       evidence = build_detection_evidence(event_id=X_val.index[i],
                                           
                                           row = row,
                                           
                                           
                                           logistic_prediction=(
                                               results["logistic_prediction"][i]
                                           ),
                                           logistic_probability=(results["logistic_probability"][i]),
                                           isolation_prediction=(results["isolation_prediction"][i]),
                                           
                                           isolation_score=(results["isolation_score"][i]
                                                            )
                                            )


    print("\nDetection Evidence:")
    print(
        evidence.to_dict()
    )

if __name__ == "__main__":
    main()