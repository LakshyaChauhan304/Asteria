import pandas as pd 


from sklearn.model_selection import train_test_split 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer



DATA_PATH = "data/UNSW_NB15_training-set.csv"



def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)


    print("=" * 60)
    print("ASTERIA - DATA PREPROCESSING")
    print("=" * 60)



    print(f"\nDataset shape: {df.shape}")

    return df



def prepare_features(df):


    y = df["label"]


    X = df.drop(columns=["id","attack_cat","label"])


    print("\n Target distribution:")
    print(y.value_counts())


    print("\nFeature shape before encoding.")
    print(X.shape)


    return X,y


def create_preprocessor(X):

    categorical_features = [
        "proto",
        "service",
        "state"
    ]

    numerical_features = [
        column 
        for column in X.columns
        if column not in categorical_features
    ]

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer",
             SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer",
             SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            )
        ]
    )

    return preprocessor


def split_data(X,y):

    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


    print("\nData split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")


    return X_train,X_val,y_train,y_val



def main():

    df = load_dataset()

    X,y = prepare_features(df)

    X_train,X_val,y_train,y_val = split_data(X,y)

    preprocessor = create_preprocessor(X_train)


    X_train_processed = preprocessor.fit_transform(X_train)


    X_val_processed = preprocessor.transform(X_val)


    print("\nProcessed feature shapes:")
    print   (
        "Training:",
        X_train_processed.shape
    )

    print(
        "Validation:",
        X_val_processed.shape
    )


    print("\nPreprocessing completed successfully.")

if __name__ == "__main__":
    main()

