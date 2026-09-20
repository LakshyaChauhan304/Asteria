from pathlib import Path 
import pandas as pd 



REQUIRED_COLUMNS = [
    "event_id",
    "timestamp",
    "agent_id",
    "tool",
    "action",
    "resource",
    "result",

]


def load_telemetry(path: str) -> pd.DataFrame:

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Telemetry file not found: {path}"
        )

    df = pd.read_csv(file_path)


    missing_columns = [
        column 
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


    df["timestamp"] = pd.to_datetime(df["timestamp"],errors = "coerce")

    if df["timestamp"].isna().any():
        raise ValueError("Invalid timestamp found.")

    df = df.drop_duplicates(
        subset=["event_id"]
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df

