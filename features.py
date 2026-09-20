import pandas as pd


def build_agent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw security events into behavioral features
    for each agent.
    """

    features = (
        df.groupby("agent_id")
        .agg(
            event_count=("event_id", "count"),

            unique_tools=("tool", "nunique"),

            denied_operations=(
                "result",
                lambda x: (x == "denied").sum()
            ),

            database_queries=(
                "tool",
                lambda x: (x == "database").sum()
            ),

            external_actions=(
                "resource",
                lambda x: (x == "external").sum()
            ),
        )
        .reset_index()
    )

    
    features["denied_ratio"] = (
        features["denied_operations"]
        / features["event_count"]
    )

    
    features["database_ratio"] = (
        features["database_queries"]
        / features["event_count"]
    )

    
    features["external_ratio"] = (
        features["external_actions"]
        / features["event_count"]
    )

    return features