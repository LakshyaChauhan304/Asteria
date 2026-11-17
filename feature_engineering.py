
import pandas as pd


def engineer_features(df):

    df = df.copy()

    # 1. Protocol and connection-state interaction
    df["proto_state"] = (
        df["proto"].astype(str) +
        "_" +
        df["state"].astype(str)
    )

    # 2. Total traffic volume
    df["total_bytes"] = (
        df["sbytes"] + df["dbytes"]
    )

    # 3. Total packet count
    df["total_packets"] = (
        df["spkts"] + df["dpkts"]
    )

    # 4. Combined traffic load
    df["total_load"] = (
        df["sload"] + df["dload"]
    )

    # 5. Source/destination byte ratio
    df["src_dst_byte_ratio"] = (
        df["sbytes"] /
        (df["dbytes"] + 1)
    )

    # 6. Source/destination packet ratio
    df["src_dst_packet_ratio"] = (
        df["spkts"] /
        (df["dpkts"] + 1)
    )

    return df
