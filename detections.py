import pandas as pd


def calculate_detection_score(row: pd.Series) -> float:
    """
    Calculate an interpretable behavioral risk indicator.

    This is NOT a probability of malicious activity.
    It is a heuristic score based on observable telemetry features.
    """

    score = 0

    

    if row["denied_operations"] >= 1:
        score += 20

    if row["denied_operations"] >= 3:
        score += 15

    

    if row["database_ratio"] >= 0.25:
        score += 15

    if row["database_ratio"] >= 0.50:
        score += 10

    

    if row["unique_tools"] >= 3:
        score += 10

    if row["unique_tools"] >= 4:
        score += 5

   

    if row["external_actions"] >= 1:
        score += 15

    

    if row["denied_ratio"] >= 0.25:
        score += 10

    
    

    return min(score, 100)


def generate_reasons(row: pd.Series) -> list:
    """
    Generate evidence-backed explanations
    for the detection score.
    """

    reasons = []

    if row["denied_operations"] >= 1:
        reasons.append(
            f"{int(row['denied_operations'])} denied operations detected"
        )

    if row["denied_operations"] >= 3:
        reasons.append(
            "Repeated denied operations observed"
        )

    if row["database_ratio"] >= 0.25:
        reasons.append(
            f"{row['database_ratio'] * 100:.1f}% of activity involved database operations"
        )

    if row["unique_tools"] >= 3:
        reasons.append(
            f"High tool diversity: {int(row['unique_tools'])} different tools used"
        )

    if row["external_actions"] >= 1:
        reasons.append(
            "External resource activity detected"
        )

    if row["denied_ratio"] >= 0.25:
        reasons.append(
            f"High denied-operation ratio: {row['denied_ratio'] * 100:.1f}%"
        )

    return reasons


def classify_severity(score: float) -> str:
    """
    Convert the heuristic score into an investigation priority.
    """

    if score >= 70:
        return "HIGH"

    if score >= 40:
        return "MEDIUM"

    return "LOW"


def detect_anomalies(features: pd.DataFrame) -> pd.DataFrame:

    results = features.copy()

    results["detection_score"] = results.apply(
        calculate_detection_score,
        axis=1
    )

    results["severity"] = results["detection_score"].apply(
        classify_severity
    )

    results["reasons"] = results.apply(
        generate_reasons,
        axis=1
    )

    results["flagged"] = (
        results["detection_score"] >= 40
    )

    return results