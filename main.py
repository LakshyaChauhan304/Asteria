from loader import load_telemetry
from features import build_agent_features
from detections import detect_anomalies
from investigation import (
    generate_investigation_id,
    create_investigation
)


def main():

    

    df = load_telemetry("telemetry.csv")

    print("\n" + "=" * 60)
    print("                     ASTERIA")
    print("              Security Analytics Engine")
    print("=" * 60)

    print("\n[+] Telemetry loaded")

    print(f"    Events      : {len(df)}")
    print(f"    Agents      : {df['agent_id'].nunique()}")
    print(f"    Tools       : {df['tool'].nunique()}")

    

    agent_features = build_agent_features(df)

    print("\n[+] Behavioral features generated")

    

    detections = detect_anomalies(agent_features)

    print("[+] Detection engine completed")

    

    investigations = []

    investigation_number = 1

    for _, detection in detections.iterrows():

        if not detection["flagged"]:
            continue

        agent_id = detection["agent_id"]

        
        agent_events = df[
            df["agent_id"] == agent_id
        ].copy()

        investigation = create_investigation(
            agent_id=agent_id,
            detection_score=detection["detection_score"],
            severity=detection["severity"],
            reasons=detection["reasons"],
            events=agent_events
        )

        investigation["investigation_id"] = (
            generate_investigation_id(
                investigation_number
            )
        )

        investigations.append(investigation)

        investigation_number += 1

  

    print("\n" + "=" * 60)
    print("              ACTIVE INVESTIGATIONS")
    print("=" * 60)

    if not investigations:

        print("\nNo investigations created.")

    else:

        for investigation in investigations:

            print(
                f"\n[{investigation['investigation_id']}]"
            )

            print(
                "Agent:",
                investigation["agent_id"]
            )

            print(
                "State:",
                investigation["state"]
            )

            print(
                "Severity:",
                investigation["severity"]
            )

            print(
                "Detection Score:",
                investigation["detection_score"]
            )

            print(
                "Events:",
                investigation["event_count"]
            )

            print("\nEvidence:")

            for reason in investigation["reasons"]:
                print("  -", reason)

            print("\nTimeline:")

            for event in investigation["events"]:

                print(
                    f"  {event['timestamp']} | "
                    f"{event['tool']} | "
                    f"{event['action']} | "
                    f"{event['resource']} | "
                    f"{event['result']}"
                )


if __name__ == "__main__":
    main()