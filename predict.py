import joblib
import pandas as pd


MODEL_PATH = "models/logistic_regression.pkl"


def load_model():

    print("[+] Loading Asteria detection model...")

    model = joblib.load(
        MODEL_PATH
    )

    print("[+] Model loaded.")

    return model


def predict_attack(model, data):

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0]

    normal_probability = probability[0]
    attack_probability = probability[1]

    if prediction == 1:
        result = "ATTACK"
    else:
        result = "NORMAL"

    return {
        "prediction": result,
        "normal_probability": normal_probability,
        "attack_probability": attack_probability
    }


def main():

    model = load_model()


    sample = pd.DataFrame([
        {
            "dur": 0.5,
            "proto": "tcp",
            "service": "http",
            "state": "FIN",
            "spkts": 10,
            "dpkts": 8,
            "sbytes": 500,
            "dbytes": 400,
            "rate": 20,
            "sttl": 64,
            "dttl": 64,
            "sload": 1000,
            "dload": 800,
            "sloss": 0,
            "dloss": 0,
            "sinpkt": 0.1,
            "dinpkt": 0.1,
            "sjit": 0,
            "djit": 0,
            "swin": 255,
            "stcpb": 0,
            "dtcpb": 0,
            "dwin": 255,
            "tcprtt": 0.1,
            "synack": 0.05,
            "ackdat": 0.05,
            "smean": 50,
            "dmean": 50,
            "trans_depth": 0,
            "response_body_len": 0,
            "ct_srv_src": 1,
            "ct_state_ttl": 1,
            "ct_dst_ltm": 1,
            "ct_src_dport_ltm": 1,
            "ct_dst_sport_ltm": 1,
            "ct_dst_src_ltm": 1,
            "is_ftp_login": 0,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 1,
            "ct_src_ltm": 1,
            "ct_srv_dst": 1,
            "is_sm_ips_ports": 0
        }
    ])

    result = predict_attack(
        model,
        sample
    )

    print("\n" + "=" * 60)
    print("ASTERIA NETWORK DETECTION")
    print("=" * 60)

    print(
        f"\nPrediction: "
        f"{result['prediction']}"
    )

    print(
        f"Normal probability: "
        f"{result['normal_probability']:.4f}"
    )

    print(
        f"Attack probability: "
        f"{result['attack_probability']:.4f}"
    )


if __name__ == "__main__":
    main()