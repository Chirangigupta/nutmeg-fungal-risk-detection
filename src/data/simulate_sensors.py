import argparse
import numpy as np
import pandas as pd

def simulate(days=30, seed=42):
    rng = np.random.default_rng(seed)
    rows = days * 24  # hourly
    # baseline seasonal-ish trends (simplified)
    temp = rng.normal(27, 2.5, rows)      # deg C
    humidity = rng.normal(70, 12, rows)   # %
    leaf_wet = rng.normal(40, 15, rows)   # %
    soil_moist = rng.normal(55, 10, rows) # %

    # introduce risky periods
    for i in range(0, rows, 120):
        humidity[i:i+24] += rng.uniform(10, 20)
        leaf_wet[i:i+24] += rng.uniform(10, 25)

    # label: simple heuristic risk (proxy for fungal conducive conditions)
    risk_score = (
        (humidity > 80).astype(int) * 0.4 +
        ((temp >= 24) & (temp <= 30)).astype(int) * 0.2 +
        (leaf_wet > 65).astype(int) * 0.3 +
        (soil_moist > 60).astype(int) * 0.1
    )
    prob = np.clip(risk_score, 0, 1)
    label = (prob > 0.5).astype(int)  # 1 = high risk

    time = pd.date_range("2025-01-01", periods=rows, freq="H")
    df = pd.DataFrame({
        "timestamp": time,
        "temperature": temp,
        "humidity": humidity,
        "leaf_wetness": leaf_wet,
        "soil_moisture": soil_moist,
        "risk_label": label,
        "risk_prob": prob
    })
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/processed/sensors.csv")
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()

    df = simulate(days=args.days)
    os = __import__("os")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")
