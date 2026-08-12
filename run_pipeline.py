"""
run_pipeline.py
------------------
One command that runs the entire SentinelIQ pipeline end to end:

  1. Generate synthetic network traffic (benign + 5 attack archetypes)
  2. Extract per-IP behavioral features over 60s windows
  3. Run adaptive EWMA per-IP baselining
  4. Train Isolation Forest + Random Forest
  5. Run the hybrid fusion detector over every window
  6. Export a JSON summary the Flask dashboard reads directly

Run with:  python run_pipeline.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd

from traffic_generator import TrafficGenerator
from feature_engineering import extract_window_features
from ewma_baseline import run_over_dataset
import train_models
from hybrid_detector import HybridDetector

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("[1/5] Generating synthetic traffic...")
    gen = TrafficGenerator()
    raw = gen.generate(hours=6, benign_per_min=25, n_attacks=16,
                        out_path=f"{DATA_DIR}/raw_flows.csv")
    print(f"      {len(raw):,} flows generated "
          f"({(raw.label=='attack').sum():,} attack flows, {(raw.label=='benign').sum():,} benign)")

    print("[2/5] Extracting per-IP behavioral features (60s windows)...")
    feats = extract_window_features(raw, window="60s")
    feats.to_csv(f"{DATA_DIR}/window_features.csv", index=False)
    print(f"      {len(feats):,} (ip, window) rows")

    print("[3/5] Running adaptive EWMA baselining...")
    feats = run_over_dataset(feats)
    feats.to_csv(f"{DATA_DIR}/window_features_ewma.csv", index=False)

    print("[4/5] Training Isolation Forest + Random Forest...")
    train_models.train(feat_path=f"{DATA_DIR}/window_features_ewma.csv")

    print("[5/5] Scoring every window through the hybrid fusion engine...")
    detector = HybridDetector()
    alerts = detector.score_dataframe(feats).sort_values("risk_score", ascending=False)
    alerts.to_csv(f"{DATA_DIR}/alerts.csv", index=False)

    # ---- export dashboard payload ----
    top_alerts = alerts[alerts["severity"].isin(["CRITICAL", "HIGH", "MEDIUM"])].head(60)
    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "totals": {
            "total_windows": int(len(alerts)),
            "critical": int((alerts.severity == "CRITICAL").sum()),
            "high": int((alerts.severity == "HIGH").sum()),
            "medium": int((alerts.severity == "MEDIUM").sum()),
            "low": int((alerts.severity == "LOW").sum()),
            "info": int((alerts.severity == "INFO").sum()),
            "unique_ips_flagged": int(alerts.loc[alerts.severity.isin(["CRITICAL","HIGH","MEDIUM"]), "src_ip"].nunique()),
        },
        "severity_by_type": (
            alerts[alerts.true_attack_type != "none"]
            .groupby(["true_attack_type", "severity"]).size()
            .unstack(fill_value=0).to_dict(orient="index")
        ),
        "alerts": json.loads(top_alerts.to_json(orient="records", date_format="iso")),
    }
    with open(f"{DATA_DIR}/dashboard_payload.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("\nDone. Dashboard payload written to data/dashboard_payload.json")
    print(f"Alerts: CRITICAL={payload['totals']['critical']} HIGH={payload['totals']['high']} "
          f"MEDIUM={payload['totals']['medium']} across {payload['totals']['unique_ips_flagged']} unique IPs")


if __name__ == "__main__":
    main()
