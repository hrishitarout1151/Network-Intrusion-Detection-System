"""
hybrid_detector.py
---------------------
The fusion layer. This is the actual "AI-powered hybrid behavioral-signature
engine" the project is named after -- everything upstream (signatures, EWMA,
Isolation Forest, Random Forest) produces one opinion each; this module
combines them into a single explainable risk score per (ip, window), rather
than just picking whichever model says "attack" the loudest.

Composite risk formula (weights chosen deliberately, not equal-split):
    risk = 0.30 * signature_score        (deterministic, high precision when it fires)
         + 0.20 * isolation_forest_score (unsupervised outlier signal)
         + 0.20 * ewma_score             (host-relative behavioral drift)
         + 0.30 * rf_attack_probability  (supervised pattern match)

Signatures and the supervised model are weighted slightly higher because they
are precision-oriented (few false positives when they fire); the unsupervised
signals (Isolation Forest, EWMA) are weighted lower individually because they
are recall-oriented / noisier, but they matter most for catching things the
other two have never seen before -- which is the entire point of running four
engines instead of one.

Severity bands turn the continuous score into an actionable SOC label.
"""

import numpy as np
import pandas as pd
import joblib

from feature_engineering import NUMERIC_FEATURES
import signature_engine

MODEL_DIR = "/home/claude/sentineliq/models"

WEIGHTS = dict(signature=0.30, isolation=0.20, ewma=0.20, rf=0.30)

SEVERITY_BANDS = [
    (0.75, "CRITICAL"),
    (0.55, "HIGH"),
    (0.35, "MEDIUM"),
    (0.15, "LOW"),
    (0.0, "INFO"),
]


def severity_for(score):
    for threshold, label in SEVERITY_BANDS:
        if score >= threshold:
            return label
    return "INFO"


class HybridDetector:
    def __init__(self, model_dir=MODEL_DIR):
        self.scaler = joblib.load(f"{model_dir}/scaler.joblib")
        self.iso = joblib.load(f"{model_dir}/isolation_forest.joblib")
        self.rf = joblib.load(f"{model_dir}/random_forest.joblib")
        self.iso_range = joblib.load(f"{model_dir}/iso_score_range.joblib")

    def _isolation_score(self, x_scaled_row):
        raw = -self.iso.decision_function(x_scaled_row.reshape(1, -1))[0]
        lo, hi = self.iso_range["iso_min"], self.iso_range["iso_max"]
        return float(np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1))

    def _rf_attack_probability(self, x_scaled_row):
        probs = self.rf.predict_proba(x_scaled_row.reshape(1, -1))[0]
        classes = list(self.rf.classes_)
        none_idx = classes.index("none") if "none" in classes else None
        p_benign = probs[none_idx] if none_idx is not None else 0.0
        top_idx = int(np.argmax(probs))
        return float(1 - p_benign), classes[top_idx], float(probs[top_idx])

    def score_row(self, row: dict):
        """row must already contain NUMERIC_FEATURES + precomputed 'ewma_score'."""
        x_df = pd.DataFrame([[row.get(f, 0.0) for f in NUMERIC_FEATURES]], columns=NUMERIC_FEATURES)
        x_scaled = self.scaler.transform(x_df)[0]

        sig_score, sig_reasons = signature_engine.evaluate(row)
        iso_score = self._isolation_score(x_scaled)
        ewma_score = float(row.get("ewma_score", 0.0))
        rf_attack_prob, rf_top_class, rf_top_conf = self._rf_attack_probability(x_scaled)

        composite = (
            WEIGHTS["signature"] * sig_score
            + WEIGHTS["isolation"] * iso_score
            + WEIGHTS["ewma"] * ewma_score
            + WEIGHTS["rf"] * rf_attack_prob
        )
        composite = float(np.clip(composite, 0, 1))

        return {
            "src_ip": row.get("src_ip"),
            "window_start": row.get("window_start"),
            "risk_score": round(composite, 4),
            "severity": severity_for(composite),
            "signature_score": round(sig_score, 3),
            "signature_hits": sig_reasons,
            "isolation_score": round(iso_score, 3),
            "ewma_score": round(ewma_score, 3),
            "ewma_top_deviation": row.get("ewma_top_deviation", ""),
            "rf_predicted_type": rf_top_class,
            "rf_confidence": round(rf_top_conf, 3),
            "true_attack_type": row.get("true_attack_type", "unknown"),
        }

    def score_dataframe(self, feat_df: pd.DataFrame):
        return pd.DataFrame([self.score_row(r) for r in feat_df.to_dict("records")])


if __name__ == "__main__":
    feats = pd.read_csv("/home/claude/sentineliq/data/window_features_ewma.csv")
    detector = HybridDetector()
    alerts = detector.score_dataframe(feats)
    alerts = alerts.sort_values("risk_score", ascending=False)
    alerts.to_csv("/home/claude/sentineliq/data/alerts.csv", index=False)

    print(alerts[["src_ip", "risk_score", "severity", "rf_predicted_type", "true_attack_type"]].head(20))
    print("\nSeverity distribution:")
    print(alerts["severity"].value_counts())

    real_attacks = alerts[alerts["true_attack_type"] != "none"]
    caught = (real_attacks["severity"].isin(["CRITICAL", "HIGH", "MEDIUM"])).mean()
    print(f"\nRecall @ MEDIUM-or-above severity on true attack windows: {caught*100:.1f}%")
