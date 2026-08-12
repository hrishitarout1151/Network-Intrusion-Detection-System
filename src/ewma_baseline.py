"""
ewma_baseline.py
------------------
Adaptive per-IP behavioral baselining using Exponentially Weighted Moving
Average / Variance (EWMA / EWMV).

Why this matters and what it adds beyond the ML models:
A global anomaly detector (Isolation Forest) learns what's "normal" for the
WHOLE network. But a database server that always makes 500 connections/minute
and a laptop that makes 5 are both "normal" for themselves -- a single global
model either misses the laptop's anomaly (drowned out by the server's normal
noise) or falsely flags the server constantly. EWMA baselining solves this by
giving each IP its own continuously-updated mean/variance per feature, so
"anomalous" is judged relative to that host's own recent history, not the
network at large. This is the same technique production tools like Zeek's
anomaly scripts and many commercial UEBA products use, implemented here from
first principles (no external stats library needed beyond numpy).

Update rule (per feature, per IP):
    mean_t  = alpha * x_t + (1 - alpha) * mean_(t-1)
    var_t   = alpha * (x_t - mean_t)^2 + (1 - alpha) * var_(t-1)
    z_t     = (x_t - mean_(t-1)) / sqrt(var_(t-1) + eps)

alpha controls how fast the baseline adapts (higher = forgets history faster).
"""

import numpy as np
import pandas as pd

from feature_engineering import NUMERIC_FEATURES

EPS = 1e-6


class EWMABaselineEngine:
    def __init__(self, alpha=0.2, warmup_windows=3):
        self.alpha = alpha
        self.warmup_windows = warmup_windows
        # per-ip state: {ip: {feature: {"mean": float, "var": float, "n": int}}}
        self.state = {}

    def _get_state(self, ip, feature):
        return self.state.setdefault(ip, {}).setdefault(
            feature, {"mean": 0.0, "var": 1.0, "n": 0}
        )

    def update_and_score(self, ip, feature_values: dict):
        """Given one window's feature dict for an IP, returns z-scores and
        updates that IP's rolling baseline. Returns dict {feature: z_score}."""
        z_scores = {}
        for feat in NUMERIC_FEATURES:
            x = float(feature_values.get(feat, 0.0))
            s = self._get_state(ip, feat)

            if s["n"] < self.warmup_windows:
                # not enough history yet -> low-confidence, neutral score
                z = 0.0
            else:
                z = (x - s["mean"]) / np.sqrt(s["var"] + EPS)

            # update baseline AFTER scoring (score reflects deviation from *prior* baseline)
            prev_mean = s["mean"]
            s["mean"] = self.alpha * x + (1 - self.alpha) * s["mean"] if s["n"] > 0 else x
            s["var"] = (self.alpha * (x - s["mean"]) ** 2 + (1 - self.alpha) * s["var"]
                        if s["n"] > 0 else 1.0)
            s["n"] += 1

            z_scores[feat] = float(np.clip(z, -8, 8))
        return z_scores

    def composite_score(self, z_scores: dict):
        """Aggregate per-feature z-scores into a single 0-1 EWMA anomaly score
        using the mean of absolute z-scores, squashed with a logistic curve so
        it's comparable to the other engines' 0-1 scores."""
        mean_abs_z = float(np.mean([abs(v) for v in z_scores.values()]))
        return 1 / (1 + np.exp(-0.6 * (mean_abs_z - 3)))  # centered around z~3


def run_over_dataset(feat_df: pd.DataFrame, alpha=0.2):
    """Streams window-features chronologically through the engine (simulating
    real-time arrival) and returns the dataframe augmented with an
    'ewma_score' column plus the top deviating feature per row."""
    engine = EWMABaselineEngine(alpha=alpha)
    feat_df = feat_df.sort_values("window_start").reset_index(drop=True)

    scores, top_features = [], []
    for _, row in feat_df.iterrows():
        z = engine.update_and_score(row["src_ip"], row[NUMERIC_FEATURES].to_dict())
        scores.append(engine.composite_score(z))
        top_feat = max(z, key=lambda k: abs(z[k]))
        top_features.append(f"{top_feat} (z={z[top_feat]:.1f})")

    feat_df["ewma_score"] = scores
    feat_df["ewma_top_deviation"] = top_features
    return feat_df


if __name__ == "__main__":
    feats = pd.read_csv("/home/claude/sentineliq/data/window_features.csv")
    scored = run_over_dataset(feats)
    scored.to_csv("/home/claude/sentineliq/data/window_features_ewma.csv", index=False)
    print(scored[["src_ip", "true_attack_type", "ewma_score", "ewma_top_deviation"]].tail(15))
