"""
train_models.py
------------------
Trains the two ML components of the hybrid engine:

1. Isolation Forest (unsupervised) -- learns the shape of "normal" behavior
   across ALL hosts and flags statistical outliers. Catches novel attacks
   with no signature and no label, at the cost of more false positives.

2. Random Forest (supervised) -- trained on the labeled synthetic data to
   recognize the specific fingerprint of each known attack archetype and
   output calibrated class probabilities. Sharper on known attack types,
   but blind to genuinely new ones -- which is exactly why it's fused with
   the unsupervised + rule-based + EWMA signals rather than used alone.

Both are saved to /models so hybrid_detector.py can load them without
retraining, mirroring how a real detection pipeline separates the
(expensive, periodic) training job from the (cheap, continuous) scoring job.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from feature_engineering import NUMERIC_FEATURES

MODEL_DIR = "/home/claude/sentineliq/models"


def train(feat_path="/home/claude/sentineliq/data/window_features_ewma.csv"):
    df = pd.read_csv(feat_path)
    X = df[NUMERIC_FEATURES].fillna(0.0)
    y_binary = (df["true_label"] == "attack").astype(int)
    y_multiclass = df["true_attack_type"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- Isolation Forest: train ONLY on rows that look benign, so it
    # learns a clean boundary of normal behavior (contamination is a
    # reasonable prior estimate, not ground truth, mirroring production
    # where you rarely have perfectly labeled data going into this model). ----
    iso = IsolationForest(
        n_estimators=200, contamination=0.12, random_state=42, n_jobs=-1
    )
    iso.fit(X_scaled)
    # decision_function: higher = more normal. Flip & min-max scale to 0-1 "anomaly score".
    raw_scores = -iso.decision_function(X_scaled)
    iso_score_min, iso_score_max = raw_scores.min(), raw_scores.max()

    # ---- Random Forest: supervised multiclass on attack_type ----
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_multiclass, test_size=0.25, random_state=42, stratify=y_multiclass
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    print("=== Random Forest classification report (held-out test set) ===")
    print(classification_report(y_test, preds, zero_division=0))

    try:
        probs = rf.predict_proba(X_test)
        auc = roc_auc_score(
            (y_test != "none").astype(int),
            1 - probs[:, list(rf.classes_).index("none")],
        )
        print(f"Attack-vs-benign ROC-AUC: {auc:.4f}")
    except Exception as e:
        print("AUC skipped:", e)

    feat_importance = pd.Series(rf.feature_importances_, index=NUMERIC_FEATURES).sort_values(ascending=False)
    print("\nTop features driving Random Forest decisions:")
    print(feat_importance.head(8).to_string())

    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
    joblib.dump(iso, f"{MODEL_DIR}/isolation_forest.joblib")
    joblib.dump(rf, f"{MODEL_DIR}/random_forest.joblib")
    joblib.dump(
        {"iso_min": float(iso_score_min), "iso_max": float(iso_score_max)},
        f"{MODEL_DIR}/iso_score_range.joblib",
    )
    print(f"\nModels saved to {MODEL_DIR}/")
    return scaler, iso, rf


if __name__ == "__main__":
    train()
