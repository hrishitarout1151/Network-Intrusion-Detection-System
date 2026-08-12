"""
feature_engineering.py
-----------------------
Converts raw per-flow records into per-(src_ip, time-window) BEHAVIORAL features.

This is the core modeling decision that makes the system "behavioral" rather than
purely signature-based: instead of classifying individual packets/flows, we
profile *what a host was doing* over a rolling window, because attacks are
patterns over time (a scan is many small flows; a beacon is regularly-spaced
flows) that are invisible if you only look at one flow at a time.

Window size and stride are configurable. Each window produces one row per
active source IP with hand-engineered features chosen because each one maps
to a specific attacker archetype in traffic_generator.py:

  unique_dst_ports / port_entropy   -> port scanning
  syn_ratio, failed_ratio, flow_rate-> SYN flood / brute force
  bytes_out, night_ratio            -> exfiltration
  interval_cv (coefficient of var.) -> C2 beaconing (low CV = regular beacon)
"""

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "flow_count", "unique_dst_ips", "unique_dst_ports", "port_entropy",
    "avg_duration", "total_bytes_sent", "total_bytes_recv", "bytes_per_sec",
    "avg_packet_size", "syn_ratio", "failed_ratio", "flow_rate_per_sec",
    "interval_cv", "night_ratio", "admin_port_ratio",
]


def _shannon_entropy(counts):
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def _interval_cv(timestamps):
    """Coefficient of variation of inter-arrival times.
    Near 0 => machine-regular cadence (beaconing). High => bursty/human."""
    if len(timestamps) < 3:
        return 1.0
    ts = np.sort(pd.to_datetime(timestamps).values.astype("int64") / 1e9)
    deltas = np.diff(ts)
    if deltas.mean() == 0:
        return 0.0
    return float(deltas.std() / (deltas.mean() + 1e-9))


def extract_window_features(df: pd.DataFrame, window="60s"):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    rows = []
    for (win_start, ip), group in df.groupby([pd.Grouper(freq=window), "src_ip"]):
        if group.empty:
            continue
        dst_ports = group["dst_port"].values
        port_counts = pd.Series(dst_ports).value_counts().values
        n = len(group)
        duration_s = pd.Timedelta(window).total_seconds()

        row = {
            "window_start": win_start,
            "src_ip": ip,
            "flow_count": n,
            "unique_dst_ips": group["dst_ip"].nunique(),
            "unique_dst_ports": group["dst_port"].nunique(),
            "port_entropy": _shannon_entropy(port_counts),
            "avg_duration": group["duration"].mean(),
            "total_bytes_sent": group["bytes_sent"].sum(),
            "total_bytes_recv": group["bytes_recv"].sum(),
            "bytes_per_sec": group["bytes_sent"].sum() / duration_s,
            "avg_packet_size": (group["bytes_sent"].sum() + 1) / (group["packet_count"].sum() + 1),
            "syn_ratio": group["syn_flag"].mean(),
            "failed_ratio": group["connection_failed"].mean(),
            "flow_rate_per_sec": n / duration_s,
            "interval_cv": _interval_cv(group.index),
            "night_ratio": ((group.index.hour >= 0) & (group.index.hour <= 5)).mean(),
            "admin_port_ratio": group["dst_port"].isin([22, 3389, 21, 23, 3306, 5432]).mean(),
            # ground-truth (kept for training / evaluation, NOT used as a model input)
            "true_label": "attack" if (group["label"] == "attack").any() else "benign",
            "true_attack_type": group.loc[group["label"] == "attack", "attack_type"].mode().iloc[0]
                                 if (group["label"] == "attack").any() else "none",
        }
        rows.append(row)

    feat_df = pd.DataFrame(rows).sort_values(["src_ip", "window_start"]).reset_index(drop=True)
    return feat_df


if __name__ == "__main__":
    raw = pd.read_csv("/home/claude/sentineliq/data/raw_flows.csv")
    feats = extract_window_features(raw, window="60s")
    feats.to_csv("/home/claude/sentineliq/data/window_features.csv", index=False)
    print(f"Built {len(feats):,} (ip, window) feature rows from {len(raw):,} flows")
    print(feats["true_attack_type"].value_counts())
