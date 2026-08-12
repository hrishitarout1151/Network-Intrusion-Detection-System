"""
signature_engine.py
---------------------
Deterministic, human-readable rule engine -- the "signature" half of the
hybrid system. These are intentionally simple, auditable thresholds (in the
spirit of Snort/Suricata rules) rather than a black box, because a real SOC
analyst needs to be able to see *exactly* why an alert fired, and because
signatures catch known attack shapes instantly without waiting on a trained
model. Each rule is independent and returns a (fired: bool, weight, reason).
"""

RULES = [
    dict(
        name="PORT_SCAN",
        weight=0.9,
        check=lambda r: r["unique_dst_ports"] >= 15 and r["avg_duration"] < 0.2,
        reason=lambda r: f"{int(r['unique_dst_ports'])} distinct destination ports probed "
                          f"in one window with sub-200ms average flow duration",
    ),
    dict(
        name="SYN_FLOOD",
        weight=1.0,
        check=lambda r: r["syn_ratio"] >= 0.85 and r["flow_rate_per_sec"] >= 4,
        reason=lambda r: f"SYN ratio {r['syn_ratio']:.2f} with {r['flow_rate_per_sec']:.1f} "
                          f"flows/sec sustained against a single target",
    ),
    dict(
        name="BRUTE_FORCE_AUTH",
        weight=0.85,
        check=lambda r: r["admin_port_ratio"] >= 0.8 and r["failed_ratio"] >= 0.6 and r["flow_count"] >= 10,
        reason=lambda r: f"{int(r['flow_count'])} connections to admin ports with "
                          f"{r['failed_ratio']*100:.0f}% failure rate",
    ),
    dict(
        name="DATA_EXFILTRATION",
        weight=0.95,
        check=lambda r: r["total_bytes_sent"] >= 3_000_000 and r["night_ratio"] >= 0.5,
        reason=lambda r: f"{r['total_bytes_sent']/1e6:.1f} MB outbound during off-hours "
                          f"(night_ratio={r['night_ratio']:.2f})",
    ),
    dict(
        name="C2_BEACONING",
        weight=0.8,
        check=lambda r: r["interval_cv"] <= 0.15 and r["flow_count"] >= 5,
        reason=lambda r: f"connection interval coefficient of variation {r['interval_cv']:.3f} "
                          f"(machine-regular cadence) across {int(r['flow_count'])} connections",
    ),
]


def evaluate(row: dict):
    """Runs all signature rules against one (ip, window) feature row.
    Returns (signature_score in [0,1], list of {rule, reason} that fired)."""
    fired = []
    total_weight = 0.0
    for rule in RULES:
        try:
            if rule["check"](row):
                fired.append({"rule": rule["name"], "reason": rule["reason"](row)})
                total_weight = max(total_weight, rule["weight"])  # worst-case rule dominates
        except (KeyError, ZeroDivisionError, TypeError):
            continue
    return min(total_weight, 1.0), fired
