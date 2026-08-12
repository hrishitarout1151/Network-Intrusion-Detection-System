"""
traffic_generator.py
---------------------
Generates a synthetic but behaviorally-realistic network flow dataset.

Why synthetic instead of downloading a public dataset (NSL-KDD / CICIDS)?
Every "IDS project" on GitHub trains on the exact same public CSVs, which is why
they all look identical and why recruiters can spot a copy-pasted Kaggle notebook
instantly. This generator instead *simulates the underlying behavior* of benign
users and five distinct attacker archetypes at the flow level, so the resulting
dataset -- and every downstream feature/model -- is unique to this project and
reproducible from first principles (a real differentiator to talk about in an
interview: "I modeled attacker behavior, I didn't just fit a classifier to a
famous CSV").

Each row = one network flow (a single TCP/UDP connection or connection attempt),
which is the same abstraction NetFlow/Zeek/Suricata use in production.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
COMMON_PORTS = [80, 443, 53, 25, 110, 143, 993, 995, 8080, 8443]
ADMIN_PORTS = [22, 3389, 21, 23, 3306, 5432]

PROTOCOLS = ["TCP", "UDP", "ICMP"]


class TrafficGenerator:
    def __init__(self, seed=RNG_SEED):
        self.rng = np.random.default_rng(seed)
        self.internal_hosts = [f"10.0.{i // 254}.{(i % 254) + 1}" for i in range(40)]
        self.external_hosts = [
            ".".join(str(self.rng.integers(1, 255)) for _ in range(4)) for _ in range(200)
        ]

    # ------------------------------------------------------------------ #
    # Benign traffic: diurnal pattern, realistic port/protocol mixture
    # ------------------------------------------------------------------ #
    def _benign_flow(self, ts):
        src = self.rng.choice(self.internal_hosts)
        dst = self.rng.choice(self.external_hosts)
        dport = int(self.rng.choice(COMMON_PORTS))
        proto = "TCP" if dport != 53 else "UDP"
        duration = float(np.clip(self.rng.exponential(4.0), 0.05, 120))
        pkt_count = int(np.clip(self.rng.poisson(15), 1, 500))
        bytes_sent = int(np.clip(self.rng.normal(4000, 2500), 40, 200000))
        bytes_recv = int(np.clip(bytes_sent * self.rng.uniform(1.5, 6.0), 40, 900000))
        syn_flag = self.rng.random() < 0.15
        failed = self.rng.random() < 0.02
        return dict(
            timestamp=ts, src_ip=src, dst_ip=dst, src_port=int(self.rng.integers(1024, 65535)),
            dst_port=dport, protocol=proto, duration=duration, packet_count=pkt_count,
            bytes_sent=bytes_sent, bytes_recv=bytes_recv, syn_flag=syn_flag,
            connection_failed=failed, label="benign", attack_type="none",
        )

    # ------------------------------------------------------------------ #
    # Attack archetypes -- each has a DISTINCT statistical signature so the
    # feature engineering step downstream has something meaningful to learn
    # ------------------------------------------------------------------ #
    def _port_scan_burst(self, ts, attacker):
        """Many destination ports on one host, in rapid succession, near-zero payload."""
        target = self.rng.choice(self.internal_hosts)
        n = int(self.rng.integers(40, 250))
        flows = []
        for i in range(n):
            flows.append(dict(
                timestamp=ts + timedelta(milliseconds=int(i * self.rng.uniform(5, 40))),
                src_ip=attacker, dst_ip=target, src_port=int(self.rng.integers(1024, 65535)),
                dst_port=int(self.rng.integers(1, 65535)), protocol="TCP",
                duration=float(self.rng.uniform(0.001, 0.05)), packet_count=int(self.rng.integers(1, 3)),
                bytes_sent=int(self.rng.integers(40, 60)), bytes_recv=0, syn_flag=True,
                connection_failed=True, label="attack", attack_type="port_scan",
            ))
        return flows

    def _syn_flood(self, ts, attacker):
        """High-rate single-destination SYN storm -- classic volumetric DoS."""
        target = self.rng.choice(self.internal_hosts)
        dport = int(self.rng.choice([80, 443]))
        n = int(self.rng.integers(300, 1200))
        flows = []
        for i in range(n):
            flows.append(dict(
                timestamp=ts + timedelta(milliseconds=int(i * self.rng.uniform(0.5, 4))),
                src_ip=attacker, dst_ip=target, src_port=int(self.rng.integers(1024, 65535)),
                dst_port=dport, protocol="TCP", duration=float(self.rng.uniform(0.001, 0.01)),
                packet_count=1, bytes_sent=int(self.rng.integers(40, 64)), bytes_recv=0,
                syn_flag=True, connection_failed=True, label="attack", attack_type="syn_flood",
            ))
        return flows

    def _brute_force(self, ts, attacker):
        """Repeated auth attempts against SSH/RDP/FTP, mostly failing, steady cadence."""
        target = self.rng.choice(self.internal_hosts)
        dport = int(self.rng.choice(ADMIN_PORTS))
        n = int(self.rng.integers(60, 300))
        flows = []
        for i in range(n):
            fail = self.rng.random() < 0.92
            flows.append(dict(
                timestamp=ts + timedelta(seconds=i * self.rng.uniform(0.3, 1.2)),
                src_ip=attacker, dst_ip=target, src_port=int(self.rng.integers(1024, 65535)),
                dst_port=dport, protocol="TCP", duration=float(self.rng.uniform(0.1, 1.5)),
                packet_count=int(self.rng.integers(3, 10)), bytes_sent=int(self.rng.integers(150, 600)),
                bytes_recv=int(self.rng.integers(60, 300)), syn_flag=False,
                connection_failed=fail, label="attack", attack_type="brute_force",
            ))
        return flows

    def _exfiltration(self, ts, attacker_internal_host):
        """Large outbound transfer to an unusual external host, biased to off-hours."""
        dst = self.rng.choice(self.external_hosts)
        hour = ts.hour
        # bias timestamp toward night if not already
        if not (0 <= hour <= 5):
            ts = ts.replace(hour=int(self.rng.integers(0, 5)))
        n = int(self.rng.integers(3, 10))
        flows = []
        for i in range(n):
            flows.append(dict(
                timestamp=ts + timedelta(seconds=i * self.rng.uniform(20, 90)),
                src_ip=attacker_internal_host, dst_ip=dst, src_port=int(self.rng.integers(1024, 65535)),
                dst_port=int(self.rng.choice([443, 22, 8443])), protocol="TCP",
                duration=float(self.rng.uniform(30, 300)), packet_count=int(self.rng.integers(2000, 20000)),
                bytes_sent=int(self.rng.integers(5_000_000, 80_000_000)),
                bytes_recv=int(self.rng.integers(1000, 5000)), syn_flag=False,
                connection_failed=False, label="attack", attack_type="exfiltration",
            ))
        return flows

    def _c2_beacon(self, ts, infected_host, n_beacons=30):
        """Low-and-slow periodic callbacks to the same C2 host at a near-constant interval
        -- the interval REGULARITY (low jitter) is the tell, not the volume."""
        c2_host = self.rng.choice(self.external_hosts)
        interval = float(self.rng.choice([30, 60, 120, 300]))  # seconds
        flows = []
        cur = ts
        for i in range(n_beacons):
            jitter = self.rng.normal(0, interval * 0.03)  # very low jitter = regular
            cur = cur + timedelta(seconds=interval + jitter)
            flows.append(dict(
                timestamp=cur, src_ip=infected_host, dst_ip=c2_host,
                src_port=int(self.rng.integers(1024, 65535)), dst_port=443, protocol="TCP",
                duration=float(self.rng.uniform(0.2, 1.0)), packet_count=int(self.rng.integers(4, 12)),
                bytes_sent=int(self.rng.integers(200, 800)), bytes_recv=int(self.rng.integers(200, 900)),
                syn_flag=False, connection_failed=False, label="attack", attack_type="c2_beacon",
            ))
        return flows

    # ------------------------------------------------------------------ #
    def generate(self, hours=6, benign_per_min=25, n_attacks=14, out_path=None):
        start = datetime(2026, 8, 10, 0, 0, 0)
        end = start + timedelta(hours=hours)
        rows = []

        # benign background traffic across the whole window, diurnal weighting
        cur = start
        while cur < end:
            hour_weight = 0.3 + 0.7 * np.sin(np.pi * (cur.hour % 24) / 24) ** 2
            count = self.rng.poisson(max(1, benign_per_min * hour_weight))
            for _ in range(count):
                jitter = timedelta(seconds=int(self.rng.integers(0, 60)))
                rows.append(self._benign_flow(cur + jitter))
            cur += timedelta(minutes=1)

        # scatter attack campaigns randomly through the window
        attack_pool = ["port_scan", "syn_flood", "brute_force", "exfiltration", "c2_beacon"]
        external_attackers = [f"185.220.{self.rng.integers(1,255)}.{self.rng.integers(1,255)}" for _ in range(30)]
        for _ in range(n_attacks):
            kind = self.rng.choice(attack_pool)
            t0 = start + timedelta(seconds=int(self.rng.integers(0, int((end - start).total_seconds()))))
            attacker = self.rng.choice(external_attackers)
            if kind == "port_scan":
                rows += self._port_scan_burst(t0, attacker)
            elif kind == "syn_flood":
                rows += self._syn_flood(t0, attacker)
            elif kind == "brute_force":
                rows += self._brute_force(t0, attacker)
            elif kind == "exfiltration":
                rows += self._exfiltration(t0, self.rng.choice(self.internal_hosts))
            elif kind == "c2_beacon":
                rows += self._c2_beacon(t0, self.rng.choice(self.internal_hosts))

        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        if out_path:
            df.to_csv(out_path, index=False)
        return df


if __name__ == "__main__":
    gen = TrafficGenerator()
    df = gen.generate(hours=6, benign_per_min=25, n_attacks=14,
                       out_path="/home/claude/sentineliq/data/raw_flows.csv")
    print(f"Generated {len(df):,} flows | attacks: {(df.label=='attack').sum():,} "
          f"| benign: {(df.label=='benign').sum():,}")
    print(df.attack_type.value_counts())
