# NIDS — Network Intrusion Detection System

An AI-powered IDS that fuses **four independent detection engines** — a
deterministic rule/signature engine, an unsupervised Isolation Forest, a
supervised Random Forest, and an adaptive per-host EWMA behavioral baseline —
into a single explainable risk score, served through a live SOC-style
dashboard.

## Why this project is different from the typical "IDS with sklearn" repo

Most student cybersecurity portfolio projects download NSL-KDD or CICIDS2017,
throw a RandomForestClassifier at it, and report 99% accuracy. That's a
tutorial, not a system, and every recruiter has seen the exact same notebook
a hundred times. NIDS is built differently on purpose:

| Typical project | NIDS |
|---|---|
| One public dataset everyone uses | Custom traffic simulator you designed, modeling 5 distinct attacker behaviors from first principles |
| One model (usually RF or a simple NN) | Four engines fused by a weighted, explainable formula |
| Global "normal" for the whole network | Per-IP adaptive EWMA baseline — a host is compared to *its own* history |
| Classifies individual rows | Classifies **behavior over a rolling time window**, which is how scans, floods, and beacons actually look |
| Black-box score | Every alert ships with a human-readable explanation: which rule fired, which feature deviated, and each engine's individual contribution |
| Static Jupyter notebook | A running Flask app + dashboard with a "Run New Simulation" button that re-executes the entire pipeline live |

## Architecture

```
traffic_generator.py   -> synthetic flows (benign + port_scan, syn_flood,
                           brute_force, exfiltration, c2_beacon)
        |
        v
feature_engineering.py -> per-(src_ip, 60s window) behavioral features
                           (port entropy, SYN ratio, byte rate, interval CV, ...)
        |
        v
ewma_baseline.py       -> adaptive per-IP mean/variance -> z-score deviation
        |
        v
train_models.py        -> Isolation Forest (unsupervised) + Random Forest (supervised)
        |
        v
signature_engine.py    -> deterministic Snort-style rules (independent of ML)
        |
        v
hybrid_detector.py     -> FUSION: composite_risk =
                             0.30 * signature_score
                           + 0.20 * isolation_forest_score
                           + 0.20 * ewma_score
                           + 0.30 * random_forest_probability
                           -> severity band (CRITICAL/HIGH/MEDIUM/LOW/INFO)
        |
        v
dashboard/app.py       -> Flask API + SOC-style live dashboard
```

## Why fuse four engines instead of picking the "best" one

Each engine has a distinct failure mode, and the fusion is designed so each
one covers another's blind spot:

- **Signature engine** — zero false positives when it fires, but only catches
  attack shapes someone already thought to write a rule for.
- **Random Forest** — sharp on attack types present in training data, blind
  to genuinely novel attacks.
- **Isolation Forest** — catches novel outliers with no label needed, but
  noisier and has no concept of "normal for this specific host."
- **EWMA baseline** — solves the "normal for this specific host" problem the
  other three ignore: a database server making 500 conn/min and a laptop
  making 5 are each judged against their *own* history, not a global average.

## Running it

```bash
cd IDS
pip install -r requirements.txt

# Run the full pipeline once (generates data, trains models, scores alerts)
python run_pipeline.py

# Launch the dashboard
cd dashboard
python app.py
# -> open http://localhost:5055
```

The dashboard's **"Run New Simulation"** button re-executes the entire
pipeline (fresh synthetic attack campaign, fresh EWMA state, model retrain,
fresh scoring) on demand — it is not replaying a canned demo.

## Project structure

```
IDS/
├── run_pipeline.py            # one-command end-to-end orchestrator
├── requirements.txt
├── src/
│   ├── traffic_generator.py   # synthetic flow-level traffic + 5 attack archetypes
│   ├── feature_engineering.py # per-IP sliding-window behavioral features
│   ├── ewma_baseline.py       # adaptive per-host statistical baselining
│   ├── signature_engine.py    # deterministic rule engine
│   ├── train_models.py        # Isolation Forest + Random Forest training
│   └── hybrid_detector.py     # weighted fusion + severity + explanations
├── data/                      # generated CSVs + dashboard_payload.json
├── models/                    # saved scaler / IsolationForest / RandomForest
└── dashboard/
    ├── app.py                 # Flask backend (serves payload, /api/simulate)
    ├── templates/index.html
    └── static/{css,js}
```

## Talking points for interviews

- **Feature engineering, not just modeling**: port entropy (Shannon entropy
  of destination ports) distinguishes a scan from normal multi-service
  traffic; interval coefficient-of-variation distinguishes machine-regular
  C2 beaconing from human browsing, even at low volume.
- **Adaptive baselining**: implemented EWMA mean/variance tracking from
  scratch (no external stats library) to give every host its own drifting
  baseline instead of one global threshold.
- **Explainability by design**: every alert's fusion panel shows exactly
  which of the four engines contributed what, and why — not just a single
  opaque "attack probability."
- **End-to-end system, not a notebook**: data generation, feature
  engineering, training, and a live Flask dashboard with a working
  "re-run the whole pipeline" action.

## Extending this into a resume-ready production story

- Swap `traffic_generator.py` for a real NetFlow/Zeek log ingester — the
  feature engineering and fusion layers don't need to change, since they
  operate on the same flow schema.
- Persist EWMA state to Redis so baselines survive restarts.
- Add a feedback loop where analyst-confirmed false positives retrain the
  Random Forest (active learning).
