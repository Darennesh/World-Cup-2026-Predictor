# 2026 FIFA World Cup — Knockout Stage Predictor

A probabilistic, simulation-based forecasting system for the 2026 World Cup.
The model trains a **match-level goal model** on historical international results,
updates each team's strength from **2026 group-stage results**, then runs a
**Monte Carlo simulation** of the 48-team bracket to produce round-by-round
probabilities for every team.

> Primary metric: **Ranked Probability Score (RPS)** — not raw accuracy.
> Goal: be well-calibrated and beat the Elo / market baseline on back-tests.

---

## Architecture

```
Historical matches ─┐
Elo / market value ─┼─► Feature table ─► Match model ─┐
2026 group results ─┘     (as-of date)   (goals)      │
                                                       ▼
                                          Bayesian rating update
                                                       │
                                                       ▼
                                       Monte Carlo tournament sim
                                                       │
                                                       ▼
                              P(reach R16 / QF / SF / Final / Win)
```

## Implementation Phases

| Phase                   | Goal                                              | Key output                       |
| ----------------------- | ------------------------------------------------- | -------------------------------- |
| **0. Scaffold**         | Project structure, deps, config                   | this repo                        |
| **1. Data pipeline**    | Ingest results + Elo + market values, leak-free   | `data/processed/matches.parquet` |
| **2. Features**         | As-of features (ratings, form, rest, host)        | `build_features.py`              |
| **3. Baseline model**   | Dixon-Coles Poisson; establish RPS floor          | `models/dixon_coles.py`          |
| **4. Challenger model** | LightGBM goal model; beat baseline on temporal CV | `models/gbm.py`                  |
| **5. Rating update**    | Bayesian Elo update from 2026 group games         | `ratings/elo.py`                 |
| **6. Simulation**       | Encode exact 2026 format + tiebreakers; MC sim    | `simulation/`                    |
| **7. Back-test**        | Validate on 2014/2018/2022; calibrate             | `scripts/04_backtest.py`         |
| **8. Predict**          | Feed 2026 groups → knockout probabilities         | `scripts/05_predict_2026.py`     |

## Project layout

```
wc2026-predictor/
├── config/tournament_2026.yaml   # groups, host cities, bracket rules
├── data/{raw,processed}/         # datasets (gitignored)
├── src/
│   ├── config.py                 # paths & constants
│   ├── data/                     # ingest + feature building
│   ├── ratings/elo.py            # Elo + Bayesian updating
│   ├── models/                   # dixon_coles.py, gbm.py
│   ├── simulation/               # tournament engine + tiebreakers
│   └── evaluation/metrics.py     # RPS, log loss, calibration
├── scripts/                      # numbered runnable pipeline steps
└── tests/                        # tiebreaker / format unit tests
```

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/01_download_data.py
```
