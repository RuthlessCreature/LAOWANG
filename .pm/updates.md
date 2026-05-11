## 2026-03-04
- Update model iteration

## 2026-04-30
- Added ingest watermarks, chunked DB upserts, process-shard download orchestration, and documentation cleanup.
- Moved inactive optimization/backtest scripts and root-level generated artifacts into legacy.

## 2026-05-05
- Reviewed model market viability and wrote `reports/market_viability_review_20260505.md`.
- Identified Relay as the only near-term real-money candidate; LAOWANG/STWG should remain scanners, YWCX/FHKQ need more samples before live use.
- Added a manual T+1 open execution rectification plan in `reports/manual_signal_rectification_plan_20260505.md`.
- Revised the manual plan to treat Relay as a hybrid watchlist + auction filter + intraday trigger flow, while keeping other models on T+1 open execution.
- Added `manual_open_v1` strict ordinary-model follow backtest, with `manual_open_trades.csv`, `manual_open_equity.csv`, and `manual_open_summary.json`.
- Added strategy audit tables and performance indexes in `init.py`, then verified them in the configured MySQL database.
- Added `risk_summary.csv` and `monthly_returns.csv` to surface max single-trade loss, consecutive losses, and monthly return slices.

## 2026-03-04
- Add Medusa A

## 2026-03-03
- Update on npz model

## 2026-02-23
- Add daily review

## 2026-01-27
- d d d

## 2026-01-27
- add sop
