# LAOWANG

A-share daily-line quant workflow for four ordinary numeric models:

- `LAOWANG`
- `STWG`
- `YWCX`
- `FHKQ`

The active production path is deliberately small:

```text
BaoStock daily bars -> stock_daily -> scoring_*.py -> model_*_pool
                                     -> generate_trade_plan.py
                                     -> strategy_signal_log / CSV
                                     -> UI / Telegram / manual trade journal
```

## Active Files

| File | Purpose |
| --- | --- |
| `init.py` | Create the active database tables and indexes. |
| `getDataBaoStock.py` | Download BaoStock K-line data and maintain ingest watermarks. |
| `everyday.py` | Daily pipeline: BaoStock daily update, then four scoring models. |
| `scoring_laowang.py` | LAOWANG score and pool. |
| `scoring_stwg.py` | STWG score and pool. |
| `scoring_ywcx.py` | YWCX score and pool. |
| `scoring_fhkq.py` | FHKQ event score and pool. |
| `strategy_protocols.py` | Versioned model protocols, thresholds, T+1 buy rules, T+2+ sell rules. |
| `generate_trade_plan.py` | Generate `ordinary_next_day_plan.csv` and `signal_audit.csv`; optionally write `strategy_signal_log`. |
| `record_trade.py` | Manual buy/sell journal and export. |
| `ui.py` | Lightweight read-only Web UI for four pools, plans, and trade records. |
| `tgBot.py` | Telegram query/push bot. |
| `tests/smoke_test_generate_trade_plan.py` | SQLite smoke test for plan generation and DB upsert. |

## Setup

```bash
pip install -r requirements.txt
python init.py --config config.ini
```

Runtime config priority:

```text
--db-url > ASTOCK_DB_URL > --db > config.ini > data/stock.db
```

Keep real DB passwords, Telegram tokens, and proxy values out of committed docs.
Use `config.example.ini` and `.env.example` as placeholders only.

## Daily Workflow

```bash
python everyday.py --config config.ini --initial-start-date 2020-01-01
python generate_trade_plan.py --config config.ini --write-db
python ui.py --config config.ini
```

Open the UI at:

```text
http://127.0.0.1:8765
```

## BaoStock Performance Notes

BaoStock is fragile under high concurrency. Treat two independent connections as the practical upper bound.

Recommended daily update:

```bash
python everyday.py --config config.ini \
  --getdata-workers 2 \
  --getdata-shards 2 \
  --getdata-write-chunk-size 20000
```

`getDataBaoStock.py` now caps `--process-shards` at `2`. Daily writes are batched before database upsert to reduce transaction overhead.

Minute bars remain available through `getDataBaoStock.py --frequency 5/15/30/60`, but they are no longer part of the active strategy path. Do not run minute backfills unless a new model explicitly needs them.

## Trade Plan Contract

Each model output now carries:

- `t1_buy_condition`: the T+1 manual buy condition.
- `t2_sell_condition`: the T+2 or later exit condition.
- `planned_position_pct`: suggested position size when the precheck passes.
- `skip_reason`: why a row is not eligible for T+1 review.

Generated files:

```text
reports/trade_plans/YYYYMMDD/ordinary_next_day_plan.csv
reports/trade_plans/YYYYMMDD/signal_audit.csv
```

## Manual Trade Journal

```bash
python record_trade.py buy \
  --signal-date 2026-05-11 --model laowang \
  --stock-code 000001 --buy-date 2026-05-12 \
  --buy-price 12.30 --shares 1000

python record_trade.py sell \
  --model laowang --stock-code 000001 \
  --sell-date 2026-05-14 --sell-price 12.95 \
  --exit-reason t2_protocol_exit

python record_trade.py export --output-dir reports/trade_journal/
```

## API Endpoints

| Route | Purpose |
| --- | --- |
| `/` | Web UI. |
| `/api/dates` | Available trade dates. |
| `/api/status` | Daily row counts and pool counts. |
| `/api/model/laowang` | LAOWANG pool. |
| `/api/model/stwg` | STWG pool. |
| `/api/model/ywcx` | YWCX pool. |
| `/api/model/fhkq` | FHKQ pool. |
| `/api/plan` | Four-model trade plan from `strategy_signal_log`. |
| `/api/positions` | Manual trade journal rows. |

## Smoke Test

```bash
python tests/smoke_test_generate_trade_plan.py
```

## Active Tables

| Table | Purpose |
| --- | --- |
| `stock_info` | Stock metadata. |
| `stock_daily` | Daily OHLCV. |
| `stock_ingest_watermark` | Latest ingested date per stock/frequency. |
| `stock_scores_v3` / `stock_levels` | LAOWANG scores and levels. |
| `stock_scores_stwg` / `model_stwg_pool` | STWG score and pool. |
| `stock_scores_ywcx` / `model_ywcx_pool` | YWCX score and pool. |
| `model_laowang_pool` | LAOWANG pool. |
| `model_fhkq` | FHKQ pool. |
| `strategy_signal_log` | Signal audit and action plan JSON. |
| `strategy_trade_journal` | Manual execution journal. |

## Operator Notes

- Keep daily-line models boring and auditable first; do not reintroduce minute data until a concrete feature proves it pays rent.
- Use `strategy_signal_log` as the contract between model output and manual execution. It gives you replayable decisions instead of half-remembered screenshots.
- FHKQ is an event model. Treat size as deliberately small unless live journal evidence improves.
- YWCX is active again, but with smaller size until sample quality is better.

This project is for research and tooling, not investment advice.
