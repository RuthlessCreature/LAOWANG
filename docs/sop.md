# Daily SOP

## 1. Initialize

```bash
pip install -r requirements.txt
python init.py --config config.ini
```

This SOP assumes MySQL as the runtime database. Database resolution:

```text
--db-url > ASTOCK_DB_URL > config.ini [database].db_url > config.ini [mysql]
```

SQLite is reserved for smoke tests or explicit temporary probes.

## 2. Daily Data + Four Scores

Recommended default:

```bash
python everyday.py --config config.ini \
  --initial-start-date 2020-01-01 \
  --getdata-workers 1 \
  --getdata-shards 1 \
  --getdata-write-chunk-size 20000
```

Notes:

- BaoStock is single-flight: worker concurrency and process shards are capped at 1 to avoid provider blocking.
- Transient BaoStock socket errors are retried with relogin and exponential backoff.
- `everyday.py` reads the latest `stock_daily` date and only pulls missing daily bars.
- After data update it runs `scoring_laowang.py`, `scoring_ywcx.py`, `scoring_stwg.py`, and `scoring_fhkq.py`.
- Minute bars are not part of the active workflow.

Manual daily data pull:

```bash
python getDataBaoStock.py --config config.ini \
  --frequency d \
  --start-date 20200101 \
  --end-date 20260513 \
  --workers 1 \
  --process-shards 1 \
  --upsert-chunk-size 20000
```

## 3. Manual Scoring

```bash
python scoring_laowang.py --config config.ini --start-date 2026-01-01 --end-date 2026-05-13 --workers 16 --top 200 --min-score 60
python scoring_stwg.py    --config config.ini --start-date 2026-01-01 --end-date 2026-05-13 --workers 16 --top 150 --min-score 55
python scoring_ywcx.py    --config config.ini --start-date 2026-01-01 --end-date 2026-05-13 --workers 16 --top 120 --min-score 55
python scoring_fhkq.py    --config config.ini --start-date 2026-01-01 --end-date 2026-05-13 --workers 8
```

## 4. Generate Trade Plan

```bash
python generate_trade_plan.py --config config.ini --write-db
```

Outputs:

- `ordinary_next_day_plan.csv`: four-model rows with `skip` / `conditional_buy`, `t1_buy_condition`, and `t2_sell_condition`.
- `signal_audit.csv`: full audit rows with feature JSON and action plan JSON.
- `strategy_signal_log`: written when `--write-db` is enabled.

Dry run:

```bash
python generate_trade_plan.py --config config.ini --trade-date 2026-05-11 --no-write-db
```

## 5. Manual Buy/Sell Journal

Buy:

```bash
python record_trade.py buy \
  --signal-date 2026-05-11 --model laowang \
  --stock-code 000001 --buy-date 2026-05-12 \
  --buy-price 12.30 --shares 1000
```

Sell:

```bash
python record_trade.py sell \
  --model laowang --stock-code 000001 \
  --sell-date 2026-05-14 --sell-price 12.95 \
  --exit-reason t2_protocol_exit
```

Export:

```bash
python record_trade.py export --output-dir reports/trade_journal/
```

## 6. UI / Telegram

```bash
python ui.py --config config.ini --host 127.0.0.1 --port 8765
```

Telegram:

```powershell
$env:TG_BOT_TOKEN="123456:REPLACE_WITH_YOUR_TOKEN"
python tgBot.py --config config.ini --mode serve
```

## 7. Smoke Test

```bash
python tests/smoke_test_generate_trade_plan.py
```

## 8. Quick SQL Checks

```sql
SELECT MAX(date) FROM stock_daily;
SELECT COUNT(*) FROM model_laowang_pool WHERE trade_date='2026-05-13';
SELECT COUNT(*) FROM model_stwg_pool    WHERE trade_date='2026-05-13';
SELECT COUNT(*) FROM model_ywcx_pool    WHERE trade_date='2026-05-13';
SELECT COUNT(*) FROM model_fhkq         WHERE trade_date='2026-05-13';

SELECT model, COUNT(*)
FROM strategy_signal_log
WHERE signal_date='2026-05-13'
GROUP BY model;
```
