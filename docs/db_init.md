# Database Init And Troubleshooting

## Connection

Prefer environment variables for real credentials:

```bash
set ASTOCK_DB_URL=mysql+pymysql://user:password@127.0.0.1:3306/astock?charset=utf8mb4
```

Config priority:

```text
--db-url > ASTOCK_DB_URL > --db > config.ini > data/stock.db
```

`config.example.ini` and `.env.example` are placeholders only. Do not commit real DB passwords or bot tokens.

## Initialize

```bash
python init.py --config config.ini
```

Active tables:

- `stock_info`
- `stock_daily`
- `stock_ingest_watermark`
- `stock_levels`
- `stock_scores_v3`
- `stock_scores_stwg`
- `stock_scores_ywcx`
- `model_laowang_pool`
- `model_stwg_pool`
- `model_ywcx_pool`
- `model_fhkq`
- `strategy_signal_log`
- `strategy_trade_journal`

`stock_minute` is only created when manually running `getDataBaoStock.py --frequency 5/15/30/60`. It is not used by the active strategy path.

## Indexes

`init.py` creates these active indexes when the database allows it:

```sql
idx_stock_daily_date_code
idx_scores_v3_date_score
idx_scores_stwg_date_score
idx_scores_ywcx_date_score
idx_model_fhkq_date_score
idx_strategy_signal_model_date
idx_strategy_trade_model_buy_date
```

## Watermark

`stock_ingest_watermark` stores the latest ingested date per stock and frequency. Data update reads this table first, so it avoids expensive `stock_daily` scans during incremental runs.

Check:

```sql
SELECT frequency, COUNT(*) AS stock_count, MAX(latest_date) AS latest_date
FROM stock_ingest_watermark
GROUP BY frequency;
```

If the watermark is wrong, back up the database first, clear the affected rows, and run the corresponding data update again. The script can backfill watermarks from existing K-line tables.

## Common Checks

Daily data:

```sql
SELECT MAX(date) FROM stock_daily;
SELECT COUNT(*) FROM stock_daily WHERE date='2026-05-13';
```

Model pools:

```sql
SELECT COUNT(*) FROM model_laowang_pool WHERE trade_date='2026-05-13';
SELECT COUNT(*) FROM model_stwg_pool    WHERE trade_date='2026-05-13';
SELECT COUNT(*) FROM model_ywcx_pool    WHERE trade_date='2026-05-13';
SELECT COUNT(*) FROM model_fhkq         WHERE trade_date='2026-05-13';
```

Trade plan:

```sql
SELECT model, COUNT(*)
FROM strategy_signal_log
WHERE signal_date='2026-05-13'
GROUP BY model;
```

Manual journal:

```sql
SELECT trade_status, COUNT(*)
FROM strategy_trade_journal
GROUP BY trade_status;
```

## SQLite Notes

SQLite is fine for smoke tests and light local experiments. Use MySQL for long-running daily data, because large historical K-line writes are much faster and safer there.
