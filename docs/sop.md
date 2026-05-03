# 每日运行 SOP

## 目标

1. 用单一数据库保存 K 线、复盘池和模型结果。
2. 每日流程覆盖“抓取 -> 评分 -> 展示 -> 推送”。
3. UI 只读展示，自动任务按交易日定时触发。

## 初始化

```bash
pip install -r requirements.txt
python init.py --config config.ini
```

确认 BaoStock 版本：

```bash
python -m pip show baostock
```

版本必须为 `0.9.1` 或更高；旧版会连接旧服务端，容易报 `10002007 网络接收错误`。

数据库连接优先级：

`--db-url` > `ASTOCK_DB_URL` > `--db` > `config.ini` > `data/stock.db`

## 每日主流程

```bash
python everyday.py --config config.ini \
  --initial-start-date 2020-01-01 \
  --getdata-workers 4 \
  --getdata-shards 2 \
  --getdata-write-chunk-size 5000
```

说明：

- 自动读取 `stock_daily` 最新日期并增量抓取。
- `--getdata-workers` 会传给 `getDataBaoStock.py`。
- `--getdata-shards` 会启动多个独立 BaoStock 进程分片，适合大范围补数据。
- 下载完成后依次运行老王、YWCX、STWG、FHKQ 四个评分模型。
- `everyday.py` 结束后会尝试触发 Telegram 推送。

## 手动拉取 K 线

日线：

```bash
python getDataBaoStock.py --config config.ini --frequency d --start-date 20200101 --end-date 20260430 --workers 4
```

分钟线：

```bash
python getDataBaoStock.py --config config.ini --frequency 5 --start-date 20260201 --end-date 20260430 --minute-backfill-days 3
```

大范围补分钟线：

```bash
python getDataBaoStock.py --config config.ini --frequency 5 --start-date 20200101 --end-date 20260430 --workers 1 --process-shards 4 --upsert-chunk-size 5000
```

常用参数：

- `--minute-lookback-days N`：只拉最近 N 天分钟线。
- `--minute-backfill-days N`：增量时向前回补 N 天；`0` 为严格增量。
- `--minute-retries N` / `--minute-retry-sleep S`：分钟线失败重试。
- `--api-min-interval S`：BaoStock 请求最小间隔。
- `--baostock-proxy auto|direct|none|http://127.0.0.1:7890`：BaoStock 数据端口代理设置。

## 手动评分

```bash
python scoring_laowang.py --config config.ini --start-date 2026-01-01 --end-date 2026-04-30 --workers 32 --top 200 --min-score 60
python scoring_ywcx.py    --config config.ini --start-date 2026-01-01 --end-date 2026-04-30 --workers 32 --top 120 --min-score 55
python scoring_stwg.py    --config config.ini --start-date 2026-01-01 --end-date 2026-04-30 --workers 32 --top 150 --min-score 55
python scoring_fhkq.py    --config config.ini --start-date 2026-01-01 --end-date 2026-04-30 --workers 16
```

## 次日接力流程

首次或全量：

```bash
python fit_relay_model_file.py --config config.ini --model-file models/relay_model_active.npz
python getDataDailyReview.py --config config.ini --start-date 20250101 --end-date 20260430
python scoring_relay.py --config config.ini --start-date 20250101 --end-date 20260430 --model-file models/relay_model_active.npz --full-rebuild
```

日常增量：

```bash
python everydayReview.py --config config.ini
```

## UI 自动任务

```bash
python ui.py --config config.ini \
  --start-date 20260101 \
  --auto-time 17:35 \
  --auto-review-time 21:00 \
  --auto-review-model-file models/relay_model_active.npz
```

可选开关：

- `--disable-auto-update`
- `--disable-auto-review-update`
- `--host 0.0.0.0 --port 8000`

## 校验 SQL

```sql
SELECT MAX(date) FROM stock_daily;
SELECT COUNT(*) FROM model_laowang_pool WHERE trade_date='2026-04-30';
SELECT COUNT(*) FROM model_ywcx_pool   WHERE trade_date='2026-04-30';
SELECT COUNT(*) FROM model_stwg_pool   WHERE trade_date='2026-04-30';
SELECT COUNT(*) FROM model_fhkq        WHERE trade_date='2026-04-30';
SELECT COUNT(*) FROM model_relay_pool  WHERE trade_date='2026-04-30';
```

分钟线水位：

```sql
SELECT frequency, COUNT(*), MAX(latest_date)
FROM stock_ingest_watermark
GROUP BY frequency;
```
