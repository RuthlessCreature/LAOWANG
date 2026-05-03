# LAOWANG

A 股日线、分钟线、模型评分和只读 Web UI。默认数据源为 BaoStock，主流程围绕日线增量、四个评分模型、次日接力模型和 Telegram 推送运行。

## Active Entry Points

| 文件 | 作用 |
| --- | --- |
| `init.py` | 初始化数据库和核心表结构 |
| `getDataBaoStock.py` | 拉取日线/分钟线，写入 `stock_info`、`stock_daily`、`stock_minute` |
| `everyday.py` | 每日主流程：增量抓取日线，然后计算四个评分模型 |
| `everydayReview.py` | 次日接力增量流程：抓取复盘池并写入接力模型结果 |
| `ui.py` | 只读 Web UI 和自动任务调度入口 |
| `tgBot.py` | Telegram 查询与推送 |
| `scoring_laowang.py` | 老王评分 |
| `scoring_ywcx.py` | 阳痿次新评分 |
| `scoring_stwg.py` | 缩头乌龟评分 |
| `scoring_fhkq.py` | FHKQ 连板信号 |
| `scoring_relay.py` | 次日接力模型评分 |

## Quick Start

```bash
pip install -r requirements.txt
python init.py --config config.ini
python everyday.py --config config.ini --initial-start-date 2020-01-01
python ui.py --config config.ini --start-date 20260101
```

BaoStock 需要 `baostock>=0.9.1`。老版本会连接旧的 `www.baostock.com:10030`，可能出现 `10002007 网络接收错误`。

数据库连接优先级：

`--db-url` > `ASTOCK_DB_URL` > `--db` > `config.ini` > `data/stock.db`

## Data Download

日线增量：

```bash
python getDataBaoStock.py --config config.ini --frequency d --start-date 20200101 --end-date 20260430 --workers 4
```

5 分钟线增量：

```bash
python getDataBaoStock.py --config config.ini --frequency 5 --start-date 20260201 --end-date 20260430 --minute-backfill-days 3
```

下载性能参数：

- `--workers`：单进程线程数。BaoStock 单连接仍会串行访问 API，但数据解析和主线程写库可以重叠。
- `--process-shards N`：启动 N 个独立 BaoStock 进程分片，适合大范围补数据。建议先从 `2` 或 `4` 开始。
- `--upsert-chunk-size N`：数据库批量写入分块大小，默认 `5000`。
- `--baostock-proxy`：BaoStock TCP 代理，默认 `auto`；直连失败时会尝试 `BAOSTOCK_PROXY`、`TG_BOT_PROXY`、`http://127.0.0.1:7890`。
- `--minute-lookback-days N`：分钟线只拉最近 N 天。
- `--minute-backfill-days N`：分钟线增量时向前回补 N 天，默认 `3`，设为 `0` 表示严格从下一天开始。

`getDataBaoStock.py` 会维护 `stock_ingest_watermark` 水位表，避免每次启动都扫描大体量分钟线表。

## Daily Pipeline

```bash
python everyday.py --config config.ini \
  --initial-start-date 2020-01-01 \
  --getdata-workers 4 \
  --getdata-shards 2 \
  --getdata-write-chunk-size 5000
```

`everyday.py` 会读取 `stock_daily` 最新日期，自动决定抓取起点，然后依次运行四个评分模型。UI 自动任务默认在每个交易日 17:35 触发该流程。

## Daily Review / Relay

```bash
python fit_relay_model_file.py --config config.ini --model-file models/relay_model_active.npz
python getDataDailyReview.py --config config.ini --start-date 20250101 --end-date 20260430
python scoring_relay.py --config config.ini --start-date 20250101 --end-date 20260430 --model-file models/relay_model_active.npz --full-rebuild
python everydayReview.py --config config.ini
```

UI 默认在每日主流程结束后触发一次 `everydayReview.py`，夜间也可按 `--auto-review-time` 独立调度。

## Core Tables

| 表 | 内容 |
| --- | --- |
| `stock_info` | 股票基础信息 |
| `stock_daily` | 日线 OHLCV |
| `stock_minute` | 分钟线 OHLCV |
| `stock_ingest_watermark` | 每只股票每个频率的入库水位 |
| `stock_scores_v3` / `stock_levels` | 老王评分与支撑阻力 |
| `stock_scores_ywcx` / `stock_scores_stwg` | 两个辅助评分模型 |
| `model_laowang_pool` / `model_ywcx_pool` / `model_stwg_pool` / `model_fhkq` | UI 读取的模型池 |
| `daily_review_akshare_pool` / `daily_review_xgb_limit_up` | 次日接力原始池 |
| `model_relay_pool` / `model_relay_registry` | 次日接力评分结果与模型登记 |

## Disclaimer

本项目仅用于学习和研究，不构成投资建议。
