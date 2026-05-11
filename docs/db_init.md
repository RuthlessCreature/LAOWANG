# 数据库初始化与排错

## 连接方式

推荐在 `config.ini` 中配置完整 SQLAlchemy URL：

```ini
[database]
db_url = mysql+pymysql://user:password@127.0.0.1:3306/astock?charset=utf8mb4
```

也可以通过 `[mysql]` 段、`--db-url`、`--db` 或环境变量 `ASTOCK_DB_URL` 指定。

优先级：

`--db-url` > `ASTOCK_DB_URL` > `--db` > `config.ini` > `data/stock.db`

## 初始化

```bash
python init.py --config config.ini
```

该命令会创建主流程需要的核心表：

- `stock_info`
- `stock_daily`
- `stock_ingest_watermark`
- `stock_scores_v3`
- `stock_levels`
- `stock_scores_ywcx`
- `stock_scores_stwg`
- `model_laowang_pool`
- `model_ywcx_pool`
- `model_stwg_pool`
- `model_fhkq`
- `daily_review_akshare_pool`
- `daily_review_xgb_limit_up`
- `model_relay_pool`
- `model_relay_registry`
- `strategy_signal_log`
- `strategy_trade_journal`

分钟线表 `stock_minute` 会在首次运行 `getDataBaoStock.py --frequency 5/15/30/60` 时自动创建。

## 性能相关表

`stock_ingest_watermark` 保存每只股票、每个频率的最新入库日期。下载任务优先读取该表，避免在 `stock_minute` 体量很大时反复执行全表聚合。

`init.py` 还会补充常用查询索引：

```sql
idx_stock_daily_date_code
idx_scores_v3_date_score
idx_scores_stwg_date_score
idx_scores_ywcx_date_score
idx_model_fhkq_date_score
idx_model_relay_date_rank_score
idx_strategy_signal_model_date
idx_strategy_trade_model_buy_date
```

`strategy_signal_log` 用于保存每个信号当时的特征、模型版本和行动计划；`strategy_trade_journal` 用于保存人工执行、卖出原因、股数、手续费、盈亏和备注。

手工记录一笔买入：

```bash
python3 record_trade.py buy --signal-date 2026-05-11 --model relay --stock-code 000005 --buy-date 2026-05-12 --buy-price 6.70 --shares 1000 --buy-fee 5
```

手工记录卖出并导出复盘：

```bash
python3 record_trade.py sell --model relay --stock-code 000005 --sell-date 2026-05-13 --sell-price 7.10 --sell-fee 6 --exit-reason t2_close_exit
python3 record_trade.py export --output-dir reports/trade_journal/20260513
```

```sql
SELECT frequency, COUNT(*) AS stock_count, MAX(latest_date) AS latest_date
FROM stock_ingest_watermark
GROUP BY frequency;
```

如果需要重建水位，可以先备份数据库，再清空该表并重新运行一次对应频率的下载任务。程序会从基础 K 线表回填水位。

## 常用排错 SQL

日线是否更新：

```sql
SELECT MAX(date) FROM stock_daily;
SELECT COUNT(*) FROM stock_daily WHERE date='2026-04-30';
```

分钟线规模和水位：

```sql
SELECT table_rows, data_length, index_length
FROM information_schema.tables
WHERE table_schema = DATABASE()
  AND table_name = 'stock_minute';

SELECT stock_code, frequency, latest_date
FROM stock_ingest_watermark
ORDER BY updated_at DESC
LIMIT 20;
```

模型输出是否存在：

```sql
SELECT COUNT(*) FROM model_laowang_pool WHERE trade_date='2026-04-30';
SELECT COUNT(*) FROM model_ywcx_pool    WHERE trade_date='2026-04-30';
SELECT COUNT(*) FROM model_stwg_pool    WHERE trade_date='2026-04-30';
SELECT COUNT(*) FROM model_fhkq         WHERE trade_date='2026-04-30';
SELECT COUNT(*) FROM model_relay_pool   WHERE trade_date='2026-04-30';
```

## SQLite 提示

使用 `--db data/stock.db` 时，脚本会自动创建目录并启用 WAL。SQLite 适合轻量试跑；如果要长期保存分钟线，建议使用 MySQL。
