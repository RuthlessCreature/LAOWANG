# 数据库初始化与排错

## 1. 连接方式

- 推荐：在 `config.ini` 中写 `db_url = mysql+pymysql://user:pass@127.0.0.1:3306/astock?charset=utf8mb4`
- 也可填写 `[mysql]` 段，脚本会自动拼接。
- 连接优先级：`--db-url` > `ASTOCK_DB_URL` > `--db` > `config.ini` > `data/stock.db`

## 2. 初始化步骤

```bash
python init.py --config config.ini
```

- 若连接的是 MySQL，会自动 `CREATE DATABASE IF NOT EXISTS`。
- 创建表：`stock_info`、`stock_daily`、`stock_levels`、`stock_scores_v3`、`stock_scores_stwg`、`model_laowang_pool`、`model_stwg_pool`、`model_fhkq`。

## 3. 核心脚本与表对应关系

| 脚本 | 依赖表 | 写入表 |
| --- | --- | --- |
| `getData.py` | `stock_info`*、`stock_daily`* | `stock_info`、`stock_daily` |
| `scoring_laowang.py` | `stock_info`、`stock_daily` | `stock_scores_v3`、`stock_levels`、`model_laowang_pool` |
| `scoring_stwg.py` | `stock_info`、`stock_daily` | `stock_scores_stwg`、`model_stwg_pool` |
| `scoring_fhkq.py` | `stock_info`、`stock_daily` | `model_fhkq` |
| `ui.py` | `model_laowang_pool`、`model_stwg_pool`、`model_fhkq` | —— |

（带 * 的表会在不存在时创建）

## 4. 常见排错 SQL

```sql
-- 确认日线覆盖
SELECT COUNT(*) FROM stock_daily WHERE date='2024-05-31';

-- 查看 LAOWANG 评分是否写入
SELECT COUNT(*) FROM stock_scores_v3 WHERE score_date='2024-05-31';

-- 查看 UI 数据源
SELECT COUNT(*) FROM model_laowang_pool WHERE trade_date='2024-05-31';
SELECT COUNT(*) FROM model_stwg_pool WHERE trade_date='2024-05-31';
SELECT COUNT(*) FROM model_fhkq WHERE trade_date='2024-05-31';
```

## 5. SQLite 提示

- 若使用 `--db data/stock.db`，脚本会自动创建目录并启用 WAL 模式（并发读更快）。
- 单机轻量使用可快速上手；后续迁移至 MySQL 只需改配置并重新 `init.py`。

## 6. 其他

- 旧版的 `a_stock_analyzer/`、`modeling/` 等仍在 `recycle_bin/`，但与新架构无关。
- 若需要自定义表结构，可在运行 `init.py` 前修改脚本中的 DDL 语句。
