# fhkq（连续跌停开板 / 反抽博弈评分）

> 风险声明：连续跌停博弈存在极高风险。本模块仅用于研究与辅助决策，不构成投资建议。

## 1. 前置条件

1) 安装依赖：

```powershell
pip install -r requirements.txt
```

2) 先用现有流程把日线数据写入数据库（`stock_daily` / `stock_info`）：

```powershell
python astock_analyzer.py init-db
python astock_analyzer.py run --workers 16 --start-date 20000101 --end-date 20260107
```

数据库连接的配置规则与 `astock_analyzer.py` 一致（`--db-url` / `ASTOCK_DB_URL` / `--db` / `config.ini` / 默认 `data/stock.db`）。

## 2. 运行（从数据库导出 CSV）

默认输出：`output/fhkq_YYYYMMDD.csv`（UTF-8 / 逗号分隔）。

```powershell
# 自动取 stock_daily 的最新日期
python fhkq.py

# 指定日期（支持 YYYYMMDD 或 YYYY-MM-DD）
python fhkq.py --trade-date 20260107

# 指定数据库与输出路径
python fhkq.py --trade-date 2026-01-07 --db-url "mysql+pymysql://user:pass@127.0.0.1:3306/astock?charset=utf8mb4" --output output/fhkq_20260107.csv --workers 16
```

## 3. 输出字段（CSV）

| 字段名 | 含义 |
| --- | --- |
| trade_date | 评分日期 |
| stock_code | 股票代码 |
| stock_name | 股票名称 |
| consecutive_limit_down | 连续跌停天数 |
| last_limit_down | 昨日是否跌停（1/0） |
| volume_ratio | 当日成交量 / 5 日均量 |
| amount_ratio | 当日成交额 / 5 日均额 |
| open_board_flag | 当日是否出现开板（1/0） |
| liquidity_exhaust | 流动性衰竭标志（1/0） |
| fhkq_score | 博弈评分（0～100） |
| fhkq_level | 博弈等级（A/B/C/D） |

## 4. 作为库调用（DataFrame）

`fhkq.py` 也提供纯计算函数 `run_fhkq(df_daily, trade_date)`，用于你已经拿到“标准日K DataFrame”的场景。

DataFrame 需要至少包含：
- `stock_code`, `date`, `open`, `high`, `low`, `close`, `volume`, `amount`
- 可选：`stock_name`, `is_st`, `limit_down`（若未提供会按规则推算跌停价）

示例：

```python
import pandas as pd
from fhkq import run_fhkq

df_daily = pd.DataFrame(...)  # 你的标准日K数据（可多股票）
out = run_fhkq(df_daily, trade_date="2026-01-07")
print(out.head())
```

## 5. 硬性剔除（实现与规范一致）

以下股票直接剔除，不参与评分：
1) `is_st == 1`（若提供该列）
2) 股票名称包含：`ST`、`*ST`、`退`
3) 最近 5 日：连续跌停 且 `volume == 0`
4) 最近 10 日累计跌幅 > 60%

