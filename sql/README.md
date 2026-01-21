# MySQL 库表说明（LAOWANG）

本目录用于**落地可执行 SQL**（不依赖 ORM），以及说明「库/表」的用途与依赖关系。

## 1) 初始化

- 建议使用：`sql/schema_mysql.sql`
- 也可以用项目自带初始化命令（等价建表）：`python astock_analyzer.py init-db`

## 2) 表清单与用途

### 2.1 基础表

- `stock_info`
  - 股票基础信息：`stock_code`、`name`
  - 主要被 `fhkq.py`、`laowang.py` 通过 JOIN 使用

- `stock_daily`
  - 日线 OHLCV：`open/high/low/close/volume/amount`
  - 全部模型依赖（更新管线写入；`fhkq.py`/`bingwu.py`/回测读取）

### 2.2 指标与结构表

- `stock_indicators`
  - 技术指标：MA/RSI/MACD/ATR 等（由 pipeline 计算写入）

- `stock_levels`
  - 支撑/压力位（由 pipeline 计算写入）

### 2.3 评分表（LAOWANG v3）

- `stock_scores_v3`
  - 核心评分结果：`total_score` + 分项 + `status_tags`（JSON 文本）
  - `laowang.py` 输出 CSV 主要来源

- `stock_scores`
  - 旧评分表（保留兼容；当前主要用 v3）

### 2.4 后验表现表（可选）

- `stock_future_perf`
  - 信号后 ne 日窗口表现统计（`future-perf` 写入）

### 2.5 视图（MySQL 专用）

- `vw_stock_pool_v3_latest`
  - `stock_scores_v3` 最新评分日的股票池视图（方便 BI/查询）

### 2.6 模型输出表（用于 UI/BI；可增量物化）

- `model_runs`
  - 记录每个模型每天是否“已计算”（即使当天 0 条信号也会写一条 run 记录）
  - 主键：`(model_name, trade_date)`

- `model_laowang_pool`
  - LAOWANG 每日池（TopN / 阈值由 `models_update.py` 控制）
  - 主键：`(trade_date, stock_code)`

- `model_fhkq`
  - FHKQ 每日信号明细（只存满足条件的少量股票）
  - 主键：`(trade_date, stock_code)`

## 3) 索引建议

`sql/schema_mysql.sql` 已包含项目当前使用的关键索引：
- `idx_stock_daily_date`：按日期查询/抽样
- `idx_stock_scores_v3_date`：按评分日导出
- `idx_stock_future_perf_signal_date`：按信号日统计
