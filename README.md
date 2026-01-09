# A股中线右侧交易分析系统（v3：低位主升浪模型）

日线级别选股与回测工具：自动拉取 A 股日线行情 → 入库（推荐 MySQL）→ 计算指标/支撑压力 → **强制风控过滤** → **v3 评分** → 导出股票池 → 回测/统计信号后表现。

> 免责声明：本项目仅用于学习与研究，不构成任何投资建议；请自行承担交易风险。

---

## 功能特性

- **数据源**：AkShare（股票列表、日线行情、市值等）
- **数据库**：默认支持 SQLite；强烈推荐 MySQL 8.x（支持并发写入）
- **并发更新**：按“股票粒度”多线程抓取 + 多线程入库（MySQL 下生效；SQLite 会自动强制 `workers=1`）
- **v3 模型（低位主升浪）**
  - 强制过滤（否决项）：急跌/恐慌放量、高位套牢盘距离过近
  - v3 评分表：`stock_scores_v3`（含 `status_tags`：JSON 标签数组）
- **导出股票池（laowang.py）**：按最新评分日导出 CSV（支持 `min-score` / `require-tags` / `min-resistance-distance`）
- **FHKQ（fhkq.py）*F*：连续跌停开板 / 反抽博弈评分（极端情绪模型），从日线数据中筛出当日跌停并输出评分 CSV
- **回测**：随机抽 `nd` 个交易日，筛 `score>=k`，观察后续 `ne` 日窗口最高/最低/最终表现，输出 CSV+Markdown
- **信号后表现**：统计信号后 `ne=5/10/20/30` 的最大涨幅/最大回撤/最终收益，写入 `stock_future_perf` 并输出报告

---

## 目录结构

- `astock_analyzer.py`：命令行入口
- `laowang.py`：v3 股票池导出脚本（从 DB 读取 `stock_scores_v3`）
- `fhkq.py`：FHKQ 连续跌停博弈评分导出脚本
- `a_stock_analyzer/`：核心逻辑
- `config.ini`：数据库配置（MySQL/SQLite）
- `data/`：默认 SQLite DB 目录
- `output/`：导出与报告输出目录
- `document.md`：完整使用说明

---

## 环境与安装

- Python：建议 `3.9+`
- 需要联网（AkShare 拉取行情需要网络）

安装依赖：

```powershell
pip install -r requirements.txt
```

---

## 数据库配置（本地 MySQL 推荐）

编辑根目录 `config.ini`，二选一：

1) 直接写完整连接串（推荐）

```ini
[database]
db_url = mysql+pymysql://user:password@127.0.0.1:3306/astock?charset=utf8mb4
```

2) 填写 MySQL 段（程序自动拼接）

```ini
[mysql]
host = 127.0.0.1
port = 3306
user = your_user
password = your_password
database = astock
charset = utf8mb4
```

连接串优先级（从高到低）：

1. 命令行 `--db-url`
2. 环境变量 `ASTOCK_DB_URL`
3. 命令行 `--db`（强制 SQLite 文件路径）
4. `config.ini`

---

## 快速开始

### 1) 初始化表结构

```powershell
python astock_analyzer.py init-db
```

MySQL 下会额外创建一个视图：`vw_stock_pool_v3_latest`（最新评分日股票池）。

### 2) 全量/增量更新 + v3 评分（并发）

```powershell
python astock_analyzer.py run --workers 16 --start-date 20000101 --end-date 20260107
```

说明：
- 首次全量跑需要较早的 `--start-date`；后续会按每只股票 `MAX(date)+1` 自动增量。
- MySQL 才能真正并发写入；SQLite 会自动强制 `workers=1`。

### 3) 导出股票池

```powershell
python laowang.py --output output/pool.csv --top 200 --min-score 80
```

`status_tags`（JSON 数组）常见值：
`TREND_UP` / `LOW_BASE` / `PULLBACK` / `AT_SUPPORT` / `SPACE_OK` / `NEAR_RESISTANCE` / `RISK_FILTERED`

### 4) FHKQ 连续跌停博弈评分（可选）

```powershell
# 自动取 stock_daily 的最新日期
python fhkq.py

# 指定日期（支持 YYYYMMDD 或 YYYY-MM-DD）
python fhkq.py --trade-date 20260107 --workers 16
```

输出：
- `output/fhkq_YYYYMMDD.csv`

### 5) 信号后表现（ne=5/10/20/30）

```powershell
python astock_analyzer.py future-perf --ne 5,10,20,30 --min-score 80 --workers 16 --out-dir output
```

输出：
- `output/future_perf_*.csv`
- `output/future_perf_*.md`
- 入库：`stock_future_perf`

### 6) 回测（随机 nd 个交易日）

```powershell
python astock_analyzer.py backtest --nd 50 --ne 20 --k 80 --seed 42 --workers 16 --out-dir output
```

输出：
- `output/backtest_*.csv`
- `output/backtest_*.md`

---

## 数据表（核心）

- `stock_info`：股票基础信息（代码/名称）
- `stock_daily`：日线 OHLCV
- `stock_indicators`：技术指标（MA/RSI/MACD/ATR）
- `stock_levels`：支撑/压力位
- `stock_scores_v3`：v3 评分与标签（`status_tags`）
- `stock_future_perf`：信号后 `ne` 日窗口表现

---

## 常见问题

- **跑得很慢/像卡住**：如果库里某些日期没有预计算评分，回测会进入“即时算分”路径，计算量大；建议先用 `run` 把 `stock_scores_v3` 跑起来，或先降低 `--nd`。
- **AkShare 偶发失败**：属于正常现象；可适当降低 `--workers`、重试或分批跑。

---

## 文档

- 完整说明：`document.md`
- LAOWANG v3 评分机制：`docs/scoring_laowang.md`
- FHKQ 评分机制：`docs/scoring_fhkq.md`
- FHKQ 使用说明：`docs/fhkq.md`
- MySQL 快捷说明：`docs/mysql_and_backtest.md`
