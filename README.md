# LAOWANG / FHKQ / BINGWU —— A 股日线分析工具集

本仓库提供 3 个日常模型脚本（LAOWANG/FHKQ 按日物化到 MySQL；bingwu 输出 CSV），以及一套公共数据管线模块 `a_stock_analyzer/`。

## 模块说明
- `astock_analyzer.py`：公共数据管线入口（抓取日线 → 入库 → 指标/结构 → 评分）
- `laowang.py`：LAOWANG v3 评分池导出（右侧中线低位主升浪）
- `fhkq.py`：FHKQ 连续跌停开板/反抽博弈评分
- `bingwu.py`：bingwu 超短复盘 & 次日作战（复盘 + 候选股 + 次日计划）
- `bingwu_report.py`：把 `bingwu.py` 的分析结果导出为每日 CSV（用于统一输出口径）

## 约束（当前项目约定）
- 统一使用本地 **MySQL**（配置见 `config.ini`）
- 模型输出写入 MySQL（`model_*` 表），UI 直接读取（不依赖 CSV）
- 每日报告输出到 `outputs/`：`outputs/bingwu_YYYYMMDD.csv`
- 不需要的旧文件/旧产物移动到 `recycle_bin/`（不删除）

## 快速开始
1) 配置 MySQL：编辑 `config.ini`（推荐写 `db_url`）
2) 初始化表结构：
   - `python init.py --config config.ini`
3) 首次跑数（可能较慢）：
   - `python astock_analyzer.py --config config.ini run --start-date 20000101 --end-date YYYYMMDD --workers 16`
4) 每日收盘后（增量更新 K 线 + 物化模型输出 + 导出 bingwu CSV）：
   - `python everyday.py --config config.ini`

## 手工运行（可选）
- LAOWANG：
  - `python laowang.py --output outputs/laowang_YYYYMMDD.csv --top 200 --min-score 60`
- FHKQ：
  - `python fhkq.py --trade-date YYYYMMDD --output outputs/fhkq_YYYYMMDD.csv --workers 16`
- bingwu：
  - `python bingwu_report.py --trade-date YYYYMMDD --output outputs/bingwu_YYYYMMDD.csv`

## 本地 Web UI（可选）
- 启动：`python ui.py --config config.ini`
- 功能：
  - 选择交易日查看 LAOWANG / FHKQ 表格（来自 MySQL `model_*`）
  - 若所选交易日已存在 `stock_daily` 但未计算 `model_*`，UI 会自动补算并显示运行状态
  - 15:05 收盘后的“增量更新”建议用任务计划/cron 运行 `python everyday.py --config config.ini`

## 数据库与排错
DB 解析优先级（高 → 低）：
1. CLI `--db-url`
2. 环境变量 `ASTOCK_DB_URL`
3. CLI `--db`（SQLite 文件路径）
4. `config.ini`
5. 默认 SQLite：`data/stock.db`（不推荐）

工具与物料：
- `scripts/db_doctor.ps1`：诊断当前最终连到哪个 DB、是否存在核心表
- `scripts/models_update_incremental.ps1` / `.bat`：只物化模型输出（智能增量）
- `scripts/models_update_full.ps1` / `.bat`：全量重算（慢）
- `sql/schema_mysql.sql`：MySQL 建库建表 SQL
- `sql/README.md`：库/表说明

## 文档
- `docs/sop.md`：每日运行 SOP
- `docs/db_init.md`：数据库初始化与排错
- `docs/scoring_laowang.md` / `docs/scoring_fhkq.md` / `docs/scoring_bingwu.md`：评分标准

## 免责声明
仅供学习研究，不构成投资建议；风险自担。
