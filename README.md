# LAOWANG / STWG / FHKQ —— A 股日线分析工具集（极简版）

本仓库重新收敛为 5 个单文件脚本，职责清晰：

| 脚本 | 作用 |
| --- | --- |
| `init.py` | 初始化数据库（自动建库 + 表结构） |
| `getData.py` | 通过 AkShare 拉取 A 股 K 线并写入 DB，支持日期范围 + 多线程 + 进度条 |
| `scoring_laowang.py` | 基于 K 线数据计算 LAOWANG v3 评分，写入 `stock_scores_v3` / `model_laowang_pool` |
| `scoring_stwg.py` | 依据 docs/scoring_stwg.md 的缩头乌龟模型，写入 `stock_scores_stwg` / `model_stwg_pool` |
| `scoring_fhkq.py` | 计算 FHKQ 连板开板信号，写入 `model_fhkq` |
| `everyday.py` | “每日流程”脚本：按需运行 getData → scoring_laowang → scoring_stwg → scoring_fhkq |
| `ui.py` | 只读 Web UI，直接从数据库展示模型结果（含 status_tags 徽章，内置 15:05 自动任务） |

## 快速开始

1. **准备 Python 环境**
   ```bash
   pip install -r requirements.txt
   ```
2. **配置数据库**
   - 编辑 `config.ini`（推荐写 `db_url` 或 `[mysql]`）
   - 也支持 CLI `--db-url` / `--db` / 环境变量 `ASTOCK_DB_URL`
3. **初始化表结构**
   ```bash
   python init.py --config config.ini
   ```
4. **拉取 K 线数据**
   ```bash
   python getData.py --config config.ini --start-date 20200101 --end-date 20240531 --workers 32
   ```
5. **计算模型**
   ```bash
   python scoring_laowang.py --config config.ini --start-date 2024-01-01 --end-date 2024-05-31 --workers 32 --top 200
   python scoring_stwg.py    --config config.ini --start-date 2024-01-01 --end-date 2024-05-31 --workers 16 --top 150
   python scoring_fhkq.py    --config config.ini --start-date 2024-01-01 --end-date 2024-05-31 --workers 16
   ```
6. **查看 UI（含自动任务）**
   ```bash
   python ui.py --config config.ini --start-date 20240101
   # 浏览 http://127.0.0.1:8000
   ```
   > UI 默认会在每日 15:05 启动后台线程，自动运行 `everyday.py`。

7. **（可选）手动运行每日流程**
   ```bash
   python everyday.py --config config.ini --initial-start-date 2020-01-01
   ```

## 运行说明

- 所有脚本默认读取 `config.ini`，也可直接传入 `--db-url` 或 `--db`。
- `getData.py`/`scoring_laowang.py`/`scoring_stwg.py`/`scoring_fhkq.py` 均支持 `--start-date` / `--end-date`，格式 `YYYYMMDD` 或 `YYYY-MM-DD`。
- 进度条基于 `tqdm`，并发单位为“股票”（LAOWANG）或“交易日/候选股”（FHKQ）。
- UI 只读展示，如需补齐数据可运行 `everyday.py`（UI 也会在 15:05 自动调度）。
- `ui.py` 支持 `--start-date`，用于限制交易日下拉框只列出最近区间。

## 数据表（MySQL/SQLite 通用）

| 表名 | 说明 |
| --- | --- |
| `stock_info` | 股票基础信息（`getData.py` 写入） |
| `stock_daily` | 日线 OHLCV 数据 |
| `stock_scores_v3` | LAOWANG 单股评分（`scoring_laowang.py` 写入） |
| `stock_levels` | 支撑/阻力明细（`scoring_laowang.py`） |
| `stock_scores_stwg` | STWG 单股评分（`scoring_stwg.py` 写入） |
| `model_laowang_pool` | 每日 LAOWANG TopN（UI 读取） |
| `model_stwg_pool` | 每日 STWG TopN（UI 读取） |
| `model_fhkq` | 每日 FHKQ 信号（UI 读取） |

## recycle_bin

历史版本的多模块实现（`a_stock_analyzer/`、`modeling/`、bingwu 系列等）已全部迁入 `recycle_bin/`，如需参考旧代码可在那里查找。

## 自动任务说明

- `everyday.py` 会自动判断 `stock_daily` 的最新交易日，补齐到当日，然后重算 LAOWANG / STWG / FHKQ。
- `ui.py` 启动时默认开启后台线程，每天 15:05 触发 `everyday.py`（可通过 `--disable-auto-update` 关闭；时间用 `--auto-time HH:MM` 指定）。
- 相关并发参数：
  - `--auto-getdata-workers`
  - `--auto-laowang-workers` / `--auto-stwg-workers` / `--auto-fhkq-workers`
  - `--auto-laowang-top` / `--auto-laowang-min-score`
  - `--auto-stwg-top` / `--auto-stwg-min-score`
  - `--auto-init-start-date`（数据库为空时的起始日）

## 注意

- 本项目仅供学习交流，不构成投资建议；据此操作风险自担。
