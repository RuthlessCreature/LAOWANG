# LAOWANG / YWCX / STWG / FHKQ

A 股日线数据 + 四个模型评分 + 只读 Web UI。数据抓取默认使用 BaoStock（`getDataBaoStock.py`），评分模型包含：老王 / 阳痿次新 / 缩头乌龟 / 粪海狂蛆。

## 结构与入口

| 脚本 | 作用 |
| --- | --- |
| `init.py` | 初始化数据库（建库 + 建表） |
| `getDataBaoStock.py` | 从 BaoStock 拉取 K 线，写入 `stock_info` / `stock_daily`，可选分钟线（`--frequency`）写入 `stock_minute` |
| `scoring_laowang.py` | 计算老王评分，写入 `stock_scores_v3` / `stock_levels` / `model_laowang_pool` |
| `scoring_ywcx.py` | 计算阳痿次新评分，写入 `stock_scores_ywcx` / `model_ywcx_pool` |
| `scoring_stwg.py` | 计算缩头乌龟评分，写入 `stock_scores_stwg` / `model_stwg_pool` |
| `scoring_fhkq.py` | 计算粪海狂蛆连板信号，写入 `model_fhkq` |
| `everyday.py` | 每日自动流程（增量抓取 + 评分），抓取阶段强制 `workers=1` |
| `ui.py` | 只读 UI（不主动更新数据），保留每个交易日 17:35 自动调度 |
| `tgBot.py` | Telegram 机器人 / 推送辅助（查询四个股票池或推送最新结果） |

> 历史抓取脚本已迁移到 legacy/getData.py 和 legacy/getData2.py，当前主流程默认使用 getDataBaoStock.py。

## 快速开始

1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
2. 配置数据库
   - 编辑 `config.ini`（推荐写 `db_url`）
   - 或使用 `--db-url` / `--db` / 环境变量 `ASTOCK_DB_URL`

   优先级：`--db-url` > `ASTOCK_DB_URL` > `--db` > `config.ini` > `data/stock.db`
3. 初始化表结构
   ```bash
   python init.py --config config.ini
   ```
4. 拉取 K 线数据（BaoStock）
   ```bash
   python getDataBaoStock.py --config config.ini --start-date 20200101 --end-date 20260123 --workers 1
   ```
   可选：只拉分钟线（不会拉日线）
   ```bash
   # 频率仅支持 d/5/15/30/60（BaoStock 不支持 1 分钟）
   # 网络不稳可降低 workers 或增加重试：--minute-retries 5 --minute-retry-sleep 1.5
   python getDataBaoStock.py --config config.ini --frequency 5 --start-date 20260201 --end-date 20260213
   # 回测范围控制（仅分钟线）：最近 N 天
   python getDataBaoStock.py --config config.ini --frequency 5 --end-date 20260212 --minute-lookback-days 20
   # 增量回补窗口（仅分钟线）：默认 3 天；0 表示严格增量
   python getDataBaoStock.py --config config.ini --frequency 5 --minute-backfill-days 0 --start-date 20260201 --end-date 20260212
   ```
5. 计算模型
   ```bash
   python scoring_laowang.py --config config.ini --start-date 2026-01-01 --end-date 2026-01-23 --workers 32 --top 200 --min-score 60
   python scoring_ywcx.py    --config config.ini --start-date 2026-01-01 --end-date 2026-01-23 --workers 32 --top 120 --min-score 55
   python scoring_stwg.py    --config config.ini --start-date 2026-01-01 --end-date 2026-01-23 --workers 32 --top 150 --min-score 55
   python scoring_fhkq.py    --config config.ini --start-date 2026-01-01 --end-date 2026-01-23 --workers 16
   ```
6. 启动 UI
   ```bash
   python ui.py --config config.ini --start-date 20260101
   # 浏览 http://127.0.0.1:8000
   ```

## 一条命令跑每日流程

```bash
python everyday.py --config config.ini --initial-start-date 2020-01-01
```

- 自动根据 `stock_daily` 的最新日期增量抓取
- 抓取阶段强制 `workers=1`（即使传参也会覆盖）
- 评分阶段参数仍可通过 CLI 覆盖

## UI 与自动更新

- UI **只读**：仅从数据库读取模型结果
- 默认 **每个交易日（周一至周五）17:35** 触发 `everyday.py`（以服务器时间为准）
- **启动 UI 不会立即触发** 自动任务
- 可通过 `--disable-auto-update` 关闭自动任务
- 可通过 `--auto-time HH:MM` 指定自动时间

## Telegram Bot

- 运行轮询服务：`python tgBot.py --config config.ini --mode serve`
- 手动推送一次：`python tgBot.py --config config.ini --mode push`
- Token 默认读取 `TG_BOT_TOKEN`（可覆盖默认 8322336287:AAHR4RqsL1SwZsYuRfzNL_rbMNUPL87Bd0c）
- 订阅者列表：`data/tg_subscribers.json`（与 bot 对话一次即可自动加入）
- everyday.py 完成后会自动调用 push，将当日四个股票池推送给所有订阅者
- 消息为纯文本格式（无 Markdown），避免 Telegram 解析失败
- 代理：`--proxy` 默认为 `http://127.0.0.1:7890`，可设为空字符串 / `none` 关闭，也可用环境变量 `TG_BOT_PROXY`

## 数据表（MySQL / SQLite 通用）

| 表名 | 说明 |
| --- | --- |
| `stock_info` | 股票基础信息（含流通市值） |
| `stock_daily` | 日线 OHLCV |
| `stock_minute` | 分钟线 OHLCV（frequency=5/15/30/60） |
| `stock_scores_v3` | 老王评分 |
| `stock_levels` | 支撑/阻力明细 |
| `stock_scores_ywcx` | 阳痿次新评分 |
| `stock_scores_stwg` | 缩头乌龟评分 |
| `model_laowang_pool` | 老王 TopN 池 |
| `model_ywcx_pool` | 阳痿次新 TopN 池 |
| `model_stwg_pool` | 缩头乌龟 TopN 池 |
| `model_fhkq` | 粪海狂蛆连板信号 |

## 免责声明

本项目仅供学习与交流，不构成任何投资建议。风险自担。

## DailyReview / 次日接力（新）

- UI 的“次日接力方案（排除一字板）”现在**只读取数据库结果**（`model_relay_pool`），不再在 UI 内做回测训练。
- 市场情绪相关指标在页面刷新时实时计算（不落库），用于明日决策展示。
- 模型改为**单文件可替换**：默认 `models/relay_model_active.npz`。

### 新增脚本

- 采集池子数据（Xuangubao/涨停宝）并入库：  
  `python getDataDailyReview.py --config config.ini --start-date 20250101 --end-date 20260213`
- 训练并导出可替换模型文件：  
  `python fit_relay_model_file.py --config config.ini --model-file models/relay_model_active.npz`
- 用模型全量/区间打分并写库（覆盖旧结果）：  
  `python scoring_relay.py --config config.ini --start-date 20250101 --end-date 20260213 --model-file models/relay_model_active.npz --full-rebuild`
- 每日增量流程（数据 + 打分）：  
  `python everydayReview.py --config config.ini`

### UI 自动任务

- `everyday.py`：默认每个交易日 `17:35`
- `everydayReview.py`：默认每个交易日 `21:00`
- 可通过 UI 启动参数控制：
  - `--disable-auto-update` / `--auto-time HH:MM`
  - `--disable-auto-review-update` / `--auto-review-time HH:MM`
  - `--auto-review-model-file models/relay_model_active.npz`

### 新增数据表

| 表名 | 说明 |
| --- | --- |
| `daily_review_akshare_pool` | Xuangubao（涨停宝）涨停/跌停/炸板池原始数据 |
| `daily_review_xgb_limit_up` | 涨停宝涨停池明细（题材、连板等） |
| `model_relay_pool` | 次日接力模型打分结果（UI 直接读取） |
| `model_relay_registry` | 当前/历史模型文件注册信息（版本、hash、meta） |

