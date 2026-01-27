# LAOWANG / YWCX / STWG / FHKQ

A 股日线数据 + 四个模型评分 + 只读 Web UI。数据抓取默认使用 BaoStock（`getDataBaoStock.py`），评分模型包含：老王 / 阳痿次新 / 缩头乌龟 / 粪海狂蛆。

## 结构与入口

| 脚本 | 作用 |
| --- | --- |
| `init.py` | 初始化数据库（建库 + 建表） |
| `getDataBaoStock.py` | 从 BaoStock 拉取 K 线，写入 `stock_info` / `stock_daily` |
| `scoring_laowang.py` | 计算老王评分，写入 `stock_scores_v3` / `stock_levels` / `model_laowang_pool` |
| `scoring_ywcx.py` | 计算阳痿次新评分，写入 `stock_scores_ywcx` / `model_ywcx_pool` |
| `scoring_stwg.py` | 计算缩头乌龟评分，写入 `stock_scores_stwg` / `model_stwg_pool` |
| `scoring_fhkq.py` | 计算粪海狂蛆连板信号，写入 `model_fhkq` |
| `everyday.py` | 每日自动流程（增量抓取 + 评分），抓取阶段强制 `workers=1` |
| `ui.py` | 只读 UI（不主动更新数据），保留每个交易日 17:35 自动调度 |
| `tgBot.py` | Telegram 机器人 / 推送辅助（查询四个股票池或推送最新结果） |

> 仓库中仍保留 `getData.py` / `getData2.py` 等抓取脚本，当前主流程默认使用 `getDataBaoStock.py`。

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

## 数据表（MySQL / SQLite 通用）

| 表名 | 说明 |
| --- | --- |
| `stock_info` | 股票基础信息（含流通市值） |
| `stock_daily` | 日线 OHLCV |
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
