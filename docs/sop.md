# SOP（每日运行流程）

## 0. 目标

1. 使用单一数据库保存 K 线与模型结果  
2. 日常流程覆盖“抓取 → 评分 → 展示”  
3. UI 只读展示，自动任务固定在每个交易日（周一至周五）17:35（启动 UI 不会立即触发）  

## 1. 环境准备

- 安装依赖
  ```bash
  pip install -r requirements.txt
  ```
- 配置数据库（任选其一）
  - `config.ini` 中填写 `db_url` 或 `[mysql]`
  - CLI：`--db-url` / `--db`
  - 环境变量：`ASTOCK_DB_URL`

优先级：`--db-url` > `ASTOCK_DB_URL` > `--db` > `config.ini` > `data/stock.db`

## 2. 初始化（仅一次）

```bash
python init.py --config config.ini
```

会自动创建：
`stock_info` / `stock_daily` / `stock_scores_*` / `model_*`

## 3. 每日流程（推荐）

```bash
python everyday.py --config config.ini --initial-start-date 2026-01-01
```

说明：
- 自动读取 `stock_daily` 的最新日期，按需增量抓取
- 抓取阶段固定使用 `getDataBaoStock.py` 且 **workers=1**
- 评分阶段参数（workers / top / min-score）可通过 CLI 覆盖

## 4. 手动分步流程

### 4.1 拉取 K 线（BaoStock）
```bash
python getDataBaoStock.py --config config.ini --start-date 20200101 --end-date 20250101 --workers 4
# 频率仅支持 d/5/15/30/60（BaoStock 不支持 1 分钟）
# 网络不稳可降低 workers 或增加重试：--minute-retries 5 --minute-retry-sleep 1.5
# 只拉分钟线（不会拉日线）
python getDataBaoStock.py --config config.ini --frequency 5 --start-date 20200201 --end-date 20250101
# 分钟线回测范围控制（最近 N 天）
python getDataBaoStock.py --config config.ini --frequency 5 --end-date 20200101 --minute-lookback-days 20
# 分钟线增量回补窗口（默认 3 天；0=严格增量）
python getDataBaoStock.py --config config.ini --frequency 5 --minute-backfill-days 0 --start-date 20260201 --end-date 20260212
```

### 4.2 计算模型
```bash
python scoring_laowang.py --config config.ini --start-date 2026-01-01 --end-date 2026-01-28 --workers 32 --top 200 --min-score 60
python scoring_ywcx.py    --config config.ini --start-date 2026-01-01 --end-date 2026-01-28 --workers 32 --top 120 --min-score 40
python scoring_stwg.py    --config config.ini --start-date 2026-01-01 --end-date 2026-01-28 --workers 32 --top 150 --min-score 55
python scoring_fhkq.py    --config config.ini --start-date 2026-01-01 --end-date 2026-01-28 --workers 16
```

### 4.3 启动 UI
```bash
python ui.py --config config.ini --start-date 20250101 --port 80 --host 0.0.0.0
# http://127.0.0.1:8000
```

## 5. UI 自动任务

- 默认每个交易日（周一至周五）**17:35** 触发 `everyday.py`（以服务器时间为准）
- 启动 UI 不会立即触发（需等到当日 17:35）
- 关闭自动任务：`--disable-auto-update`
- 调整自动时间：`--auto-time HH:MM`
- Telegram Bot（可选）：
  - `python tgBot.py --config config.ini --mode serve`：开启轮询，支持随时查询股票池
  - `python tgBot.py --config config.ini --mode push`：手动向订阅者推送当日四个股票池
  - 默认 Token 由 `TG_BOT_TOKEN` 或仓库内置值提供，消息以纯文本发送避免 Markdown 报错
  - 代理：`--proxy` 默认为 `http://127.0.0.1:7890`，传空/none 关闭，可用 `TG_BOT_PROXY` 覆盖
  - everyday.py 结束后会自动触发推送

## 6. 常用校验

- `stock_daily` 是否有当日数据：
  ```sql
  SELECT MAX(date) FROM stock_daily;
  ```
- 模型是否有输出：
  ```sql
  SELECT COUNT(*) FROM model_laowang_pool WHERE trade_date='2026-01-23';
  SELECT COUNT(*) FROM model_ywcx_pool   WHERE trade_date='2026-01-23';
  SELECT COUNT(*) FROM model_stwg_pool   WHERE trade_date='2026-01-23';
  SELECT COUNT(*) FROM model_fhkq        WHERE trade_date='2026-01-23';
  ```

## 7. 常见问题

- **UI 显示 load error**
  - 检查 UI 控制台 / 后端日志
  - 确认 DB 配置是否指向正确库
  - 检查对应 `model_*` 表是否有该交易日的数据
- **自动任务不执行**
  - 确认当前时间是否已过 17:35（以服务器本地时间为准）
  - 检查是否启用了 `--disable-auto-update`
  - 查看 UI 启动日志中是否打印 “自动任务每个交易日 17:35 运行”

## 8. 次日接力流程（dailyReview）

### 8.1 首次初始化（建议全量）

1) 初始化新表（若已存在会跳过）

```bash
python init.py --config config.ini
```

2) 训练并导出单文件模型

```bash
python fit_relay_model_file.py --config config.ini --model-file models/relay_model_active.npz
```

3) 拉取池子数据（Xuangubao/涨停宝）

```bash
python getDataDailyReview.py --config config.ini --start-date 20200101 --end-date 20260213
```

4) 计算接力结果并覆盖旧结果

```bash
python scoring_relay.py --config config.ini --start-date 20250101 --end-date 20260213 --model-file models/relay_model_active.npz --full-rebuild
```

### 8.2 日常增量

```bash
python everydayReview.py --config config.ini
```

说明：该命令会自动执行 `getDataDailyReview.py` + `scoring_relay.py`，只跑增量区间并覆盖区间旧值。

### 8.3 UI 夜间自动任务

- `everyday.py` 默认每个交易日 `17:35`
- `everydayReview.py` 默认每个交易日 `21:00`

可选参数：

```bash
python ui.py --config config.ini \
  --auto-time 17:35 \
  --auto-review-time 21:00 \
  --auto-review-model-file models/relay_model_active.npz
```

禁用任一任务：

- `--disable-auto-update`
- `--disable-auto-review-update`

### 8.4 快速校验 SQL

```sql
SELECT COUNT(*) FROM daily_review_akshare_pool WHERE trade_date='2026-02-13';
SELECT COUNT(*) FROM daily_review_xgb_limit_up WHERE trade_date='2026-02-13';
SELECT COUNT(*) FROM model_relay_pool WHERE trade_date='2026-02-13';
SELECT model_version, updated_at FROM model_relay_registry ORDER BY updated_at DESC LIMIT 5;
```

### 8.5 模型替换规范

- 线上读取模型文件路径：`models/relay_model_active.npz`
- 替换方式：直接覆盖该文件，然后重新跑 `scoring_relay.py`（增量或全量）
- UI 不在页面内训练模型，只读取 `model_relay_pool` 进行“次日接力方案”展示。

