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
python getDataBaoStock.py --config config.ini --start-date 20250101 --end-date 20260123 --workers 1
```

### 4.2 计算模型
```bash
python scoring_laowang.py --config config.ini --start-date 2026-01-01 --end-date 2026-01-23 --workers 32 --top 200 --min-score 60
python scoring_ywcx.py    --config config.ini --start-date 2026-01-01 --end-date 2026-01-23 --workers 32 --top 120 --min-score 40
python scoring_stwg.py    --config config.ini --start-date 2026-01-01 --end-date 2026-01-23 --workers 32 --top 150 --min-score 55
python scoring_fhkq.py    --config config.ini --start-date 2026-01-01 --end-date 2026-01-23 --workers 16
```

### 4.3 启动 UI
```bash
python ui.py --config config.ini --start-date 20260101
# http://127.0.0.1:8000
```

## 5. UI 自动任务

- 默认每个交易日（周一至周五）**17:35** 触发 `everyday.py`（以服务器时间为准）
- 启动 UI 不会立即触发（需等到当日 17:35）
- 关闭自动任务：`--disable-auto-update`
- 调整自动时间：`--auto-time HH:MM`

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
