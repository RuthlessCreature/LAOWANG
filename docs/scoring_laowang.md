# LAOWANG v3（低位主升浪）评分机制说明

> 风险声明：本模型仅用于学习研究与辅助决策，不构成任何投资建议；请自行承担交易风险。

本文解释“LAOWANG v3”评分（`stock_scores_v3`）的计算逻辑，以及 `laowang.py` 导出股票池时使用的筛选口径。

> 重要提示：`laowang.py` **不计算评分**，它只从数据库导出 `astock_analyzer.py run` 已写入的最新评分结果（表：`stock_scores_v3`）。

---

## 1. 总体流程（从原始日线到评分）

1) 拉取并入库日线 OHLCV → `stock_daily`
2) 计算技术指标 → `stock_indicators`
   - MA20 / MA60 / MA120
   - RSI14
   - MACD（diff/dea/hist）
   - ATR14
3) 计算支撑/压力位 → `stock_levels`
4) 计算 v3 评分与标签 → `stock_scores_v3`

---

## 2. 风险否决（先执行）

只要命中任一否决项，当日评分直接归零，并打上 `RISK_FILTERED`：

### 2.1 急跌/恐慌过滤（Crash Filter）

在最近约 20 个交易日内，满足任一条件即否决：

1) 单日跌幅 ≥ 9%
2) 3 日累计跌幅 ≥ 15%
3) 恐慌放量下跌：`volume > 20 日均量 × 1.8` 且 `close < MA20`

> 直觉：主升浪模型不参与“急跌/恐慌阶段”的抄底。

### 2.2 高位否决（High Position Filter）

计算最近 120 日最高价到当前收盘价的距离：

```
distance = (high_120d_max - close) / high_120d_max
```

若 `distance < 0.15`（距离新高不足 15%），则否决。

> 直觉：右侧中线模型尽量避免追在阶段性高位附近。

---

## 3. 子分项（每项 0～10）

当未触发风险否决时，计算 8 个子分项（每项按 0～10 归一化，最后加权到总分 0～100）。

### 3.1 趋势分（trend_score）

先做一个“快均线过滤”：要求收盘价至少在 MA5 或 MA10 之上，否则趋势分为 0。

然后按均线结构与斜率打分：

- 强趋势：`MA20 > MA60 > MA120` 且 `close > MA60`
  - 若 MA60 在近 15 日是上升的：`10`
  - 否则：`6`
- 均线粘合：MA20/60/120 的离散度很小（`(max-min)/close <= 0.02`）：`3`
- 其它：`0`

### 3.2 回踩分（pullback_score）

定义“接近支撑”（二选一）：

- 接近 `support_level`：`(close - support_level)/close <= at_support_pct`
- 或接近 MA60：`abs(close - MA60)/close <= at_support_pct`

其中 `at_support_pct` 默认 0.03（3%）。

再结合“止跌”条件：

- 接近支撑/MA60 且 `close >= prev_close`：`10`
- 仅接近支撑/MA60：`6`
- 否则：`0`

### 3.3 量价分（volume_price_score）

核心思想：突破/缩量回踩/放量下跌否决。

- 若出现“重挫放量”（最新日 `close < prev_close` 且 `volume >= 1.5 * 20 日均量`）：直接 `0`
- 否则检查是否在近 `breakout_lookback`（默认 60）日内出现过放量突破：
  - `close >= rolling_high_prev`（前 60 日最高价，不含当日）
  - 且 `volume >= 1.3 * 20 日均量`
  - 若出现过突破：
    - 最近 5 日内有“缩量回踩”（`close.diff()<0` 且 `volume <= 0.9 * 20 日均量`）：`10`
    - 否则：`5`
  - 若未出现突破：`5`

> 该项在实现上更偏“结构识别”，所以多数情况下在 5/10 两档之间。

### 3.4 RSI 分（rsi_score）

按 RSI14 区间打分：

| RSI14 | 分数 |
| --- | --- |
| 45～60 | 10 |
| 35～45 | 6 |
| 60～70 | 5 |
| 30～35 | 3 |
| > 70 | 0 |
| 其它 | 0 |

### 3.5 MACD 分（macd_score）

仅使用 diff 与 dea 的关系与金叉/死叉：

- 若发生死叉（`prev_diff >= prev_dea` 且 `diff < dea`）：`0`
- 若发生金叉（`prev_diff <= prev_dea` 且 `diff > dea`）：
  - 且 `diff > 0`：`10`
  - 否则：`5`
- 未交叉：
  - `diff > dea`：`5`
  - 否则：`0`

### 3.6 底部结构分（base_structure_score）

该项由两部分平均得到（都在 0/6/10 档）：

**(a) 平台结构（platform_score）**

在最近 120 日：

- 价格振幅 `amplitude = (high_max - low_min)/low_min <= 0.20` 认为“平台”
- MA60 近 20 日基本走平（不明显下滑）
- ATR14 近 20 日不放大（波动不扩张）

若平台成立：
- 若当日同时满足“放量突破”（同 3.3 的突破条件）：`10`
- 否则：`6`
否则：`0`

**(b) 换手完成度（turnover_score）**

最近 20 日，计算：

```
ratio = avg(volume on up days) / avg(volume on down days)
```

打分：
- `ratio >= 1.8`：`10`
- `1.2 <= ratio < 1.8`：`6`
- 否则：`0`

最终：

```
base_structure_score = (platform_score + turnover_score) / 2
```

### 3.7 空间分（space_score）

以波动率（ATR）与头顶压力共同约束“可操作空间”：

1) 估算 20 日期望波动空间：

```
expected_return = (ATR14 * 20) / close
```

2) 若有压力位且在上方：

```
resistance_distance = (resistance_level - close) / close
```

3) 设最小空间门槛：

```
min_space_pct = near_resistance_pct * 2
```

其中 `near_resistance_pct` 默认 0.05（则门槛 0.10）。

打分逻辑：

- 若压力位缺失：保守按波动潜力
  - `expected_return >= 0.18`：`5`
  - 否则：`0`
- 若压力太近（`resistance_distance < min_space_pct`）：`0`
- 否则取可达空间：

```
reachable_space = min(expected_return, resistance_distance)
```

并打分：
- `reachable_space >= 0.18`：`10`
- `reachable_space >= min_space_pct`：`5`
- 否则：`0`

### 3.8 流通市值分（market_cap_score）

从 AkShare 的“流通市值”快照解析得到（单位：亿元），按区间打分：

| 流通市值（亿） | 分数 |
| --- | --- |
| 30～100 | 10 |
| 100～150 | 6 |
| 150～300 | 3 |
| 其它/缺失 | 0 |

> 直觉：偏好“中等流通市值”的可操作弹性；过大偏慢、过小易失真。

---

## 4. 总分合成（0～100）

总分为 8 个子分项加权求和（每项先按 /10 归一化）：

```
total_score =
  trend_score        * 20%
+ pullback_score     * 15%
+ volume_price_score * 10%
+ rsi_score          * 10%
+ macd_score         *  5%
+ base_structure     * 15%
+ space_score        * 15%
+ market_cap_score   * 10%
```

---

## 5. 标签（status_tags）

当未触发风险否决时，根据关键子分项与位置关系生成标签（JSON 数组）：

- `TREND_UP`：`trend_score >= 6`
- `LOW_BASE`：`base_structure_score >= 6`
- `PULLBACK`：`pullback_score >= 6`
- `AT_SUPPORT`：存在 `support_level` 且 `(close-support)/close <= at_support_pct`
- `SPACE_OK`：`space_score >= 5`
- `NEAR_RESISTANCE`：存在上方压力且 `(resistance-close)/close <= near_resistance_pct`

若触发否决，则只有：`["RISK_FILTERED"]`。

---

## 6. laowang.py 导出筛选口径

`laowang.py` 默认从 `stock_scores_v3` 取 **最新评分日**（`MAX(score_date)`），并执行导出筛选：

- 自动排除 `status_tags` 含 `RISK_FILTERED` 的股票
- 可选：
  - `--min-score`：总分阈值
  - `--require-tags`：必须包含的标签集合（如 `TREND_UP,AT_SUPPORT`）
  - `--min-resistance-distance`：要求 `(resistance_level - close)/close >= x`（用于过滤“压力太近”的票）

