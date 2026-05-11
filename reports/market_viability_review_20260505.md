# LAOWANG 市场化可用性审查

审查日期：2026-05-05  
审查对象：`docs/` 里的四个评分模型、超短复盘 SOP、接力模型代码、真实行情回测输出。  
结论层级：这是一个“选股雷达 + 研究框架”，还不是可直接上真钱的稳定盈利系统。

## 一句话判决

当前 repo 有用，但不能按现在的形态投入市场自动赚钱。

它真正有价值的部分是：

- 数据入库、评分、UI、Telegram 推送已经能形成日常工作流。
- LAOWANG / STWG 可以作为中短线观察池。
- `dailyReview.py` 和 `docs/daily_review_sop.md` 把超短交易员的情绪框架拆成了可量化指标。
- Relay 是唯一接近“可训练交易模型”的模块。

它现在不能稳定盈利的核心原因是：

- 大部分模型只有买入评分，没有完整交易系统。
- 权重主要来自主观技术分析，不是从收益目标反推训练出来的。
- 回测样本太少，尤其 LAOWANG、STWG、FHKQ、YWCX。
- Relay 有机器学习，但超参数搜索和规则搜索太容易过拟合。
- 文档说“分时回封确认”，代码实际没有把分时特征接进接力评分。
- 没有实盘级卖出、仓位、滑点、成交失败、停牌、风控熔断闭环。

## 模型逐个审判

| 模型 | 当前用途 | 能不能上真钱 | 主要问题 | 处理决定 |
| --- | --- | --- | --- | --- |
| LAOWANG | 低位主升浪观察池 | 不能单独自动交易 | 信号少，权重主观，卖点缺失 | 保留为低频扫描器 |
| YWCX | 次新极弱修复观察池 | 不能 | 2025-2026 当前库无有效信号 | 暂停实盘化，先重建样本 |
| STWG | 缩量平台突破观察池 | 只能小仓验证 | 观察阶段和买入阶段容易混在一起 | 保留，但只交易 `BREAKOUT_R` |
| FHKQ | 连续跌停开板事件 | 不能常规交易 | 样本极少，尾部风险极大 | 只做事件监控，不做自动下单 |
| Relay | 次日接力候选 | 最值得改造 | 日线特征代替分时确认，过拟合风险高 | 作为主攻方向 |

## 回测证据

真实行情回测区间：2025-01-01 至 2026-04-30。  
数据终点早于用户请求的 2026-05-01，所以实际只能测到 2026-04-30。

| 模型 | 规则 | 交易数 | 收益 | 最大回撤 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LAOWANG | MA20 风格 | 2 | 6.01% | 6.60% | 100.00% |
| LAOWANG | MA5 破位 | 4 | 8.90% | 2.68% | 50.00% |
| STWG | MA10 风格 | 6 | 3.51% | 8.94% | 50.00% |
| STWG | MA5 破位 | 7 | 3.83% | 6.32% | 57.14% |
| Relay | 原接力 T+2/T+3 | 27 | 10.68% | 24.52% | 48.15% |
| Relay | MA5 破位 | 20 | 16.98% | 38.90% | 50.00% |
| 普通模型严格计划 | manual_open_v1 | 0 | 0.00% | 0.00% | 0.00% |

解读：

- LAOWANG 和 STWG 的收益看起来可以，但交易数太少，不足以证明稳定性。
- Relay 有收益，但回撤过大，说明它不是“稳”，而是“敢波动”。
- YWCX / FHKQ 当前几乎没有可用交易样本，不能拿来谈实盘。
- 任何少于 100 笔、没有严格样本外验证的结果，都不能当成稳定盈利证据。
- 按整改后的严格普通模型计划，2025-01-01 至 2026-04-30 没有一笔通过买入过滤；这不是坏事，说明实盘层开始拒绝低质量信号。
- Relay 严格按候选池处理后，116 条候选中 48 条进入 `watch_intraday`，仍需集合竞价和盘中分时确认。
- 风控切片里最刺眼的是 Relay：`relay_style` 最大单笔亏损约 -14.92%，`ma5_break` 最大单笔亏损约 -15.46%，连续亏损最多 4 笔；这不适合直接加仓，只适合先做降回撤改造。

回测文件：

- `reports/model_follow_backtest_20250101_20260501/report.md`
- `reports/model_follow_backtest_20250101_20260501/trades.csv`
- `reports/model_follow_backtest_20250101_20260501/next_day_plan.csv`
- `reports/model_follow_backtest_20250101_20260501/relay_watchlist.csv`
- `reports/model_follow_backtest_20250101_20260501/signal_audit.csv`
- `reports/model_follow_backtest_20250101_20260501/risk_summary.csv`
- `reports/model_follow_backtest_20250101_20260501/monthly_returns.csv`
- `backtest_model_follow.py`

## 数学底层问题

现在很多模型的形式是：

```text
score = sum(weight_i * feature_score_i)
```

这不是交易模型，只是主观看法的数字化。

真钱系统需要的是：

```text
action_t = f(signal_t, market_state_t, position_t, risk_budget_t)
```

并且目标函数必须写清楚：

```text
maximize  E[R_after_cost]
          - lambda * MaxDrawdown
          - gamma  * Turnover
          - eta    * TailLoss
```

约束：

```text
position <= position_cap
daily_loss <= daily_loss_cap
single_trade_loss <= trade_loss_cap
liquidity >= liquidity_floor
trade_allowed = not limit_up_unfilled and not suspended
```

如果一个模型只输出分数，不输出买入、卖出、仓位和风控，它就不是交易系统。

单笔交易期望值必须满足：

```text
edge = p_win * avg_win - (1 - p_win) * avg_loss - cost - slippage
```

只有 `edge > 0`，且在样本外还能成立，才有资格进小资金实盘。

## 当前最大的假繁荣

### 1. “模型评分”被误当成“交易指令”

LAOWANG、YWCX、STWG、FHKQ 都是评分/选池。  
它们没有统一卖出条件，也没有仓位条件。  
把它们隔日买入，本质是在给半成品硬装交易闭环。

### 2. 超短文档和代码断裂

`docs/daily_review_sop.md` 的超短逻辑强调：

- 竞价强弱
- 分时承接
- 回封速度
- 放量走弱
- 分歧转一致

但当前 `model_relay_pool` 的评分来自：

- `stock_daily`
- `stock_info`
- 涨停/炸板/市场情绪日线统计
- MLP 对日线特征的二分类

也就是说，文档里的核心交易瞬间没有进入模型。

### 3. Relay 的搜索容易过拟合

`mock_backtest.py` 同时搜索：

- MLP 结构
- 标签阈值
- alpha 映射
- 卖出规则
- top-k
- gap 过滤
- 连板过滤
- 风险过滤
- 仓位比例
- 分数阈值

这些组合在同一个回测区间里选最优，很容易把噪音当规律。  
这不是不能做，而是必须把“搜索区间”和“验收区间”彻底隔离。

### 4. 回测执行还不够实盘

必须继续补：

- 停牌不可交易。
- 涨停开盘买不到。
- 跌停卖不掉。
- 滑点随成交额和开盘冲击变化。
- 开盘集合竞价价格不等于可成交均价。
- 交易费用至少按 2 倍压力测试。
- 同日多信号冲突时必须统一账户排队。

## 怎么改成真正能上市场

### 阶段 1：先砍掉不该自动交易的东西

保留为观察池：

- LAOWANG
- STWG

暂停实盘化：

- YWCX
- FHKQ

主攻实盘化：

- Relay

原因很简单：Relay 有明确交易场景、样本较多、可以用机器学习做概率过滤；其它模型更像人工选股辅助。

### 阶段 2：每个模型只保留一个可执行交易协议

不要同时试十套卖法。

建议：

- LAOWANG：只做 MA20 / ATR 失守卖出，最大持有 20 日。
- STWG：只做突破确认，跌回 MA10 或突破日实体下沿卖出，最大持有 10 日。
- Relay：只做 T+1 开盘后确认，T+2/T+3 退出；盘中失败立即退出。
- FHKQ：只做事件小仓，买后第一个可卖日离场。

### 阶段 3：Relay 必须接入分时特征

下一版 `stock_minute` 至少要衍生这些字段：

- `auction_gap`：竞价相对昨日收盘涨跌幅。
- `auction_amount_ratio`：竞价成交额 / 近 5 日日均成交额。
- `first_15m_ret`：开盘 15 分钟涨跌幅。
- `first_15m_vwap_ret`：15 分钟 VWAP 相对开盘。
- `seal_time`：首次封板时间。
- `reopen_count`：开板次数。
- `re_seal_speed`：开板后回封耗时。
- `intraday_high_after_buy`：买入后最高可达收益。
- `intraday_stop_hit`：是否触发盘中止损。

如果没有这些，Relay 就不是超短模型，只是“昨日涨停票的日线二分类”。

### 阶段 4：训练方法改成 walk-forward

必须使用时间滚动：

```text
Train:      2020-01-01 ~ 2023-12-31
Validate:   2024-01-01 ~ 2024-12-31
Test:       2025-01-01 ~ 2025-06-30

Train:      2020-07-01 ~ 2024-06-30
Validate:   2024-07-01 ~ 2025-06-30
Test:       2025-07-01 ~ 2025-12-31

Train:      2021-01-01 ~ 2024-12-31
Validate:   2025-01-01 ~ 2025-12-31
Test:       2026-01-01 ~ 2026-04-30
```

验收看每个 test 段，而不是全区间最好收益。

### 阶段 5：上线门槛

没有达到这些指标，不上真钱：

- 样本外交易数 ≥ 100。
- 样本外总收益 > 0，且扣 2 倍手续费和滑点后仍 > 0。
- 最大回撤 ≤ 15%。
- 任一自然月亏损不超过 6%。
- 单笔最大亏损不超过 4%。
- 连续亏损 5 笔后自动降仓或停机。
- 每次模型改动后必须重新跑 walk-forward。

### 阶段 6：实盘分层

```text
0 级：只推送，不下单，记录人工是否采纳。
1 级：模拟盘，连续 60 个交易日。
2 级：真钱 10% 资金，单票 ≤ 2%，连续 60 个交易日。
3 级：真钱 30% 资金，单票 ≤ 5%，连续 120 个交易日。
4 级：扩大仓位，但保留日亏损熔断。
```

稳定盈利不是把模型调到收益最高，而是让系统在坏行情里少死。

## 性能与工程优先级

现在性能优化方向是对的：

- `stock_ingest_watermark` 减少大表扫描。
- upsert 分块写入。
- BaoStock 进程分片。
- UI 只读展示。

本轮已经补进 `init.py` 并在当前 MySQL 库验证存在：

```sql
CREATE INDEX idx_stock_daily_date_code
ON stock_daily(date, stock_code);

CREATE INDEX idx_scores_v3_date_score
ON stock_scores_v3(score_date, total_score);

CREATE INDEX idx_scores_stwg_date_score
ON stock_scores_stwg(score_date, total_score);

CREATE INDEX idx_scores_ywcx_date_score
ON stock_scores_ywcx(score_date, total_score);

CREATE INDEX idx_model_relay_date_rank_score
ON model_relay_pool(trade_date, rank_no, model_score);
```

同时补了两个工程表：

- `strategy_signal_log`：记录每个信号当时的全部特征和模型版本。
- `strategy_trade_journal`：记录模拟/实盘订单、成交、失败原因、滑点。

没有这两张表，复盘会变成口水仗。

## 合规提醒

如果后续接入券商自动下单，就不再只是研究工具。

中国市场已经对程序化交易实施报告、监测和管理要求。至少要关注：

- 中国证监会《证券市场程序化交易管理规定（试行）》。
- 上交所、深交所、北交所程序化交易管理实施细则。

尤其是自动申报、撤单频率、账户报告、异常交易监控等要求。  
本 repo 现在偏日线低频，风险较小；但一旦做超短自动下单，就必须按程序化交易系统管理。

参考：

- 证监会：《证券市场程序化交易管理规定（试行）》，2024-05-15 发布，2024-10-08 实施：https://www.csrc.gov.cn/csrc/c100028/c7480577/content.shtml
- 上交所：《上海证券交易所程序化交易管理实施细则》，2025-04-03 发布，2025-07-07 实施：https://big5.sse.com.cn/site/cht/www.sse.com.cn/lawandrules/sselawsrules/trade/universal/c/c_20250403_10776796.shtml
- 深交所：程序化交易管理实施细则相关新闻与配套规则说明，2025-04-03：https://investor.szse.cn/English/about/news/szse/t20250620_614312.html
- 北交所：《北京证券交易所程序化交易管理实施细则》，2025-04-03 发布，2025-07-07 实施：https://www.bse.cn/jygl_list/200025383.html

## 最终路线

真正值得做的是：

1. 把 LAOWANG / STWG 当观察池，不当自动策略。
2. 把 YWCX / FHKQ 从主线实盘计划里拿掉。
3. 把 Relay 改成“日线筛选 + 分时确认 + 风控执行”的完整系统。
4. 用 walk-forward 重训和验收。
5. 先跑 60 个交易日模拟盘。
6. 小资金上，失败就停，不许边亏边改参数。

最终产品形态应该是：

```text
每日盘后：
  生成观察池、情绪阶段、候选标的。

次日盘前：
  读取竞价特征，降低/删除弱票。

盘中：
  只在分歧转一致或回封确认时触发买点。

持仓后：
  自动止损、止盈、T+2/T+3 退出。

收盘后：
  记录每个信号是否执行、为什么没执行、执行后结果。
```

这样才有资格叫交易系统。  
现在最多叫研究驾驶舱。
