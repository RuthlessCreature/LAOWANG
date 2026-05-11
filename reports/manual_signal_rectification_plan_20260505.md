# 手动信号混合执行整改方案

日期：2026-05-05  
前提：不接入自动交易；普通模型按盘后信号生成 T+1 开盘人工交易计划；Relay 只生成次日接力候选，必须叠加集合竞价和盘中信号后人工交易。  
目标：把当前“评分/选池模型”整改为可以小资金实盘验证的半自动交易系统。

## 0. 先定交易口径

执行分两条线：

```text
普通模型（LAOWANG / STWG / YWCX / FHKQ）：
  T 日收盘后：生成信号和次日计划。
  T+1 开盘：人工按计划买入。
  T+1 当日：A 股 T+1 限制下，买入当天不能卖。
  T+2 起：按模型卖出规则处理。

Relay：
  T 日收盘后：生成接力候选池，不直接等于买入指令。
  T+1 集合竞价：根据竞价强弱、开盘 gap、竞价量能做二次过滤。
  T+1 盘中：结合日 K 背景和分时触发信号，人工决定是否买入。
  T+2/T+3：按接力卖出协议退出。
```

必须接受的代价：

- 普通模型可以按 T+1 开盘执行，但必须限制高开、低开和流动性。
- Relay 如果也按开盘无脑买，会损失超短交易的核心信息。
- Relay 的整改重点不是“开盘买”，而是“盘后候选 + 竞价过滤 + 盘中触发”。

## 1. 总体整改框架

当前系统缺的是四件事：

| 缺口 | 当前状态 | 整改目标 |
| --- | --- | --- |
| 入场 | 评分高就想买 | 普通模型：信号 + 开盘过滤；Relay：候选池 + 竞价过滤 + 盘中触发 |
| 卖出 | 没有统一规则 | 每个模型固定一套卖出协议 |
| 风控 | 回测里粗略处理 | 实盘前加入滑点、印花税、跌停卖不掉 |
| 复盘 | 只看结果 | 每个信号记录是否交易、为什么交易、为什么跳过 |

统一决策函数：

```text
trade_allowed =
  model_signal_ok
  and market_regime_ok
  and open_gap_ok
  and execution_trigger_ok
  and liquidity_ok
  and portfolio_risk_ok
```

统一收益判定：

```text
edge = win_rate * avg_win - (1 - win_rate) * avg_loss - fee - stamp_tax - slippage
```

只有样本外 `edge > 0` 才能加仓。

## 2. 交易成本必须补全

当前回测主要用了佣金，实盘整改必须加入：

- 买卖双边佣金。
- 卖出侧证券交易印花税。
- 过户费/经手费等实际券商成本。
- 开盘滑点。
- 涨停买不到、跌停卖不出。

印花税参考：财政部、税务总局公告 2023 年第 39 号，自 2023-08-28 起证券交易印花税减半征收。实盘参数应做成配置项，不要写死。

参考链接：https://www.chinatax.gov.cn/chinatax/n367/c5211207/content.html

## 3. 模型整改方案

### 3.1 LAOWANG

定位：中短线低位主升浪观察池。  
实盘角色：低频仓位，不能当超短。

当前问题：

- 2025-2026 区间信号很少，不能证明稳定性。
- 权重是主观技术分析加权，不是收益反推。
- 没有卖出协议。

整改后的入场条件：

```text
total_score >= 65
必须包含：TREND_UP、LOW_BASE、SPACE_OK
不能包含：RISK_FILTERED、NEAR_RESISTANCE
T+1 开盘涨幅 <= 3%
T+1 开盘跌幅 >= -3%
20 日均成交额 >= 1 亿
```

买入方式：

- 信号出现后，T+1 开盘买。
- 如果高开超过 3%，放弃，不追。
- 如果低开超过 3%，放弃，说明结构可能失效。

卖出规则：

```text
硬止损：买入价 -5%
结构止损：收盘跌破 MA20
时间止损：最多持有 20 个交易日
止盈：盈利 >= 12% 后，跌破 MA10 卖
```

仓位：

- 单票 5% 起步。
- 完成 30 笔样本后，如果样本外最大回撤 < 8%，可提高到 8%-10%。

训练整改：

- 不再只优化 `total_score`。
- 给每个历史信号打标签：
  - `ret_5d`
  - `ret_10d`
  - `ret_20d`
  - `max_drawdown_20d`
  - `hit_ma20_break`
- 用逻辑回归/LightGBM 做概率校准：

```text
P(ret_20d > 8% and max_drawdown_20d > -6%)
```

保留结论：可用，但只能做慢信号。

### 3.2 STWG

定位：缩量平台突破。  
实盘角色：突破交易，比 LAOWANG 更接近可交易。

当前问题：

- 文档说阶段 B 只是观察区，但实际评分容易把观察信号和买入信号混在一起。
- STWG 如果未突破就买，会变成“猜方向”。

整改后的入场条件：

```text
total_score >= 65
必须包含：STAGE_B_COMPRESSED
必须包含：BREAKOUT_R 或 VOLUME_EXPANSION
T+1 开盘价 >= 区域 R 上沿
T+1 开盘涨幅 <= 4%
20 日均成交额 >= 8000 万
```

明确禁止：

- 只有 `STAGE_B_COMPRESSED`，没有突破，不买。
- 高开超过 4%，不买。
- 跌回区域 R 内，不买。

卖出规则：

```text
硬止损：买入价 -4%
结构止损：收盘跌破 MA10
突破失败：收盘跌回区域 R 上沿下方
时间止损：最多持有 10 个交易日
止盈：盈利 >= 10% 后，跌破 MA5 卖
```

仓位：

- 单票 4%-6%。
- 同时最多 2 只 STWG。

训练整改：

给突破样本打标签：

```text
success = max_ret_5d >= 8% and max_drawdown_5d >= -4%
false_breakout = close_3d < region_high
```

重点训练“假突破识别”，而不是继续给压缩形态打高分。

保留结论：可用，但必须只买突破确认，不买观察状态。

### 3.3 YWCX

定位：极弱次新修复。  
实盘角色：暂时只观察，不实盘。

当前问题：

- 当前库 2025-2026 没有有效信号。
- 次新模型依赖上市天数、发行价、流通市值，这些字段如果不完整，模型会直接失效。
- 它是低流动性、小票、弱修复模型，实盘滑点会吞收益。

整改条件：

先不谈交易，先补数据和样本：

```text
补齐 issue_price
补齐 listing_date / listing_days
补齐 float_cap_billion
回测 2020-2026 全样本
把上市窗口从 60 日扩到 120 日做对照实验
```

等样本恢复后，入场条件：

```text
total_score >= 65
必须包含：BROKEN_IPO、NEAR_IPO_LOW、VOLUME_DRY、JUST_ABOVE_MA5
T+1 开盘涨幅 <= 2%
20 日均成交额 >= 5000 万
```

卖出规则：

```text
硬止损：买入价 -4%
结构止损：收盘跌破 MA5
时间止损：最多持有 5 个交易日
止盈：盈利 >= 8% 后次日开盘或收盘卖
```

保留结论：现在不能实盘，先修数据。

### 3.4 FHKQ

定位：连续跌停开板事件。  
实盘角色：事件监控，不做常规策略。

当前问题：

- 样本极少。
- 尾部风险极大。
- 隔日开盘直接买不适合 FHKQ，因为它最关键的是流动性恢复瞬间，而不是简单次日开盘。

整改后的交易条件：

```text
fhkq_score >= 80
consecutive_limit_down in [2, 4]
open_board_flag = 1
liquidity_exhaust = 1
T+1 不是跌停开盘
T+1 开盘涨幅 <= 3%
```

卖出规则：

```text
第一个可卖日无条件卖出一半
盈利 >= 6% 全卖
亏损 <= -4% 全卖
若再次跌停，按能卖出的最早时点卖
最多持有 2 个交易日
```

仓位：

- 单票不超过 1%-2%。
- 未满 50 笔事件样本前，不允许扩大。

保留结论：不是稳定盈利主线，只能做小仓事件票。

### 3.5 Relay

定位：次日接力。  
实盘角色：唯一主攻方向，但不是 T+1 开盘无脑买模型。

当前问题：

- Relay 的本质是情绪接力，不是普通隔日买入。
- 文档里的核心是“竞价不弱 + 分时回封确认 + 分歧转一致”。
- 当前 `model_relay_pool` 主要来自日线涨停结构和市场情绪，没有真正吃进集合竞价和盘中分时。
- 现有回测收益有，但最大回撤太大，说明它需要执行层信号来降噪。

整改后的四段执行流：

```text
第一段：T 日盘后生成候选
  只回答“明天哪些票值得盯”。

第二段：T+1 集合竞价过滤
  只回答“开盘有没有资格继续盯”。

第三段：T+1 盘中触发
  只回答“此刻能不能人工买”。

第四段：T+2/T+3 退出
  只回答“什么时候必须走”。
```

T 日盘后候选条件：

```text
model_score >= 0.70
rank_no <= 1
max_board <= 3
broken_rate <= 0.40
red_rate >= 0.28
limit_down_count <= 15
pullback <= 0.08
20 日均成交额 >= 1.5 亿
```

T+1 集合竞价过滤：

```text
auction_gap >= -1%
auction_gap <= 4%
不是涨停价一字开
不是跌停价附近开
auction_amount_ratio >= 0.08
竞价最后 3 分钟价格不明显下坠
竞价成交额不能是异常虚胖后撤单
```

说明：

- `auction_gap` 是集合竞价最终价相对 T 日收盘价的涨跌幅。
- `auction_amount_ratio` 是集合竞价成交额 / 近 5 日日均成交额。
- 如果暂时没有逐笔竞价数据，先用 9:25 开盘价和首根分钟量做近似，但报告里必须标记为近似。

T+1 盘中触发条件：

```text
日 K 背景仍有效：
  T 日为非一字涨停或高质量换手板
  市场情绪不是冰点
  所属题材或梯队未明显断层

盘中触发至少满足一个：
  开盘后 15 分钟不破 T 日收盘价太深
  回踩不破分时均价线后重新上穿
  开板后能快速回封
  高开回落后重新突破开盘价
  量能放大但价格不崩
```

不买条件：

```text
开盘直接缩量冲高后回落
开盘 15 分钟跌幅 <= -3%
跌破分时均价线后无法收回
板块梯队当天明显退潮
同批候选已有更强标的
```

为什么盘后只做 top1/top2：

- Relay 噪音大，盘后候选越多，人工越容易被诱惑带偏。
- 如果有盘中触发，候选可以 top2；如果只按竞价近似，最多 top1。
- Relay 的噪音大，top2/top3 会明显增加尾部风险。

卖出规则：

```text
买入日不能卖。
T+2 集合竞价弱于预期且亏损 <= -3%，开盘卖。
T+2 盘中跌破 T+1 买入逻辑的关键支撑，卖。
T+2 收盘仍未转强，收盘卖。
T+2 收盘盈利 >= 5%，且次日仍有情绪延续，可延到 T+3。
T+3 无论盈亏，原则上清仓。
```

强制降仓条件：

```text
连续亏损 3 笔：下一笔仓位减半。
连续亏损 5 笔：暂停 Relay 5 个交易日。
单日账户亏损 >= 3%：当天不再开新仓。
当月亏损 >= 6%：停止当月 Relay。
```

训练整改：

当前标签主要是 T+1 open 到 T+2 close 的收益。下一版要拆成三类标签：

```text
candidate_label:
  T 日盘后候选是否值得进入次日观察池。

auction_label:
  T+1 集合竞价后是否仍具备参与价值。

intraday_trigger_label:
  T+1 盘中触发后，到 T+2/T+3 是否有正期望。
```

最终训练目标：

```text
target = 1 if (
  ret_exit >= 0.03
  and max_drawdown_from_entry >= -0.035
  and trade_is_executable
  and trigger_quality_ok
) else 0
```

Relay 需要新增特征：

```text
auction_gap
auction_amount_ratio
auction_price_slope
first_5m_ret
first_15m_ret
first_15m_vwap_ret
first_30m_low_drawdown
reclaim_open_price_flag
reclaim_vwap_flag
re_seal_flag
re_seal_minutes
intraday_break_vwap_count
theme_strength_today
ladder_intact_today
```

并做 walk-forward：

```text
训练：过去 3-4 年
验证：随后 6-12 个月
测试：之后 3-6 个月
```

模型不看全区间收益，只看每个样本外区间是否稳定。

保留结论：能改成实盘候选，但必须先降回撤。

## 4. 组合仓位规则

在没有 100 笔样本外交易前，仓位必须保守。

建议账户规则：

| 策略 | 单票仓位 | 最大策略仓位 | 备注 |
| --- | ---: | ---: | --- |
| LAOWANG | 5% | 20% | 慢信号 |
| STWG | 4%-6% | 20% | 只买突破 |
| Relay | 3%-5% | 15% | 盘后 top1/top2，竞价和盘中触发后才买 |
| FHKQ | 1%-2% | 3% | 事件票 |
| YWCX | 0% | 0% | 暂停 |

总规则：

```text
任意一天最多新增 1 笔。
总持仓不超过 50%。
连续 5 笔亏损后，全部策略暂停 5 个交易日。
账户回撤超过 8%，所有单票仓位减半。
账户回撤超过 12%，暂停实盘，只做模拟。
```

## 5. 每日人工执行清单

普通模型盘后生成计划表：

| 字段 | 说明 |
| --- | --- |
| trade_date | 信号日 |
| buy_date | 计划买入日 |
| model | 模型 |
| stock_code | 股票代码 |
| stock_name | 名称 |
| score | 模型分数 |
| entry_filter | T+1 开盘过滤 |
| planned_position | 计划仓位 |
| hard_stop | 硬止损 |
| structure_stop | 结构止损 |
| max_hold_days | 最大持有 |
| skip_reason | 如果跳过，记录原因 |

普通模型 T+1 开盘只做三件事：

```text
1. 看是否涨停买不到。
2. 看开盘涨跌幅是否超出模型范围。
3. 看总仓位和连续亏损规则是否允许。
```

满足才买，不满足直接跳过。

Relay 单独生成接力执行表：

| 字段 | 说明 |
| --- | --- |
| signal_date | 候选信号日 |
| watch_date | 次日观察日 |
| stock_code | 股票代码 |
| stock_name | 名称 |
| model_score | 盘后接力分 |
| rank_no | 排名 |
| daily_context | 日 K 和情绪背景 |
| auction_filter | 集合竞价过滤结果 |
| intraday_trigger | 盘中触发条件 |
| planned_position | 计划仓位 |
| invalidation_rule | 触发失效条件 |
| exit_protocol | T+2/T+3 卖出协议 |

Relay T+1 执行动作：

```text
9:25：
  更新 auction_filter。
  不通过则删除，不进入盘中盯盘。

9:30-10:00：
  只观察，不急买。
  等是否守住关键价、回收均价线、重新突破开盘价或回封。

10:00 以后：
  只有触发 intraday_trigger 才允许人工买。
  未触发则当天放弃。
```

## 6. 回测系统整改任务

本轮已经落地：

1. `backtest_model_follow.py` 已输出 `next_day_plan.csv`、`relay_watchlist.csv`、`signal_audit.csv`。
2. `manual_open_v1` 已接入普通模型严格计划回测：只在计划表 `planned_action=buy` 时按 T+1 开盘买入，并按模型卖出规则退出。
3. 已输出 `risk_summary.csv` 和 `monthly_returns.csv`，用于检查最大单笔亏损、连续亏损和分月收益。
4. `init.py` 已补 `strategy_signal_log`、`strategy_trade_journal` 和常用日期/分数索引。

还没完成、但必须做的：

1. Relay 新增真正的 `relay_hybrid_v1` 回测模式，用于盘后候选 + 集合竞价过滤 + 盘中触发。
2. 加入普通模型更多可调开盘过滤：
   - `--max-open-gap`
   - `--min-open-gap`
   - `--block-limit-up-open`
3. 加入 Relay 执行特征：
   - auction gap
   - auction amount ratio
   - first 5/15/30 minute return
   - VWAP reclaim
   - re-seal flag
   - intraday invalidation flag
4. 把 `strategy_signal_log`、`strategy_trade_journal` 接入每日流程，而不是只建表。

已补进 `init.py` 的两张表结构：

```sql
CREATE TABLE strategy_signal_log (
  signal_date VARCHAR(10) NOT NULL,
  model VARCHAR(32) NOT NULL,
  stock_code VARCHAR(16) NOT NULL,
  stock_name VARCHAR(255) NULL,
  score DOUBLE NULL,
  features_json TEXT NULL,
  model_version VARCHAR(64) NULL,
  action_plan_json TEXT NULL,
  created_at VARCHAR(19) NULL,
  PRIMARY KEY (signal_date, model, stock_code)
);

CREATE TABLE strategy_trade_journal (
  trade_id VARCHAR(64) PRIMARY KEY,
  signal_date VARCHAR(10) NOT NULL,
  model VARCHAR(32) NOT NULL,
  stock_code VARCHAR(16) NOT NULL,
  buy_date VARCHAR(10) NULL,
  buy_price DOUBLE NULL,
  planned_position DOUBLE NULL,
  sell_date VARCHAR(10) NULL,
  sell_price DOUBLE NULL,
  pnl DOUBLE NULL,
  pnl_pct DOUBLE NULL,
  exit_reason VARCHAR(255) NULL,
  manual_notes TEXT NULL,
  created_at VARCHAR(19) NULL,
  updated_at VARCHAR(19) NULL
);
```

## 7. 实盘前验收门槛

任何模型要进入小资金实盘，必须满足：

```text
样本外交易数 >= 100
扣 2 倍交易成本后仍盈利
最大回撤 <= 12%
最大单笔亏损 <= 5%
月度亏损 <= 6%
连续亏损 <= 5 笔
收益不是由单笔极端大赚贡献超过 40%
```

如果达不到，只能当观察池。

## 8. 30 天整改路线

第 1 周：

- 固化五个模型的入场、卖出、仓位规则。
- 修改普通模型回测，加入成本、滑点、开盘过滤。
- 给 Relay 定义集合竞价和盘中触发字段。
- 输出 `next_day_plan.csv`、`relay_watchlist.csv` 和 `signal_audit.csv`。

第 2 周：

- 做 2020-2026 全区间 walk-forward。
- 每个模型至少输出分年、分月、分市场阶段表现。
- 淘汰样本不足或样本外失效的配置。

第 3 周：

- 对 Relay 做降回撤训练。
- 加入更严格的竞价、盘中触发、market regime、top1/top2 约束。
- 生成手动交易看板。

第 4 周：

- 只做模拟盘。
- 每天记录：
  - 系统信号
  - 是否执行
  - 跳过原因
  - 实际开盘价
  - 实际卖出原因
  - 人工干预原因

模拟 60 个交易日后，再决定是否小资金。

## 9. 最终判断

可以实盘化，但不是把现在的模型直接拿去买。

正确路径是：

```text
LAOWANG / STWG：观察池 + 低频小仓。
Relay：主攻，但必须走盘后候选 + 集合竞价过滤 + 盘中触发。
YWCX：修数据前不交易。
FHKQ：事件监控，小仓，不当主策略。
```

如果普通模型坚持“信号出来，隔日开盘直接买”，那系统必须变得更保守：

- 少交易。
- 只买高质量信号。
- 严格限制开盘高开。
- 严格限制单票仓位。
- 卖出规则写死。
- 每笔都记录，不许凭感觉改口径。

Relay 不适用这条。Relay 的信号只是次日盯盘名单，真正买点来自集合竞价和盘中触发。

这套整改完成后，它才从“看着挺有道理”变成“可以拿小钱验证”。
