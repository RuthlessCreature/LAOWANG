# Relay Hybrid 参数扫描

- 区间：2025-01-01 ~ 2026-04-30
- 数据模式：daily_proxy
- 组合数：72

## 最优候选

- plan_threshold_relay: 0.85
- plan_relay_top_k: 2
- relay_trigger_ret_pct: 0.0
- relay_max_entry_drawdown_pct: -0.04
- trades: 24
- return_pct: 2.5328974482673683
- max_dd_pct: 0.6393401414342218
- win_rate: 58.333333333333336
- max_single_loss_pct: -6.949077042129616
- max_consecutive_losses: 6
- objective: -3.1633820244163684

## 说明

- 这是样本内参数扫描，用于找整改方向，不是上线结论。
- 当前无 `stock_minute` 时，所有盘中触发都是日线 OHLC 代理。
- 真正上线必须用 walk-forward 和真实分钟线重新验收。