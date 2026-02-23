# Mock Backtest Report (threshold 0.75)

- Range: 2026-01-01 ~ 2026-02-13
- Initial capital: 10000.00
- Final strategy: model=mlp_h64_e360_gt2 (hidden=64, epochs=360, lr=0.028, target=T+1 open -> T+2 close > 2.00%)
- Score mapping: score = 1 - (1 - CDF(raw_prob))^2
- Sell rule: strong_hold_t3
- Filters: TopK=2, gap=[-2.00%, 4.00%], max_board=2, risk=balanced, broken<=45.00%, red>=24.00%, limit_down<=20, pullback<=9.00%, capital_use=100%
- Model train acc: 65.51%

## Rules (rules.txt)
- Rules file: rules.txt
- Buy: T signal then T+1 open (current run uses T+1)
- Limit-up open unfilled: yes
- Fee: 0.0250% per side, min 5.00 CNY/order
- Slippage: ignored
- Execution: buy at T+1 open, sell at rule close, no intraday-high optimization
- Selection policy: keep candidates above max(80% of best return, 400.00%), then minimize drawdown

## Search Top 8 (current threshold)
- 1. mlp_h56_e320_gt1, alpha=15, rule=loss_rebound_t3, topk=3, gap=[None,None], board<=3, risk=off, alloc=100%, train_acc=60.45%, R@0.75=78.26%, DD@0.75=4.58%, trades=9, minR=78.26%
- 2. mlp_h72_e380_gt15, alpha=15, rule=t2_close, topk=2, gap=[None,None], board<=None, risk=off, alloc=100%, train_acc=63.28%, R@0.75=77.31%, DD@0.75=13.80%, trades=9, minR=77.31%
- 3. mlp_h72_e380_gt15, alpha=15, rule=t2_close, topk=2, gap=[None,None], board<=3, risk=off, alloc=100%, train_acc=63.28%, R@0.75=77.31%, DD@0.75=13.80%, trades=9, minR=77.31%
- 4. mlp_h72_e380_gt15, alpha=15, rule=t2_close, topk=3, gap=[None,None], board<=None, risk=off, alloc=100%, train_acc=63.28%, R@0.75=77.31%, DD@0.75=13.80%, trades=9, minR=77.31%
- 5. mlp_h72_e380_gt15, alpha=15, rule=t2_close, topk=3, gap=[None,None], board<=3, risk=off, alloc=100%, train_acc=63.28%, R@0.75=77.31%, DD@0.75=13.80%, trades=9, minR=77.31%
- 6. mlp_h56_e320_gt1, alpha=15, rule=loss_rebound_t3, topk=3, gap=[None,None], board<=3, risk=off, alloc=90%, train_acc=60.45%, R@0.75=72.74%, DD@0.75=4.02%, trades=9, minR=72.74%
- 7. mlp_h64_e360_gt2, alpha=2, rule=strong_hold_t3, topk=2, gap=[None,None], board<=2, risk=balanced, alloc=100%, train_acc=65.51%, R@0.75=65.12%, DD@0.75=5.88%, trades=6, minR=65.12%
- 8. mlp_h64_e360_gt2, alpha=2, rule=strong_hold_t3, topk=3, gap=[None,None], board<=2, risk=balanced, alloc=100%, train_acc=65.51%, R@0.75=65.12%, DD@0.75=5.88%, trades=6, minR=65.12%

## Summary
- Final capital: 13112.00
- Total return: 31.12%
- Total fees: 40.00
- Trades: 4
- Win rate: 100.00% (4W/0L)
- Avg trade return: 7.60%
- Max drawdown: 0.00%

## Skip stats
- no_candidate: 0
- threshold_blocked: 0
- board_blocked: 11
- risk_blocked: 3
- gap_blocked: 3
- rule_blocked: 0
- bad_buy_quote: 0
- calendar_miss: 0
- insufficient_cash: 0
- bad_exit: 0

## Trades

### 1. 600391 航发科技
- Signal date: 2026-01-20
- Pick reason: Top2 pick rank=1; score=99.44%; raw_prob=44.53%; board=2; ret1=10.00%.
- Buy:
  - Time: 2026-01-21 09:30:00
  - Price: 46.000
  - Shares: 200
  - Amount: 9200.00
  - Fee: 5.00
  - Reason: model=mlp_h64_e360_gt2, alpha=2; score>=threshold; gap_filter=[-2.00%, 4.00%], max_board=2; risk_profile=balanced (broken<=45.00%, red>=24.00%, limit_down<=20, pullback<=9.00%); capital_use=100%; T signal -> T+1 open buy, limit-up open is unfilled.
- Sell:
  - Time: 2026-01-23 15:00:00
  - Price: 53.650
  - Amount: 10730.00
  - Fee: 5.00
  - Reason: T+2收盘上涨16.00%>=6%，延长到T+3收盘卖出 (no intraday-high optimization).
- Result:
  - Total fee: 10.00
  - PnL: 1520.00
  - Return: 16.51%
  - Equity after trade: 11520.00

### 2. 603132 金徽股份
- Signal date: 2026-01-27
- Pick reason: Top2 pick rank=2; score=99.89%; raw_prob=48.61%; board=2; ret1=9.98%.
- Buy:
  - Time: 2026-01-28 09:30:00
  - Price: 20.500
  - Shares: 500
  - Amount: 10250.00
  - Fee: 5.00
  - Reason: model=mlp_h64_e360_gt2, alpha=2; score>=threshold; gap_filter=[-2.00%, 4.00%], max_board=2; risk_profile=balanced (broken<=45.00%, red>=24.00%, limit_down<=20, pullback<=9.00%); capital_use=100%; T signal -> T+1 open buy, limit-up open is unfilled.
- Sell:
  - Time: 2026-01-29 15:00:00
  - Price: 21.600
  - Amount: 10800.00
  - Fee: 5.00
  - Reason: T+2收盘上涨5.37%未触发延长，T+2收盘卖出 (no intraday-high optimization).
- Result:
  - Total fee: 10.00
  - PnL: 540.00
  - Return: 5.27%
  - Equity after trade: 12060.00

### 3. 000056 皇庭国际
- Signal date: 2026-02-06
- Pick reason: Top2 pick rank=2; score=98.52%; raw_prob=41.42%; board=2; ret1=10.14%.
- Buy:
  - Time: 2026-02-09 09:30:00
  - Price: 2.310
  - Shares: 5200
  - Amount: 12012.00
  - Fee: 5.00
  - Reason: model=mlp_h64_e360_gt2, alpha=2; score>=threshold; gap_filter=[-2.00%, 4.00%], max_board=2; risk_profile=balanced (broken<=45.00%, red>=24.00%, limit_down<=20, pullback<=9.00%); capital_use=100%; T signal -> T+1 open buy, limit-up open is unfilled.
- Sell:
  - Time: 2026-02-10 15:00:00
  - Price: 2.410
  - Amount: 12532.00
  - Fee: 5.00
  - Reason: T+2收盘上涨4.33%未触发延长，T+2收盘卖出 (no intraday-high optimization).
- Result:
  - Total fee: 10.00
  - PnL: 510.00
  - Return: 4.24%
  - Equity after trade: 12570.00

### 4. 603968 醋化股份
- Signal date: 2026-02-11
- Pick reason: Top2 pick rank=2; score=99.39%; raw_prob=44.11%; board=2; ret1=10.04%.
- Buy:
  - Time: 2026-02-12 09:30:00
  - Price: 15.480
  - Shares: 800
  - Amount: 12384.00
  - Fee: 5.00
  - Reason: model=mlp_h64_e360_gt2, alpha=2; score>=threshold; gap_filter=[-2.00%, 4.00%], max_board=2; risk_profile=balanced (broken<=45.00%, red>=24.00%, limit_down<=20, pullback<=9.00%); capital_use=100%; T signal -> T+1 open buy, limit-up open is unfilled.
- Sell:
  - Time: 2026-02-13 15:00:00
  - Price: 16.170
  - Amount: 12936.00
  - Fee: 5.00
  - Reason: T+2收盘上涨4.46%未触发延长，T+2收盘卖出 (no intraday-high optimization).
- Result:
  - Total fee: 10.00
  - PnL: 542.00
  - Return: 4.37%
  - Equity after trade: 13112.00
