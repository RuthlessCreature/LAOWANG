# Four-Model Trading Protocols

The active system is a daily-line, manual-execution workflow. Model output is not an automatic buy order. Each row becomes a T+1 checklist and a T+2-or-later exit plan.

## LAOWANG

- Position: 5%.
- T+1 buy: open gap must be -3% to +3%; buy only after price holds above signal close or reclaims it intraday, with no risk tag and acceptable liquidity.
- T+2+ sell: exit on -5% hard stop, close below MA20, or after +12% profit arm when price closes below MA10; force exit by max hold.
- Max hold: 20 trading days.

## STWG

- Position: 5%.
- T+1 buy: open gap must be -3% to +4%; buy only on breakout continuation above signal close or region high with volume confirmation; no chase above +4%.
- T+2+ sell: exit on -4% hard stop, close back below MA10 or failed region high; after +10% profit arm trail with close below MA5; force exit by max hold.
- Max hold: 10 trading days.

## YWCX

- Position: 3%.
- T+1 buy: open gap must be -3% to +2%; buy only if price remains just above MA5 or reclaims signal close without liquidity collapse.
- T+2+ sell: exit on -4% hard stop, close below MA5, failed volume recovery, or after +8% profit target/reduction; force exit by max hold.
- Max hold: 5 trading days.

## FHKQ

- Position: 1.5%.
- T+1 buy: buy only if the stock is tradable, not locked limit-down, gap is -10% to +3%, and liquidity recovery is still visible; never chase event spikes.
- T+2+ sell: default event exit or reduction on first sellable day; exit immediately on -4% hard stop or liquidity recovery failure.
- Max hold: 2 trading days.

## Operator Ideas

- Keep the signal log as the single audit contract. Every manual trade should point back to a `strategy_signal_log` row.
- Score quality should be reviewed by realized journal outcome, not by visual excitement in the pool table.
- Minute-line data should stay out until a model has a concrete feature hypothesis and a before/after validation.
- Add a future `model_health_daily` table after several weeks of journal data: hit rate, average R, skipped-row follow-up, and per-model drawdown.
