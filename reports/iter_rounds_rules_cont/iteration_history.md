# Iteration History

- Stop rule: consecutive 3 rounds without indicator improvement (initial metric=ret_minus_dd, initial dd_penalty=2.00)
- Variant profile (initial): base
- Infinite adaptive mode: off
- Best round: 2

| Round | AdaptCycle | SeedOffset | Variant | RetainRatio | MinFloor | Metric | DDPenalty | AvgRet | AvgDD | Indicator | Improved | R@0.75 | DD@0.75 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | base | 0.80 | 4.00 | ret_minus_dd | 2.00 | 1265.68% | 34.35% | 1196.980000 | yes | 1265.68% | 34.35% |
| 2 | 0 | 97 | base | 0.85 | 4.00 | ret_minus_dd | 2.00 | 1328.77% | 39.84% | 1249.090000 | yes | 1328.77% | 39.84% |
| 3 | 0 | 194 | base | 0.90 | 4.00 | ret_minus_dd | 2.00 | 975.60% | 22.63% | 930.340000 | no | 975.60% | 22.63% |
