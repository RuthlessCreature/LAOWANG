# Updates

## 2026-05-13

- Simplified the active workflow to four ordinary numeric models: LAOWANG, STWG, YWCX, and FHKQ.
- Moved inactive experimental scripts, old UI, old review docs, and inactive model artifacts into `legacy/`; active code does not import from that directory.
- Added explicit T+1 buy conditions and T+2-or-later sell conditions to `strategy_protocols.py` and `generate_trade_plan.py`.
- Replaced the Web UI with a smaller read-only four-model dashboard.
- Capped BaoStock process concurrency at 2 and batched daily database upserts to reduce write overhead.
- Rewrote README and SOP/database docs around the simplified daily-line workflow.
- Tightened `.gitignore` for local secrets, databases, generated reports, runtime caches, and app state.
