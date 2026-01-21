Param(
  [string]$Config = "config.ini"
)

$ErrorActionPreference = "Stop"

$code = @'
from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import text

from a_stock_analyzer.config import resolve_db_target
from a_stock_analyzer.db import make_engine


def mask_db_url(s: str) -> str:
    s = str(s or "")
    if "://" not in s:
        return s
    try:
        head, rest = s.split("://", 1)
        if "@" not in rest:
            return s
        auth, tail = rest.split("@", 1)
        if ":" in auth:
            user = auth.split(":", 1)[0]
            return f"{head}://{user}:***@{tail}"
        return f"{head}://{auth}@{tail}"
    except Exception:
        return s


cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.ini")
db_target = resolve_db_target(db_url_arg=None, db_arg=None, config_path=cfg)
engine = make_engine(db_target, pool_size=2, max_overflow=2)

print("[db] target =", mask_db_url(db_target))
print("[db] dialect =", engine.dialect.name)

if engine.dialect.name == "mysql":
    sql = "SELECT TABLE_NAME FROM information_schema.tables WHERE table_schema = database() ORDER BY TABLE_NAME"
else:
    sql = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"

with engine.connect() as conn:
    rows = conn.execute(text(sql)).fetchall()
tables = [str(r[0]) for r in rows if r and r[0]]
print("[db] tables =", len(tables))

core = [
    "stock_info",
    "stock_daily",
    "stock_indicators",
    "stock_levels",
    "stock_scores_v3",
    "model_runs",
    "model_laowang_pool",
    "model_fhkq",
]
missing = [t for t in core if t not in set(tables)]
if missing:
    print("[db] missing core tables:", ", ".join(missing))
else:
    print("[db] core tables: OK")
'@

python -c $code $Config

