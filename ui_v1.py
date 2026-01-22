#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui.py

简化版本地 Web UI：
- 仅从数据库读取 model_laowang_pool / model_fhkq
- 不再触发任何计算任务
- LAOWANG 表展示 status_tags 徽章
"""

from __future__ import annotations

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

import argparse
import datetime as dt
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import everyday


DEFAULT_DB = "data/stock.db"


@dataclass
class MySQLConfig:
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = ""
    password: str = ""
    database: str = ""
    charset: str = "utf8mb4"


@dataclass
class AppConfig:
    db_url: Optional[str] = None
    mysql: MySQLConfig = field(default_factory=MySQLConfig)


def load_config(path: Path) -> AppConfig:
    import configparser

    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    db_url = parser.get("database", "db_url", fallback=None)
    db_url = db_url.strip() if db_url else None
    mysql = MySQLConfig(
        host=parser.get("mysql", "host", fallback="127.0.0.1").strip() or "127.0.0.1",
        port=parser.getint("mysql", "port", fallback=3306),
        user=parser.get("mysql", "user", fallback="").strip(),
        password=parser.get("mysql", "password", fallback=""),
        database=parser.get("mysql", "database", fallback="").strip(),
        charset=parser.get("mysql", "charset", fallback="utf8mb4").strip() or "utf8mb4",
    )
    return AppConfig(db_url=db_url, mysql=mysql)


def build_mysql_url(cfg: MySQLConfig) -> Optional[str]:
    if not (cfg.user and cfg.database):
        return None
    from urllib.parse import quote_plus

    user = quote_plus(cfg.user)
    password = quote_plus(cfg.password or "")
    auth = f"{user}:{password}" if password else user
    return f"mysql+pymysql://{auth}@{cfg.host}:{int(cfg.port)}/{cfg.database}?charset={cfg.charset}"


def resolve_db_target(args: argparse.Namespace) -> str:
    if getattr(args, "db_url", None):
        return str(args.db_url)
    import os

    env = os.getenv("ASTOCK_DB_URL")
    if env and env.strip():
        return env.strip()
    if getattr(args, "db", None):
        return str(args.db)
    cfg_path = getattr(args, "config", None)
    cfg_file = Path(cfg_path) if cfg_path else Path("config.ini")
    if cfg_file.exists():
        cfg = load_config(cfg_file)
        if cfg.db_url:
            return cfg.db_url
        url = build_mysql_url(cfg.mysql)
        if url:
            return url
    return DEFAULT_DB


def _normalize_iso_date(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return dt.datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")


def make_engine(db_target: str) -> Engine:
    connect_args = {}
    if "://" not in db_target and db_target.endswith(".db"):
        db_path = Path(db_target).expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_target = f"sqlite:///{db_path.as_posix()}"
    if db_target.startswith("sqlite:///"):
        connect_args["check_same_thread"] = False
    engine = create_engine(db_target, pool_pre_ping=True, pool_recycle=3600, connect_args=connect_args)
    if engine.dialect.name == "sqlite":
        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys = ON")
            cur.execute("PRAGMA journal_mode = WAL")
            cur.close()
    return engine


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>LAOWANG / FHKQ</title>
    <style>
      :root {
        --bg: #070a0f;
        --panel: #0b0f17;
        --text: #dbe7ff;
        --muted: #8aa0c7;
        --line: rgba(0, 229, 255, 0.25);
        --accent: #00e5ff;
        --warn: #ffcc66;
        --err: #ff5577;
        --ok: #33ffa6;
        --shadow: rgba(0, 0, 0, 0.45);
        --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans", "Liberation Sans", sans-serif;
      }
      html, body { height: 100%; }
      body {
        margin: 0;
        background:
          radial-gradient(1200px 800px at 20% 10%, rgba(0,229,255,0.10), transparent 60%),
          radial-gradient(900px 700px at 80% 15%, rgba(124,92,255,0.10), transparent 55%),
          linear-gradient(180deg, #05070b 0%, #070a0f 35%, #070a0f 100%);
        color: var(--text);
        font-family: var(--sans);
      }
      .wrap { max-width: 1320px; margin: 0 auto; padding: 18px 18px 28px; }
      .topbar {
        display: flex; gap: 14px; align-items: center; justify-content: space-between;
        padding: 14px 16px;
        background: linear-gradient(180deg, rgba(11,15,23,0.95), rgba(11,15,23,0.75));
        border: 1px solid var(--line);
        box-shadow: 0 10px 30px var(--shadow);
        border-radius: 14px;
        position: sticky; top: 10px; z-index: 10;
        backdrop-filter: blur(8px);
      }
      .controls { display: flex; gap: 10px; align-items: center; }
      select {
        background: rgba(10,14,22,0.95);
        border: 1px solid var(--line);
        color: var(--text);
        border-radius: 10px;
        padding: 8px 10px;
        font-family: var(--mono);
      }
      .status {
        display: flex; gap: 10px; align-items: center;
        font-family: var(--mono);
        font-size: 12px;
        color: var(--muted);
      }
      .dot { width: 8px; height: 8px; border-radius: 50%; background: rgba(255,255,255,0.15); }
      .dot.ok { background: var(--ok); box-shadow: 0 0 10px rgba(51,255,166,0.55); }
      .dot.warn { background: var(--warn); box-shadow: 0 0 10px rgba(255,204,102,0.55); }
      .dot.err { background: var(--err); box-shadow: 0 0 10px rgba(255,85,119,0.55); }
      .panel {
        margin-top: 18px;
        background: linear-gradient(180deg, rgba(11,15,23,0.85), rgba(11,15,23,0.55));
        border: 1px solid var(--line);
        border-radius: 14px;
        box-shadow: 0 10px 30px var(--shadow);
        overflow: hidden;
      }
      .panel-header {
        padding: 12px 14px;
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid rgba(0,229,255,0.15);
        font-family: var(--mono);
        font-size: 12px;
      }
      table { width: 100%; border-collapse: collapse; }
      thead th {
        background: rgba(7,10,15,0.90);
        color: rgba(219,231,255,0.95);
        font-family: var(--mono);
        font-size: 12px;
        padding: 10px 10px;
        border-bottom: 1px solid rgba(0,229,255,0.22);
        text-align: left;
        position: sticky; top: 0;
      }
      tbody td {
        padding: 9px 10px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-family: var(--mono);
        font-size: 12px;
        color: rgba(219,231,255,0.92);
        white-space: nowrap;
      }
      .tag-wrap { display: flex; flex-wrap: wrap; gap: 4px; }
      .tag-pill {
        border: 1px solid rgba(0,229,255,0.25);
        border-radius: 999px;
        padding: 2px 6px;
        font-size: 11px;
        color: rgba(219,231,255,0.92);
      }
      .empty {
        padding: 18px;
        color: var(--muted);
        font-family: var(--mono);
        font-size: 12px;
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="topbar">
        <div class="controls">
          <label for="tradeDate">交易日</label>
          <select id="tradeDate"></select>
        </div>
        <div class="status">
          <span class="dot" id="statusDot"></span>
          <span id="statusText">loading</span>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">
          <div>model_laowang_pool</div>
          <div id="metaLaowang"></div>
        </div>
        <div class="table-wrap">
          <table id="tableLaowang">
            <thead></thead>
            <tbody></tbody>
          </table>
          <div class="empty" id="emptyLaowang" style="display:none;"></div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">
          <div>model_fhkq</div>
          <div id="metaFhkq"></div>
        </div>
        <div class="table-wrap">
          <table id="tableFhkq">
            <thead></thead>
            <tbody></tbody>
          </table>
          <div class="empty" id="emptyFhkq" style="display:none;"></div>
        </div>
      </div>
    </div>

    <script>
      const $ = (id) => document.getElementById(id);

      function setStatus(kind, text) {
        const dot = $("statusDot");
        dot.className = "dot";
        if (kind) dot.classList.add(kind);
        $("statusText").textContent = text || "";
      }

      function renderTable(tableId, emptyId, payload) {
        const table = $(tableId);
        const thead = table.querySelector("thead");
        const tbody = table.querySelector("tbody");
        const empty = $(emptyId);
        const cols = payload.columns || [];
        const rows = payload.rows || [];
        thead.innerHTML = "";
        tbody.innerHTML = "";
        if (!cols.length) {
          empty.style.display = "";
          empty.textContent = payload.meta && payload.meta.empty_hint ? payload.meta.empty_hint : "no data";
          return;
        }
        const trh = document.createElement("tr");
        cols.forEach(c => {
          const th = document.createElement("th");
          th.textContent = c;
          trh.appendChild(th);
        });
        thead.appendChild(trh);
        if (!rows.length) {
          empty.style.display = "";
          empty.textContent = payload.meta && payload.meta.empty_hint ? payload.meta.empty_hint : "0 rows";
          return;
        }
        empty.style.display = "none";
        rows.forEach(row => {
          const tr = document.createElement("tr");
          cols.forEach(col => {
            const td = document.createElement("td");
            const value = row[col];
            if (Array.isArray(value)) {
              const wrap = document.createElement("div");
              wrap.className = "tag-wrap";
              value.forEach(tag => {
                const pill = document.createElement("span");
                pill.className = "tag-pill";
                pill.textContent = tag;
                wrap.appendChild(pill);
              });
              td.appendChild(wrap);
            } else {
              td.textContent = value === null || value === undefined ? "" : String(value);
            }
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
      }

      async function apiGet(path) {
        const resp = await fetch(path, { cache: "no-store" });
        if (!resp.ok) throw new Error(await resp.text());
        return await resp.json();
      }

      async function loadDate(dateStr) {
        if (!dateStr) return;
        setStatus("warn", "loading");
        $("metaLaowang").textContent = "";
        $("metaFhkq").textContent = "";
        const st = await apiGet(`/api/status?trade_date=${encodeURIComponent(dateStr)}`);
        if (!st.has_stock_daily) {
          setStatus("err", "该日无K线数据");
          renderTable("tableLaowang", "emptyLaowang", { columns: [], rows: [], meta: { empty_hint: "no data" }});
          renderTable("tableFhkq", "emptyFhkq", { columns: [], rows: [], meta: { empty_hint: "no data" }});
          return;
        }
        const ok = st.laowang_rows > 0 || st.fhkq_rows > 0;
        setStatus(ok ? "ok" : "warn", `LW:${st.laowang_rows} F:${st.fhkq_rows}`);

        const [lw, fk] = await Promise.all([
          apiGet(`/api/model/laowang?trade_date=${encodeURIComponent(dateStr)}`),
          apiGet(`/api/model/fhkq?trade_date=${encodeURIComponent(dateStr)}`),
        ]);
        $("metaLaowang").textContent = `rows=${lw.meta && lw.meta.rows ? lw.meta.rows : lw.rows.length}`;
        $("metaFhkq").textContent = `rows=${fk.meta && fk.meta.rows ? fk.meta.rows : fk.rows.length}`;
        renderTable("tableLaowang", "emptyLaowang", lw);
        renderTable("tableFhkq", "emptyFhkq", fk);
      }

      async function boot() {
        const datesPayload = await apiGet("/api/dates");
        const sel = $("tradeDate");
        sel.innerHTML = "";
        (datesPayload.dates || []).forEach(d => {
          const opt = document.createElement("option");
          opt.value = d;
          opt.textContent = d.replaceAll("-", "");
          sel.appendChild(opt);
        });
        const latest = datesPayload.latest || (datesPayload.dates && datesPayload.dates[0]) || "";
        if (latest) sel.value = latest;
        sel.addEventListener("change", async () => {
          await loadDate(sel.value);
        });
        if (sel.value) await loadDate(sel.value);
      }

      boot().catch(e => {
        console.error(e);
        setStatus("err", "load error");
      });
    </script>
  </body>
</html>
"""


def _parse_status_tags(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:  # noqa: BLE001
            data = None
        if isinstance(data, list):
            return [str(x) for x in data if str(x).strip()]
        cleaned = raw.strip()
        return [cleaned] if cleaned else []
    return []


def _json(handler: BaseHTTPRequestHandler, obj: Any, *, status: int = 200) -> None:
    b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(int(status))
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(b)))
    handler.end_headers()
    handler.wfile.write(b)


def _text(handler: BaseHTTPRequestHandler, s: str, *, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
    b = (s or "").encode("utf-8")
    handler.send_response(int(status))
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(b)))
    handler.end_headers()
    handler.wfile.write(b)


class AppContext:
    def __init__(self, engine: Engine, min_trade_date: Optional[str]) -> None:
        self.engine = engine
        self.min_trade_date = min_trade_date  # YYYY-MM-DD

    def list_dates(self) -> Tuple[List[str], Optional[str]]:
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT DISTINCT date FROM stock_daily ORDER BY date DESC")).fetchall()
        dates = [str(r[0]) for r in rows if r and r[0]]
        if self.min_trade_date:
            dates = [d for d in dates if d >= self.min_trade_date]
        latest = dates[0] if dates else None
        return dates, latest

    def status(self, trade_date: str) -> Dict[str, Any]:
        if self.min_trade_date and trade_date < self.min_trade_date:
            return {
                "trade_date": trade_date,
                "has_stock_daily": False,
                "laowang_rows": 0,
                "fhkq_rows": 0,
            }
        with self.engine.connect() as conn:
            has_daily = conn.execute(text("SELECT 1 FROM stock_daily WHERE date = :d LIMIT 1"), {"d": trade_date}).fetchone()
            lw = conn.execute(text("SELECT COUNT(*) FROM model_laowang_pool WHERE trade_date = :d"), {"d": trade_date}).fetchone()[0]
            fk = conn.execute(text("SELECT COUNT(*) FROM model_fhkq WHERE trade_date = :d"), {"d": trade_date}).fetchone()[0]
        return {
            "trade_date": trade_date,
            "has_stock_daily": bool(has_daily),
            "laowang_rows": int(lw or 0),
            "fhkq_rows": int(fk or 0),
        }

    def fetch_laowang(self, trade_date: str) -> Dict[str, Any]:
        cols = ["rank_no", "stock_code", "stock_name", "close", "support_level", "resistance_level", "total_score", "status_tags"]
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT rank_no, stock_code, stock_name, close, support_level, resistance_level, total_score, status_tags
                    FROM model_laowang_pool
                    WHERE trade_date = :d
                    ORDER BY rank_no ASC
                    """
                ),
                {"d": trade_date},
            ).fetchall()
        out_rows: List[Dict[str, Any]] = []
        for row in rows:
            data = {}
            for idx, col in enumerate(cols):
                val = row[idx] if idx < len(row) else None
                if col == "status_tags":
                    data[col] = _parse_status_tags(val)
                else:
                    data[col] = val
            out_rows.append(data)
        return {"columns": cols, "rows": out_rows, "meta": {"rows": len(out_rows), "empty_hint": "0 rows"}}

    def fetch_fhkq(self, trade_date: str) -> Dict[str, Any]:
        cols = [
            "stock_code",
            "stock_name",
            "consecutive_limit_down",
            "last_limit_down",
            "volume_ratio",
            "amount_ratio",
            "open_board_flag",
            "liquidity_exhaust",
            "fhkq_score",
            "fhkq_level",
        ]
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT stock_code, stock_name, consecutive_limit_down, last_limit_down, volume_ratio,
                           amount_ratio, open_board_flag, liquidity_exhaust, fhkq_score, fhkq_level
                    FROM model_fhkq
                    WHERE trade_date = :d
                    ORDER BY fhkq_score DESC, consecutive_limit_down DESC, stock_code ASC
                    """
                ),
                {"d": trade_date},
            ).fetchall()
        out_rows = [{col: row[idx] if idx < len(row) else None for idx, col in enumerate(cols)} for row in rows]
        return {"columns": cols, "rows": out_rows, "meta": {"rows": len(out_rows), "empty_hint": "0 rows"}}


class Handler(BaseHTTPRequestHandler):
    server_version = "LAOWANGFHKQ-UI/1.0"

    @property
    def app(self) -> AppContext:  # type: ignore[override]
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ANN401
        logging.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        path = u.path
        q = parse_qs(u.query or "")
        if path in {"", "/"}:
            _text(self, HTML_PAGE, content_type="text/html; charset=utf-8")
            return
        if path == "/api/dates":
            dates, latest = self.app.list_dates()
            _json(self, {"dates": dates, "latest": latest})
            return
        if path == "/api/status":
            trade_date = (q.get("trade_date") or [""])[0]
            trade_date = str(trade_date).strip()
            if not trade_date:
                _json(self, {"error": "trade_date required"}, status=400)
                return
            _json(self, self.app.status(trade_date))
            return
        if path.startswith("/api/model/"):
            trade_date = (q.get("trade_date") or [""])[0]
            trade_date = str(trade_date).strip()
            if not trade_date:
                _json(self, {"error": "trade_date required"}, status=400)
                return
            name = path.split("/")[-1]
            if name == "laowang":
                _json(self, self.app.fetch_laowang(trade_date))
                return
            if name == "fhkq":
                _json(self, self.app.fetch_fhkq(trade_date))
                return
            _json(self, {"error": "unknown model"}, status=404)
            return
        _text(self, "not found", status=404)


class DailyJobRunner:
    def __init__(
        self,
        *,
        auto_time: str,
        config: Optional[str],
        db_url: Optional[str],
        db: Optional[str],
        initial_start: str,
        get_workers: int,
        laowang_workers: int,
        fhkq_workers: int,
        laowang_top: int,
        laowang_min_score: float,
    ) -> None:
        self.config = config
        self.db_url = db_url
        self.db = db
        self.initial_start = initial_start
        self.get_workers = int(get_workers)
        self.laowang_workers = int(laowang_workers)
        self.fhkq_workers = int(fhkq_workers)
        self.laowang_top = int(laowang_top)
        self.laowang_min_score = float(laowang_min_score)
        self.hour, self.minute = self._parse_time(auto_time)
        self.last_run_date: Optional[dt.date] = None
        self.thread = threading.Thread(target=self._loop, name="DailyJobRunner", daemon=True)
        self.thread.start()

    def _parse_time(self, s: str) -> Tuple[int, int]:
        try:
            parts = str(s or "15:05").split(":")
            h = max(0, min(23, int(parts[0])))
            m = max(0, min(59, int(parts[1]) if len(parts) > 1 else 0))
            return h, m
        except Exception:
            return 15, 5

    def _run_job(self) -> None:
        logging.info("[auto] everyday start")
        try:
            everyday.run_once(
                config=self.config,
                db_url=self.db_url,
                db=self.db,
                initial_start_date=self.initial_start,
                getdata_workers=self.get_workers,
                laowang_workers=self.laowang_workers,
                fhkq_workers=self.fhkq_workers,
                laowang_top=self.laowang_top,
                laowang_min_score=self.laowang_min_score,
            )
            logging.info("[auto] everyday finished")
        except Exception:
            logging.exception("[auto] everyday failed")

    def _loop(self) -> None:
        while True:
            now = dt.datetime.now()
            target = now.replace(hour=self.hour, minute=self.minute, second=0, microsecond=0)
            if now >= target:
                if self.last_run_date != now.date():
                    self._run_job()
                    self.last_run_date = now.date()
                target = target + dt.timedelta(days=1)
            sleep_sec = max(30.0, min(300.0, (target - now).total_seconds()))
            time.sleep(sleep_sec)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LAOWANG/FHKQ 浏览 UI（只读 + 自动任务）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--start-date", default=None, help="仅显示该日期及之后的交易日 (YYYYMMDD 或 YYYY-MM-DD)")
    parser.add_argument("--disable-auto-update", action="store_true", help="禁用 15:05 自动执行 everyday.py")
    parser.add_argument("--auto-time", default="15:05", help="HH:MM（默认 15:05）")
    parser.add_argument("--auto-init-start-date", default="2000-01-01")
    parser.add_argument("--auto-getdata-workers", type=int, default=16)
    parser.add_argument("--auto-laowang-workers", type=int, default=16)
    parser.add_argument("--auto-fhkq-workers", type=int, default=8)
    parser.add_argument("--auto-laowang-top", type=int, default=200)
    parser.add_argument("--auto-laowang-min-score", type=float, default=60.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    db_target = resolve_db_target(args)
    engine = make_engine(db_target)

    min_date_iso = _normalize_iso_date(args.start_date)
    app = AppContext(engine, min_trade_date=min_date_iso)
    scheduler: Optional[DailyJobRunner] = None
    if not args.disable_auto_update:
        scheduler = DailyJobRunner(
            auto_time=args.auto_time,
            config=args.config,
            db_url=args.db_url,
            db=args.db,
            initial_start=args.auto_init_start_date,
            get_workers=int(args.auto_getdata_workers),
            laowang_workers=int(args.auto_laowang_workers),
            fhkq_workers=int(args.auto_fhkq_workers),
            laowang_top=int(args.auto_laowang_top),
            laowang_min_score=float(args.auto_laowang_min_score),
        )

    httpd = ThreadingHTTPServer((str(args.host), int(args.port)), Handler)
    httpd.app = app  # type: ignore[attr-defined]
    url = f"http://{args.host}:{int(args.port)}"
    logging.info("UI running: %s", url)
    if scheduler:
        logging.info("自动任务每日 %s 运行", args.auto_time)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
