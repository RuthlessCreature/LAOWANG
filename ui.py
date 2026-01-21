# -*- coding: utf-8 -*-
"""
ui.py

Local minimal Web UI for browsing per-day model outputs stored in MySQL:
- LAOWANG pool (model_laowang_pool)
- FHKQ signals (model_fhkq)

Design goals:
- No external UI dependencies (pure stdlib + SQLAlchemy already in requirements)
- Dark / minimalist / "futuristic line" style
- No DB url text on the page
- No manual "update data" button; UI will auto-compute missing model outputs
  for the selected trading day (based on existing `stock_daily`).

Run:
  python ui.py --config config.ini
Then open:
  http://127.0.0.1:8000
"""

from __future__ import annotations

try:
    import sitecustomize  # noqa: F401
except Exception:
    # UI itself does not fetch network data, but keeping this import makes the
    # runtime consistent with other scripts (proxy bypass, etc.).
    pass

import argparse
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from sqlalchemy import text

from a_stock_analyzer import db as adb
from a_stock_analyzer.runtime import add_db_args, resolve_db_from_args, setup_logging

from modeling import db as mdb
from modeling.registry import build_models
from modeling.runner import ensure_tables


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
        --line2: rgba(124, 92, 255, 0.22);
        --accent: #00e5ff;
        --accent2: #7c5cff;
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
      .brand { display: flex; flex-direction: column; gap: 4px; }
      .brand .title {
        font-weight: 700; letter-spacing: 0.12em; font-size: 13px;
        color: rgba(219,231,255,0.95);
        text-transform: uppercase;
      }
      .brand .sub { font-size: 12px; color: var(--muted); }

      .controls { display: flex; gap: 10px; align-items: center; }
      .controls label { font-size: 12px; color: var(--muted); }
      select {
        background: rgba(10,14,22,0.95);
        border: 1px solid var(--line);
        color: var(--text);
        border-radius: 10px;
        padding: 8px 10px;
        outline: none;
        min-width: 160px;
        font-family: var(--mono);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
      }
      select:focus { border-color: rgba(0,229,255,0.6); }

      .status {
        display: flex; gap: 10px; align-items: center;
        font-family: var(--mono);
        font-size: 12px;
        color: var(--muted);
      }
      .dot { width: 8px; height: 8px; border-radius: 50%; background: rgba(255,255,255,0.15); }
      .dot.ok { background: rgba(51,255,166,0.90); box-shadow: 0 0 10px rgba(51,255,166,0.55); }
      .dot.warn { background: rgba(255,204,102,0.90); box-shadow: 0 0 10px rgba(255,204,102,0.55); }
      .dot.err { background: rgba(255,85,119,0.90); box-shadow: 0 0 10px rgba(255,85,119,0.55); }
      .dot.run { background: rgba(0,229,255,0.90); box-shadow: 0 0 10px rgba(0,229,255,0.55); }

      .tabs {
        margin-top: 14px;
        display: flex; gap: 10px; align-items: center;
      }
      .tabbtn {
        cursor: pointer;
        border: 1px solid var(--line);
        color: var(--text);
        background: rgba(11,15,23,0.65);
        padding: 10px 12px;
        border-radius: 12px;
        font-family: var(--mono);
        letter-spacing: 0.04em;
        box-shadow: 0 10px 28px rgba(0,0,0,0.25);
      }
      .tabbtn.active {
        border-color: rgba(0,229,255,0.65);
        box-shadow: 0 0 0 1px rgba(0,229,255,0.18), 0 10px 28px rgba(0,0,0,0.35);
      }

      .panel {
        margin-top: 12px;
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
      }
      .panel-title {
        font-family: var(--mono);
        font-size: 13px;
        color: rgba(219,231,255,0.95);
        letter-spacing: 0.06em;
      }
      .panel-meta { font-family: var(--mono); font-size: 12px; color: var(--muted); }

      .table-wrap { overflow: auto; }
      table { width: 100%; border-collapse: collapse; }
      thead th {
        position: sticky; top: 0;
        background: rgba(7,10,15,0.90);
        color: rgba(219,231,255,0.95);
        text-align: left;
        font-family: var(--mono);
        font-size: 12px;
        padding: 10px 10px;
        border-bottom: 1px solid rgba(0,229,255,0.22);
        white-space: nowrap;
      }
      tbody td {
        padding: 9px 10px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-family: var(--mono);
        font-size: 12px;
        color: rgba(219,231,255,0.92);
        white-space: nowrap;
      }
      tbody tr:hover td {
        background: rgba(0,229,255,0.06);
      }
      .muted { color: var(--muted); }

      .empty {
        padding: 18px 14px;
        color: var(--muted);
        font-family: var(--mono);
        font-size: 12px;
      }

      .spinner {
        display: inline-block;
        width: 14px; height: 14px;
        border: 2px solid rgba(0,229,255,0.18);
        border-top-color: rgba(0,229,255,0.88);
        border-radius: 50%;
        animation: spin 0.9s linear infinite;
        vertical-align: -2px;
        margin-right: 8px;
      }
      @keyframes spin { to { transform: rotate(360deg); } }

      .pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 8px;
        border: 1px solid rgba(0,229,255,0.25);
        border-radius: 999px;
        background: rgba(0,229,255,0.06);
        font-family: var(--mono);
        font-size: 12px;
        color: rgba(219,231,255,0.92);
      }
      .pill.err { border-color: rgba(255,85,119,0.35); background: rgba(255,85,119,0.10); }
      .pill.warn { border-color: rgba(255,204,102,0.35); background: rgba(255,204,102,0.10); }

      .right { text-align: right; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="topbar">
        <div class="brand">
          <div class="title">Model Console</div>
          <div class="sub">LAOWANG / FHKQ · Daily model outputs</div>
        </div>
        <div class="controls">
          <label for="tradeDate">交易日</label>
          <select id="tradeDate"></select>
          <div class="status" id="globalStatus">
            <span class="dot" id="statusDot"></span>
            <span id="statusText">loading</span>
          </div>
        </div>
      </div>

      <div class="tabs">
        <button class="tabbtn active" id="tabLaowang">LAOWANG</button>
        <button class="tabbtn" id="tabFhkq">FHKQ</button>
      </div>

      <div class="panel" id="panelLaowang">
        <div class="panel-header">
          <div class="panel-title">model_laowang_pool</div>
          <div class="panel-meta" id="metaLaowang"></div>
        </div>
        <div class="table-wrap">
          <table id="tableLaowang">
            <thead></thead>
            <tbody></tbody>
          </table>
          <div class="empty" id="emptyLaowang" style="display:none;"></div>
        </div>
      </div>

      <div class="panel" id="panelFhkq" style="display:none;">
        <div class="panel-header">
          <div class="panel-title">model_fhkq</div>
          <div class="panel-meta" id="metaFhkq"></div>
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
      const sleep = (ms) => new Promise(r => setTimeout(r, ms));

      function setStatus(kind, text) {
        const dot = $("statusDot");
        dot.className = "dot";
        if (kind) dot.classList.add(kind);
        $("statusText").textContent = text || "";
      }

      function setTab(which) {
        const isLW = which === "laowang";
        $("tabLaowang").classList.toggle("active", isLW);
        $("tabFhkq").classList.toggle("active", !isLW);
        $("panelLaowang").style.display = isLW ? "" : "none";
        $("panelFhkq").style.display = isLW ? "none" : "";
      }

      function renderTable(tableId, emptyId, metaId, payload) {
        const table = $(tableId);
        const thead = table.querySelector("thead");
        const tbody = table.querySelector("tbody");
        const empty = $(emptyId);
        const meta = $(metaId);

        const cols = payload.columns || [];
        const rows = payload.rows || [];
        const m = payload.meta || {};

        meta.textContent = m && m.message ? m.message : "";

        thead.innerHTML = "";
        tbody.innerHTML = "";

        if (!cols.length) {
          empty.style.display = "";
          empty.textContent = "no columns";
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
          empty.textContent = (m && m.empty_hint) ? m.empty_hint : "0 rows";
          return;
        }

        empty.style.display = "none";
        rows.forEach(r => {
          const tr = document.createElement("tr");
          cols.forEach(c => {
            const td = document.createElement("td");
            let v = r[c];
            if (v === null || v === undefined) v = "";
            td.textContent = String(v);
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
      }

      async function apiGet(path) {
        const resp = await fetch(path, { cache: "no-store" });
        if (!resp.ok) {
          const txt = await resp.text();
          throw new Error(`HTTP ${resp.status}: ${txt}`);
        }
        return await resp.json();
      }

      async function apiPost(path, body) {
        const resp = await fetch(path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body || {}),
        });
        if (!resp.ok) {
          const txt = await resp.text();
          throw new Error(`HTTP ${resp.status}: ${txt}`);
        }
        return await resp.json();
      }

      async function ensureComputed(dateStr) {
        // Start compute job (idempotent) and wait until both models are ok or error.
        await apiPost("/api/compute", { trade_date: dateStr });
        for (let i = 0; i < 300; i++) { // ~10 min max (2s interval)
          const st = await apiGet(`/api/status?trade_date=${encodeURIComponent(dateStr)}`);
          if (st && st.all_ok) return st;
          if (st && st.any_error) return st;
          await sleep(2000);
        }
        return { timeout: true };
      }

      async function loadDate(dateStr) {
        // Clear tables and show loading.
        setStatus("run", "computing");
        renderTable("tableLaowang", "emptyLaowang", "metaLaowang", {
          columns: ["..."],
          rows: [],
          meta: { empty_hint: "loading..." }
        });
        renderTable("tableFhkq", "emptyFhkq", "metaFhkq", {
          columns: ["..."],
          rows: [],
          meta: { empty_hint: "loading..." }
        });

        let st = null;
        try {
          st = await apiGet(`/api/status?trade_date=${encodeURIComponent(dateStr)}`);
        } catch (e) {
          setStatus("err", "status error");
          $("metaLaowang").textContent = String(e);
          $("metaFhkq").textContent = String(e);
          return;
        }

        if (!st || !st.has_stock_daily) {
          setStatus("warn", "no data");
          $("metaLaowang").textContent = "No stock_daily for this date";
          $("metaFhkq").textContent = "No stock_daily for this date";
          renderTable("tableLaowang", "emptyLaowang", "metaLaowang", { columns: ["info"], rows: [{info: ""}], meta: { empty_hint: "no data" }});
          renderTable("tableFhkq", "emptyFhkq", "metaFhkq", { columns: ["info"], rows: [{info: ""}], meta: { empty_hint: "no data" }});
          return;
        }

        const needCompute = !(st.all_ok);
        if (needCompute) {
          setStatus("run", "computing");
          $("metaLaowang").innerHTML = `<span class="spinner"></span>computing...`;
          $("metaFhkq").innerHTML = `<span class="spinner"></span>computing...`;
          st = await ensureComputed(dateStr);
        }

        if (st && st.any_error) {
          setStatus("err", "error");
        } else if (st && st.all_ok) {
          setStatus("ok", "ok");
        } else {
          setStatus("warn", "timeout/unknown");
        }

        // Fetch tables.
        try {
          const lw = await apiGet(`/api/model/laowang?trade_date=${encodeURIComponent(dateStr)}`);
          renderTable("tableLaowang", "emptyLaowang", "metaLaowang", lw);
        } catch (e) {
          renderTable("tableLaowang", "emptyLaowang", "metaLaowang", {
            columns: ["error"],
            rows: [{ error: String(e) }],
            meta: { empty_hint: "error" }
          });
        }

        try {
          const fk = await apiGet(`/api/model/fhkq?trade_date=${encodeURIComponent(dateStr)}`);
          renderTable("tableFhkq", "emptyFhkq", "metaFhkq", fk);
        } catch (e) {
          renderTable("tableFhkq", "emptyFhkq", "metaFhkq", {
            columns: ["error"],
            rows: [{ error: String(e) }],
            meta: { empty_hint: "error" }
          });
        }
      }

      async function boot() {
        setStatus(null, "loading");
        $("tabLaowang").addEventListener("click", () => setTab("laowang"));
        $("tabFhkq").addEventListener("click", () => setTab("fhkq"));

        const data = await apiGet("/api/dates");
        const dates = data.dates || [];
        const latest = data.latest || (dates.length ? dates[0] : "");

        const sel = $("tradeDate");
        sel.innerHTML = "";
        dates.forEach(d => {
          const opt = document.createElement("option");
          opt.value = d;
          opt.textContent = d.replaceAll("-", "");
          sel.appendChild(opt);
        });
        if (latest) sel.value = latest;

        sel.addEventListener("change", async () => {
          await loadDate(sel.value);
        });

        if (latest) await loadDate(latest);

        // Lightweight auto-refresh (60s):
        // - refresh available trading dates (so new date appears after your 15:05 pipeline update)
        // - refresh tables for the currently selected date (only when computed ok)
        let lastLatest = latest || "";

        async function refreshDatesMaybe() {
          try {
            const d2 = await apiGet("/api/dates");
            const dates2 = d2.dates || [];
            const latest2 = d2.latest || (dates2.length ? dates2[0] : "");
            if (!latest2) return;

            const cur = sel.value;
            const needRebuild =
              (sel.options.length !== dates2.length) ||
              (sel.options.length > 0 && sel.options[0].value !== dates2[0]);
            if (needRebuild) {
              sel.innerHTML = "";
              dates2.forEach(d => {
                const opt = document.createElement("option");
                opt.value = d;
                opt.textContent = d.replaceAll("-", "");
                sel.appendChild(opt);
              });
              if (cur && dates2.includes(cur)) sel.value = cur;
              else sel.value = latest2;
            }

            // Auto-switch only if user was already viewing the latest.
            if (cur === lastLatest && latest2 !== lastLatest) {
              sel.value = latest2;
              await loadDate(latest2);
            }
            lastLatest = latest2;
          } catch (e) {
            // ignore
          }
        }

        setInterval(async () => {
          await refreshDatesMaybe();
          const cur = sel.value;
          if (!cur) return;
          try {
            const st = await apiGet(`/api/status?trade_date=${encodeURIComponent(cur)}`);
            if (st && st.all_ok) {
              const lw = await apiGet(`/api/model/laowang?trade_date=${encodeURIComponent(cur)}`);
              renderTable("tableLaowang", "emptyLaowang", "metaLaowang", lw);
              const fk = await apiGet(`/api/model/fhkq?trade_date=${encodeURIComponent(cur)}`);
              renderTable("tableFhkq", "emptyFhkq", "metaFhkq", fk);
            }
          } catch (e) {
            // ignore
          }
        }, 60000);
      }

      boot().catch(e => {
        setStatus("err", "boot error");
        console.error(e);
      });
    </script>
  </body>
</html>
"""


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


def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    try:
        n = int(handler.headers.get("Content-Length", "0") or "0")
    except Exception:
        n = 0
    if n <= 0:
        return {}
    raw = handler.rfile.read(n)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


class AppState:
    def __init__(self, *, engine, workers: int, laowang_top: int, laowang_min_score: float) -> None:  # noqa: ANN001
        self.engine = engine
        self.workers = int(max(1, workers))
        self.models = build_models(
            only="both",
            workers=int(max(1, workers)),
            laowang_top=int(laowang_top),
            laowang_min_score=float(laowang_min_score),
        )
        ensure_tables(self.engine, self.models)

        self._jobs_lock = threading.Lock()
        self._jobs: Dict[str, Tuple[threading.Thread, float]] = {}

    def list_dates(self) -> Tuple[List[str], Optional[str]]:
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT DISTINCT date FROM stock_daily ORDER BY date DESC")).fetchall()
        dates = [str(r[0]) for r in rows if r and r[0]]
        latest = dates[0] if dates else None
        return dates, latest

    def has_stock_daily(self, trade_date: str) -> bool:
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT 1 FROM stock_daily WHERE date = :d LIMIT 1"), {"d": trade_date}).fetchone()
        return row is not None

    def _run_status(self, trade_date: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"trade_date": trade_date, "models": {}, "has_stock_daily": self.has_stock_daily(trade_date)}
        for m in self.models:
            with self.engine.connect() as conn:
                row = conn.execute(
                    text(
                        """
                        SELECT status, row_count, message, started_at, finished_at
                        FROM model_runs
                        WHERE model_name = :m AND trade_date = :d
                        """
                    ),
                    {"m": m.name, "d": trade_date},
                ).fetchone()
            if row:
                out["models"][m.name] = {
                    "status": str(row[0] or ""),
                    "row_count": int(row[1] or 0),
                    "message": str(row[2] or ""),
                    "started_at": str(row[3] or ""),
                    "finished_at": str(row[4] or ""),
                }
            else:
                out["models"][m.name] = {"status": "missing", "row_count": 0, "message": "", "started_at": "", "finished_at": ""}

        all_ok = True
        any_error = False
        for v in out["models"].values():
            st = str(v.get("status") or "")
            if st != "ok":
                all_ok = False
            if st == "error":
                any_error = True

        out["all_ok"] = bool(all_ok)
        out["any_error"] = bool(any_error)
        out["running"] = self.is_job_running(trade_date)
        return out

    def is_job_running(self, trade_date: str) -> bool:
        with self._jobs_lock:
            it = self._jobs.get(trade_date)
            if not it:
                return False
            t, _ts = it
            return t.is_alive()

    def start_compute(self, trade_date: str) -> bool:
        if not self.has_stock_daily(trade_date):
            return False
        with self._jobs_lock:
            it = self._jobs.get(trade_date)
            if it and it[0].is_alive():
                return True

            t = threading.Thread(target=self._compute_for_date, args=(trade_date,), daemon=True)
            self._jobs[trade_date] = (t, time.time())
            t.start()
            return True

    def _compute_for_date(self, trade_date: str) -> None:
        # Idempotent per date: if already ok, skip.
        for m in self.models:
            if mdb.is_model_ok(self.engine, model_name=m.name, trade_date=trade_date):
                continue
            started = mdb.now_ts()
            try:
                logging.info("[%s] compute %s (ui)", m.name, trade_date)
                df = m.compute(engine=self.engine, trade_date=trade_date, workers=int(self.workers))
                n = m.save(engine=self.engine, trade_date=trade_date, df=df)
                mdb.write_model_run(
                    self.engine,
                    model_name=m.name,
                    trade_date=trade_date,
                    status="ok",
                    row_count=int(n),
                    message="",
                    started_at=started,
                    finished_at=mdb.now_ts(),
                )
            except Exception as e:  # noqa: BLE001
                mdb.write_model_run(
                    self.engine,
                    model_name=m.name,
                    trade_date=trade_date,
                    status="error",
                    row_count=0,
                    message=f"{type(e).__name__}: {e}",
                    started_at=started,
                    finished_at=mdb.now_ts(),
                )
                logging.exception("[%s] failed %s (ui)", m.name, trade_date)

    def fetch_model_table(self, model_name: str, trade_date: str) -> Dict[str, Any]:
        # Return {columns, rows, meta}
        if model_name == "laowang":
            cols = [
                "rank_no",
                "stock_code",
                "stock_name",
                "close",
                "support_level",
                "resistance_level",
                "total_score",
            ]
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT rank_no, stock_code, stock_name, close, support_level, resistance_level, total_score
                        FROM model_laowang_pool
                        WHERE trade_date = :d
                        ORDER BY rank_no ASC
                        """
                    ),
                    {"d": trade_date},
                ).fetchall()
            out_rows: List[Dict[str, Any]] = []
            for r in rows:
                out_rows.append({c: (r[i] if i < len(r) else None) for i, c in enumerate(cols)})

            st = self._run_status(trade_date)["models"].get("laowang", {})
            msg = f"status={st.get('status','')} rows={st.get('row_count',0)}"
            if st.get("status") == "error" and st.get("message"):
                msg = msg + f" | {st.get('message')}"
            return {"columns": cols, "rows": out_rows, "meta": {"message": msg, "empty_hint": "0 rows (computed)"}}

        if model_name == "fhkq":
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
            out_rows2: List[Dict[str, Any]] = []
            for r in rows:
                out_rows2.append({c: (r[i] if i < len(r) else None) for i, c in enumerate(cols)})

            st = self._run_status(trade_date)["models"].get("fhkq", {})
            msg = f"status={st.get('status','')} rows={st.get('row_count',0)}"
            if st.get("status") == "error" and st.get("message"):
                msg = msg + f" | {st.get('message')}"
            return {"columns": cols, "rows": out_rows2, "meta": {"message": msg, "empty_hint": "0 rows (computed; no signals)"}}

        return {"columns": ["error"], "rows": [{"error": f"unknown model: {model_name}"}], "meta": {"message": ""}}


class Handler(BaseHTTPRequestHandler):
    server_version = "LAOWANG-UI/0.1"

    @property
    def app(self) -> AppState:  # type: ignore[override]
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ANN401
        logging.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        path = u.path
        q = parse_qs(u.query or "")

        if path in {"", "/"}:
            _text(self, HTML_PAGE, status=200, content_type="text/html; charset=utf-8")
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
            _json(self, self.app._run_status(trade_date))
            return

        if path.startswith("/api/model/"):
            trade_date = (q.get("trade_date") or [""])[0]
            trade_date = str(trade_date).strip()
            if not trade_date:
                _json(self, {"error": "trade_date required"}, status=400)
                return
            model_name = path.split("/")[-1].strip()
            _json(self, self.app.fetch_model_table(model_name, trade_date))
            return

        _text(self, "not found", status=404)

    def do_POST(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        if u.path == "/api/compute":
            body = _read_json_body(self)
            trade_date = str(body.get("trade_date") or "").strip()
            if not trade_date:
                _json(self, {"error": "trade_date required"}, status=400)
                return
            ok = self.app.start_compute(trade_date)
            if not ok:
                _json(self, {"ok": False, "message": "no stock_daily for date"}, status=400)
                return
            _json(self, {"ok": True})
            return
        _text(self, "not found", status=404)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local Web UI for LAOWANG/FHKQ (DB materialized)")
    p.add_argument("--log-level", default="INFO")
    add_db_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--laowang-top", type=int, default=200)
    p.add_argument("--laowang-min-score", type=float, default=0.0)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    db_target = resolve_db_from_args(args)
    w = max(1, int(args.workers))
    pool_size = max(10, min(32, w * 2))
    max_overflow = max(20, min(64, w * 2))
    engine = adb.make_engine(db_target, pool_size=pool_size, max_overflow=max_overflow)
    if engine.dialect.name != "mysql":
        raise SystemExit("ui.py requires MySQL (check config.ini / ASTOCK_DB_URL).")

    app = AppState(
        engine=engine,
        workers=int(args.workers),
        laowang_top=int(args.laowang_top),
        laowang_min_score=float(args.laowang_min_score),
    )

    httpd = ThreadingHTTPServer((str(args.host), int(args.port)), Handler)
    httpd.app = app  # type: ignore[attr-defined]

    url = f"http://{args.host}:{int(args.port)}"
    logging.info("UI: %s", url)
    logging.info("Tip: schedule `python everyday.py --config config.ini` at 15:05 to update data + models.")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
