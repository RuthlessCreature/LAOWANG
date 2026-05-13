#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small read-only Web UI for the four ordinary LAOWANG models."""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import parse_qs, urlparse

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine

from generate_trade_plan import load_config, make_engine, resolve_db_url, sql_text


FAVICON_PATH = Path(__file__).with_name("favicon.ico")
MODEL_ORDER = ("laowang", "stwg", "ywcx", "fhkq")

POOL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "laowang": {
        "title": "LAOWANG",
        "table": "model_laowang_pool",
        "score": "total_score",
        "columns": [
            "rank_no",
            "stock_code",
            "stock_name",
            "close",
            "support_level",
            "resistance_level",
            "total_score",
            "status_tags",
        ],
        "order": "rank_no ASC, total_score DESC",
    },
    "stwg": {
        "title": "STWG",
        "table": "model_stwg_pool",
        "score": "total_score",
        "columns": [
            "rank_no",
            "stock_code",
            "stock_name",
            "close",
            "total_score",
            "stageB_compression_score",
            "breakout_confirmation_score",
            "status_tags",
        ],
        "order": "rank_no ASC, total_score DESC",
    },
    "ywcx": {
        "title": "YWCX",
        "table": "model_ywcx_pool",
        "score": "total_score",
        "columns": [
            "rank_no",
            "stock_code",
            "stock_name",
            "close",
            "total_score",
            "weak_position_score",
            "volume_dry_score",
            "low_volatility_score",
            "status_tags",
        ],
        "order": "rank_no ASC, total_score DESC",
    },
    "fhkq": {
        "title": "FHKQ",
        "table": "model_fhkq",
        "score": "fhkq_score",
        "columns": [
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
        ],
        "order": "fhkq_score DESC",
    },
}


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LAOWANG</title>
  <link rel="icon" href="/favicon.ico" />
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #20242a;
      --muted: #68717d;
      --line: #d9dee6;
      --blue: #275efe;
      --green: #0f8f5f;
      --red: #b42318;
      --amber: #9a6700;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
    header { height: 56px; display: flex; align-items: center; gap: 14px; padding: 0 20px; border-bottom: 1px solid var(--line); background: var(--panel); position: sticky; top: 0; z-index: 2; }
    h1 { font-size: 18px; margin: 0; letter-spacing: 0; }
    main { padding: 16px 20px 28px; max-width: 1480px; margin: 0 auto; }
    .toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }
    select, button { height: 34px; border: 1px solid var(--line); background: #fff; border-radius: 6px; padding: 0 10px; color: var(--text); }
    button { cursor: pointer; }
    button.primary { border-color: var(--blue); color: #fff; background: var(--blue); }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(220px, 1fr)); gap: 12px; margin-bottom: 16px; }
    .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; min-height: 78px; }
    .metric b { display: block; font-size: 22px; margin-top: 4px; }
    .muted { color: var(--muted); }
    .tabs { display: flex; gap: 6px; border-bottom: 1px solid var(--line); margin-top: 8px; }
    .tab { border: 1px solid transparent; border-bottom: 0; background: transparent; border-radius: 6px 6px 0 0; }
    .tab.active { background: var(--panel); border-color: var(--line); color: var(--blue); }
    section { display: none; padding-top: 14px; }
    section.active { display: block; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    .panel-title { display: flex; align-items: center; justify-content: space-between; padding: 10px 12px; border-bottom: 1px solid var(--line); }
    .table-wrap { overflow: auto; max-height: 68vh; }
    table { border-collapse: collapse; width: 100%; min-width: 900px; }
    th, td { border-bottom: 1px solid #edf0f4; padding: 8px 10px; text-align: left; white-space: nowrap; }
    th { position: sticky; top: 0; background: #f9fafb; z-index: 1; font-weight: 650; }
    td.wrap { white-space: normal; min-width: 320px; max-width: 520px; }
    .ok { color: var(--green); font-weight: 650; }
    .bad { color: var(--red); font-weight: 650; }
    .warn { color: var(--amber); font-weight: 650; }
    .empty { padding: 18px; color: var(--muted); }
    @media (max-width: 900px) {
      main { padding: 12px; }
      .grid { grid-template-columns: repeat(2, minmax(160px, 1fr)); }
      header { padding: 0 12px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>LAOWANG</h1>
    <span class="muted">四模型日线工作台</span>
  </header>
  <main>
    <div class="toolbar">
      <label>交易日 <select id="dateSelect"></select></label>
      <button class="primary" id="refreshBtn">刷新</button>
      <span class="muted" id="statusText">loading...</span>
    </div>

    <div class="grid" id="metrics"></div>

    <div class="tabs">
      <button class="tab active" data-tab="plan">交易计划</button>
      <button class="tab" data-tab="laowang">LAOWANG</button>
      <button class="tab" data-tab="stwg">STWG</button>
      <button class="tab" data-tab="ywcx">YWCX</button>
      <button class="tab" data-tab="fhkq">FHKQ</button>
      <button class="tab" data-tab="positions">交易记录</button>
    </div>

    <section id="tab-plan" class="active">
      <div class="panel">
        <div class="panel-title"><b>交易计划</b><span class="muted">T+1 条件买入 / T+2+ 条件卖出</span></div>
        <div class="table-wrap"><table id="table-plan"></table><div id="empty-plan" class="empty"></div></div>
      </div>
    </section>

    <section id="tab-laowang"><div class="panel"><div class="panel-title"><b>LAOWANG</b><span class="muted" id="meta-laowang"></span></div><div class="table-wrap"><table id="table-laowang"></table><div id="empty-laowang" class="empty"></div></div></div></section>
    <section id="tab-stwg"><div class="panel"><div class="panel-title"><b>STWG</b><span class="muted" id="meta-stwg"></span></div><div class="table-wrap"><table id="table-stwg"></table><div id="empty-stwg" class="empty"></div></div></div></section>
    <section id="tab-ywcx"><div class="panel"><div class="panel-title"><b>YWCX</b><span class="muted" id="meta-ywcx"></span></div><div class="table-wrap"><table id="table-ywcx"></table><div id="empty-ywcx" class="empty"></div></div></div></section>
    <section id="tab-fhkq"><div class="panel"><div class="panel-title"><b>FHKQ</b><span class="muted" id="meta-fhkq"></span></div><div class="table-wrap"><table id="table-fhkq"></table><div id="empty-fhkq" class="empty"></div></div></div></section>
    <section id="tab-positions"><div class="panel"><div class="panel-title"><b>交易记录</b><span class="muted" id="meta-positions"></span></div><div class="table-wrap"><table id="table-positions"></table><div id="empty-positions" class="empty"></div></div></div></section>
  </main>

  <script>
    const MODELS = ["laowang", "stwg", "ywcx", "fhkq"];
    const $ = (id) => document.getElementById(id);

    async function api(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      return await res.json();
    }

    function esc(v) {
      if (v === null || v === undefined) return "";
      return String(v).replace(/[&<>"']/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[c]));
    }

    function renderTable(tableId, emptyId, payload) {
      const table = $(tableId);
      const empty = $(emptyId);
      const cols = payload.columns || [];
      const rows = payload.rows || [];
      table.innerHTML = "";
      empty.textContent = "";
      empty.style.display = rows.length ? "none" : "block";
      if (!rows.length) {
        empty.textContent = payload.empty_hint || "暂无数据";
        return;
      }
      const thead = `<thead><tr>${cols.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead>`;
      const body = rows.map(r => `<tr>${cols.map(c => {
        const cls = (c.includes("condition") || c.includes("reason")) ? " class='wrap'" : "";
        return `<td${cls}>${esc(r[c])}</td>`;
      }).join("")}</tr>`).join("");
      table.innerHTML = `${thead}<tbody>${body}</tbody>`;
    }

    function renderMetrics(status) {
      const counts = status.pool_counts || {};
      $("metrics").innerHTML = MODELS.map(m => {
        const val = counts[m] || 0;
        return `<div class="metric"><span class="muted">${m.toUpperCase()}</span><b>${val}</b><span class="muted">候选</span></div>`;
      }).join("");
    }

    async function loadDates() {
      const data = await api("/api/dates");
      const select = $("dateSelect");
      select.innerHTML = (data.dates || []).map(d => `<option value="${esc(d)}">${esc(d)}</option>`).join("");
      if (data.latest) select.value = data.latest;
    }

    async function refresh() {
      const date = $("dateSelect").value;
      $("statusText").textContent = "刷新中...";
      const [status, plan, positions, ...models] = await Promise.all([
        api(`/api/status?trade_date=${encodeURIComponent(date)}`),
        api(`/api/plan?trade_date=${encodeURIComponent(date)}`),
        api("/api/positions"),
        ...MODELS.map(m => api(`/api/model/${m}?trade_date=${encodeURIComponent(date)}`))
      ]);
      renderMetrics(status);
      renderTable("table-plan", "empty-plan", plan);
      renderTable("table-positions", "empty-positions", positions);
      $("meta-positions").textContent = `rows=${positions.rows ? positions.rows.length : 0}`;
      MODELS.forEach((m, i) => {
        renderTable(`table-${m}`, `empty-${m}`, models[i]);
        $(`meta-${m}`).textContent = `rows=${models[i].rows ? models[i].rows.length : 0}`;
      });
      $("statusText").textContent = `latest=${status.latest_trade_date || "-"} stock_daily=${status.stock_daily_rows || 0}`;
    }

    document.querySelectorAll(".tab").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
        document.querySelectorAll("section").forEach(x => x.classList.remove("active"));
        btn.classList.add("active");
        $(`tab-${btn.dataset.tab}`).classList.add("active");
      });
    });
    $("refreshBtn").addEventListener("click", refresh);
    $("dateSelect").addEventListener("change", refresh);

    (async function init() {
      try {
        await loadDates();
        await refresh();
      } catch (err) {
        $("statusText").textContent = `加载失败: ${err.message}`;
      }
    })();
  </script>
</body>
</html>
"""


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _send_json(handler: BaseHTTPRequestHandler, payload: Mapping[str, Any], status: int = 200) -> None:
    data = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _send_html(handler: BaseHTTPRequestHandler) -> None:
    data = HTML_PAGE.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _send_favicon(handler: BaseHTTPRequestHandler) -> None:
    if not FAVICON_PATH.exists():
        handler.send_error(404)
        return
    data = FAVICON_PATH.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", "image/x-icon")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _first(params: Mapping[str, List[str]], key: str, default: Optional[str] = None) -> Optional[str]:
    values = params.get(key)
    return values[0] if values else default


def _parse_plan(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        loaded = json.loads(str(raw))
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        return {}


class LaowangApp:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def dates(self) -> Dict[str, Any]:
        with self.engine.connect() as conn:
            rows = conn.execute(sql_text("SELECT DISTINCT date FROM stock_daily ORDER BY date DESC LIMIT 160")).all()
        dates = [str(row[0]) for row in rows if row and row[0]]
        return {"latest": dates[0] if dates else None, "dates": dates}

    def status(self, trade_date: Optional[str]) -> Dict[str, Any]:
        date_value = trade_date or self.dates().get("latest")
        pool_counts: Dict[str, int] = {}
        with self.engine.connect() as conn:
            latest = conn.execute(sql_text("SELECT MAX(date) FROM stock_daily")).scalar()
            stock_daily_rows = conn.execute(sql_text("SELECT COUNT(*) FROM stock_daily")).scalar()
            for model, cfg in POOL_CONFIGS.items():
                try:
                    count = conn.execute(
                        sql_text(f"SELECT COUNT(*) FROM {cfg['table']} WHERE trade_date = :d"),
                        {"d": date_value},
                    ).scalar()
                except SQLAlchemyError:
                    count = 0
                pool_counts[model] = int(count or 0)
        return {
            "trade_date": date_value,
            "latest_trade_date": str(latest) if latest else None,
            "stock_daily_rows": int(stock_daily_rows or 0),
            "pool_counts": pool_counts,
        }

    def model_rows(self, model: str, trade_date: Optional[str], limit: int = 300) -> Dict[str, Any]:
        key = model.lower()
        if key not in POOL_CONFIGS:
            return {"error": f"unknown model: {model}", "columns": [], "rows": []}
        cfg = POOL_CONFIGS[key]
        date_value = trade_date or self.dates().get("latest")
        columns = list(cfg["columns"])
        select_cols = ", ".join(columns)
        query = f"""
            SELECT {select_cols}
            FROM {cfg['table']}
            WHERE trade_date = :d
            ORDER BY {cfg['order']}
            LIMIT :lim
        """
        try:
            with self.engine.connect() as conn:
                rows = [dict(row) for row in conn.execute(sql_text(query), {"d": date_value, "lim": int(limit)}).mappings()]
        except SQLAlchemyError as exc:
            return {"columns": columns, "rows": [], "empty_hint": f"{cfg['table']} unavailable: {exc}"}
        return {"columns": columns, "rows": rows, "trade_date": date_value}

    def plan(self, trade_date: Optional[str]) -> Dict[str, Any]:
        date_value = trade_date or self.dates().get("latest")
        query = """
            SELECT signal_date, model, stock_code, stock_name, score, model_version, action_plan_json
            FROM strategy_signal_log
            WHERE signal_date = :d
              AND model IN ('laowang', 'stwg', 'ywcx', 'fhkq')
            ORDER BY model, score DESC
        """
        try:
            with self.engine.connect() as conn:
                source_rows = [dict(row) for row in conn.execute(sql_text(query), {"d": date_value}).mappings()]
        except SQLAlchemyError as exc:
            return {"columns": [], "rows": [], "empty_hint": f"strategy_signal_log unavailable: {exc}"}

        rows: List[Dict[str, Any]] = []
        for row in source_rows:
            plan = _parse_plan(row.get("action_plan_json"))
            skip_reason = plan.get("skip_reason") or []
            rows.append(
                {
                    "signal_date": row.get("signal_date"),
                    "model": str(row.get("model") or "").upper(),
                    "stock_code": row.get("stock_code"),
                    "stock_name": row.get("stock_name"),
                    "score": row.get("score"),
                    "action_state": plan.get("action_state") or ("skip_precheck" if skip_reason else "pending_t1_buy_check"),
                    "position_pct": plan.get("position_pct"),
                    "T+1_buy_condition": plan.get("t1_buy_condition"),
                    "T+2+_sell_condition": plan.get("t2_sell_condition"),
                    "skip_reason": "|".join(str(x) for x in skip_reason) if isinstance(skip_reason, list) else str(skip_reason or ""),
                }
            )
        columns = [
            "signal_date",
            "model",
            "stock_code",
            "stock_name",
            "score",
            "action_state",
            "position_pct",
            "T+1_buy_condition",
            "T+2+_sell_condition",
            "skip_reason",
        ]
        return {"columns": columns, "rows": rows, "trade_date": date_value, "empty_hint": "还没有运行 generate_trade_plan.py"}

    def positions(self) -> Dict[str, Any]:
        columns = [
            "trade_id",
            "signal_date",
            "model",
            "stock_code",
            "stock_name",
            "buy_date",
            "buy_price",
            "buy_shares",
            "sell_date",
            "sell_price",
            "pnl",
            "pnl_pct",
            "exit_reason",
            "trade_status",
        ]
        query = f"""
            SELECT {', '.join(columns)}
            FROM strategy_trade_journal
            ORDER BY COALESCE(buy_date, signal_date) DESC, updated_at DESC
            LIMIT 200
        """
        try:
            with self.engine.connect() as conn:
                rows = [dict(row) for row in conn.execute(sql_text(query)).mappings()]
        except SQLAlchemyError as exc:
            return {"columns": columns, "rows": [], "empty_hint": f"strategy_trade_journal unavailable: {exc}"}
        return {"columns": columns, "rows": rows}


class Handler(BaseHTTPRequestHandler):
    app: LaowangApp

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/":
                _send_html(self)
                return
            if path == "/favicon.ico":
                _send_favicon(self)
                return
            if path == "/api/dates":
                _send_json(self, self.app.dates())
                return
            if path == "/api/status":
                _send_json(self, self.app.status(_first(params, "trade_date")))
                return
            if path.startswith("/api/model/"):
                model = path.split("/")[-1]
                limit = int(_first(params, "limit", "300") or 300)
                _send_json(self, self.app.model_rows(model, _first(params, "trade_date"), limit=limit))
                return
            if path == "/api/plan":
                _send_json(self, self.app.plan(_first(params, "trade_date")))
                return
            if path == "/api/positions":
                _send_json(self, self.app.positions())
                return
            _send_json(self, {"error": "not found"}, status=404)
        except Exception as exc:  # noqa: BLE001
            logging.exception("request failed: %s", self.path)
            _send_json(self, {"error": str(exc)}, status=500)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the four-model LAOWANG read-only UI.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    engine = make_engine(resolve_db_url(args, cfg))
    Handler.app = LaowangApp(engine)
    server = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    logging.info("LAOWANG UI: http://%s:%d", args.host, int(args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("UI stopped")
    finally:
        server.server_close()
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
