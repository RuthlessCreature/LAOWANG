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

TAG_LABELS = {
    "TREND_UP": "趋势向上",
    "LOW_BASE": "低位平台",
    "PULLBACK": "回踩",
    "AT_SUPPORT": "支撑附近",
    "SPACE_OK": "空间充足",
    "NEAR_RESISTANCE": "临近压力",
    "RISK_FILTERED": "风险过滤",
    "STAGE_A_OK": "阶段A",
    "STAGE_B_COMPRESSED": "阶段B压缩",
    "VOLUME_DRY_UP": "缩量",
    "AT_PLATFORM": "平台支撑",
    "BREAKOUT_R": "突破R",
    "VOLUME_EXPANSION": "放量突破",
    "BROKEN_IPO": "破发",
    "NEAR_IPO_LOW": "接近低点",
    "VOLUME_DRY": "缩量",
    "LOW_VOL": "波动极弱",
    "JUST_ABOVE_MA5": "刚上MA5",
    "SMALL_FLOAT": "小流通",
}

DISPLAY_COLUMNS: Dict[str, Dict[str, str]] = {
    "laowang": {
        "rank_no": "排名",
        "stock_code": "代码",
        "stock_name": "名称",
        "close": "收盘价",
        "support_level": "支撑位",
        "resistance_level": "压力位",
        "total_score": "总分",
        "status_tags": "标签",
    },
    "ywcx": {
        "rank_no": "排名",
        "stock_code": "代码",
        "stock_name": "名称",
        "close": "收盘价",
        "total_score": "总分",
        "weak_position_score": "位置衰弱",
        "volume_dry_score": "缩量枯竭",
        "low_volatility_score": "极弱波动",
        "status_tags": "标签",
    },
    "fhkq": {
        "stock_code": "代码",
        "stock_name": "名称",
        "consecutive_limit_down": "连板天数",
        "last_limit_down": "前一日跌停",
        "volume_ratio": "量能比",
        "amount_ratio": "成交额比",
        "open_board_flag": "开板标记",
        "liquidity_exhaust": "流动性衰竭",
        "fhkq_score": "FHKQ得分",
        "fhkq_level": "等级",
    },
    "stwg": {
        "rank_no": "排名",
        "stock_code": "代码",
        "stock_name": "名称",
        "close": "收盘价",
        "total_score": "总分",
        "stageB_compression_score": "缩量压缩",
        "breakout_confirmation_score": "突破确认",
        "status_tags": "标签",
    },
}


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>粪海狂蛆</title>
    <link rel="icon" href="/favicon.ico" type="image/x-icon" />
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
      .brand { display: flex; flex-direction: column; gap: 4px; }
      .brand .title {
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 0.08em;
      }
      .brand .sub { font-size: 12px; color: var(--muted); }
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
      .table-wrap {
        width: 100%;
        overflow-x: auto;
        overflow-y: hidden;
        scrollbar-width: thin;
        scrollbar-color: rgba(0,229,255,0.35) rgba(255,255,255,0.04);
      }
      .table-wrap::-webkit-scrollbar { height: 8px; }
      .table-wrap::-webkit-scrollbar-track { background: rgba(255,255,255,0.04); }
      .table-wrap::-webkit-scrollbar-thumb {
        background: rgba(0,229,255,0.35);
        border-radius: 999px;
      }
      .table-wrap table {
        min-width: 100%;
        width: max-content;
      }
      table { border-collapse: collapse; }
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
      .wrap-cell {
        white-space: normal !important;
        overflow-wrap: anywhere;
        word-break: break-word;
        line-height: 1.5;
        min-width: 240px;
        max-width: 460px;
      }
      .tag-wrap { display: flex; flex-wrap: wrap; gap: 4px; }
      .tag-pill {
        border: 1px solid rgba(0,229,255,0.25);
        border-radius: 999px;
        padding: 2px 6px;
        font-size: 11px;
        color: rgba(219,231,255,0.92);
      }
      .footer {
        margin-top: 18px;
        text-align: center;
        font-family: var(--mono);
        font-size: 12px;
        color: var(--muted);
      }
      .footer .status-line { margin-bottom: 6px; }
      .footer .disclaimer { margin-top: 6px; opacity: 0.9; }
      .status-busy { color: var(--warn); }
      .status-ok { color: var(--ok); }
      .status-fail { color: var(--err); }
      @media (max-width: 768px) {
        .wrap { padding: 12px; }
        .topbar { flex-direction: column; align-items: flex-start; gap: 8px; }
        .controls { width: 100%; flex-wrap: wrap; }
        select { width: 100%; }
        thead th, tbody td { font-size: 11px; padding: 6px; }
        .panel { margin-top: 14px; }
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
        <div class="brand">
          <div class="title">爆头先锋</div>
          <div class="sub">老王 · 阳痿次新 · 粪海狂蛆 · 缩头乌龟</div>
        </div>
        <div class="controls">
          <label for="tradeDate">最新交易日</label>
          <select id="tradeDate"></select>
        </div>
        <div class="status">
          <span class="dot" id="statusDot"></span>
          <span id="statusText">loading</span>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">
          <div>老王 股票池</div>
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
          <div>阳痿次新 股票池</div>
          <div id="metaYwcx"></div>
        </div>
        <div class="table-wrap">
          <table id="tableYwcx">
            <thead></thead>
            <tbody></tbody>
          </table>
          <div class="empty" id="emptyYwcx" style="display:none;"></div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">
          <div>粪海狂蛆 连板博弈</div>
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
      <div class="panel">
        <div class="panel-header">
          <div>缩头乌龟 股票池</div>
          <div id="metaStwg"></div>
        </div>
        <div class="table-wrap">
          <table id="tableStwg">
            <thead></thead>
            <tbody></tbody>
          </table>
          <div class="empty" id="emptyStwg" style="display:none;"></div>
        </div>
      </div>




      <div class="footer">
        <div id="autoStatusMain" class="status-line status-busy">everyday: waiting...</div>
        <div id="autoStatusReview" class="status-line status-busy">everydayReview: waiting...</div>
        <div>Georgij Xe & his boys</div>
        <div class="disclaimer">免责声明：本页面内容仅供学习交流，不构成任何投资建议，盈亏自负。</div>
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
              const text = value === null || value === undefined ? "" : String(value);
              td.textContent = text;
              if (text.length > 24) td.classList.add("wrap-cell");
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
        $("metaYwcx").textContent = "";
        $("metaStwg").textContent = "";
        $("metaLaowang").textContent = "";
        $("metaFhkq").textContent = "";
        const st = await apiGet(`/api/status?trade_date=${encodeURIComponent(dateStr)}`);
        if (!st.has_stock_daily) {
          setStatus("err", "该日无K线数据");
          renderTable("tableLaowang", "emptyLaowang", { columns: [], rows: [], meta: { empty_hint: "no data" }});
          renderTable("tableYwcx", "emptyYwcx", { columns: [], rows: [], meta: { empty_hint: "no data" }});
          renderTable("tableFhkq", "emptyFhkq", { columns: [], rows: [], meta: { empty_hint: "no data" }});
          renderTable("tableStwg", "emptyStwg", { columns: [], rows: [], meta: { empty_hint: "no data" }});
          return;
        }
        const ok = st.laowang_rows > 0 || st.ywcx_rows > 0 || st.fhkq_rows > 0 || st.stwg_rows > 0;
        setStatus(
          ok ? "ok" : "warn",
          `老王:${st.laowang_rows} 阳痿次新:${st.ywcx_rows} 粪海狂蛆:${st.fhkq_rows} 缩头乌龟:${st.stwg_rows}`
        );

        const [lw, yw, fk, stwg] = await Promise.all([
          apiGet(`/api/model/laowang?trade_date=${encodeURIComponent(dateStr)}`),
          apiGet(`/api/model/ywcx?trade_date=${encodeURIComponent(dateStr)}`),
          apiGet(`/api/model/fhkq?trade_date=${encodeURIComponent(dateStr)}`),
          apiGet(`/api/model/stwg?trade_date=${encodeURIComponent(dateStr)}`),
        ]);
        $("metaLaowang").textContent = `rows=${lw.meta && lw.meta.rows ? lw.meta.rows : lw.rows.length}`;
        $("metaYwcx").textContent = `rows=${yw.meta && yw.meta.rows ? yw.meta.rows : yw.rows.length}`;
        $("metaFhkq").textContent = `rows=${fk.meta && fk.meta.rows ? fk.meta.rows : fk.rows.length}`;
        $("metaStwg").textContent = `rows=${stwg.meta && stwg.meta.rows ? stwg.meta.rows : stwg.rows.length}`;
        renderTable("tableLaowang", "emptyLaowang", lw);
        renderTable("tableYwcx", "emptyYwcx", yw);
        renderTable("tableFhkq", "emptyFhkq", fk);
        renderTable("tableStwg", "emptyStwg", stwg);
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

      function applyAutoStatus(elId, prefix, payload) {
        const el = $(elId);
        if (!el) return;
        const p = payload || {};
        el.textContent = `${prefix}: ${p.message || "waiting..."}`;
        el.classList.remove("status-busy", "status-ok", "status-fail");
        if (p.state === "ok") el.classList.add("status-ok");
        else if (p.state === "fail") el.classList.add("status-fail");
        else el.classList.add("status-busy");
      }

      async function pollAutoStatus() {
        try {
          const data = await apiGet("/api/auto-status");
          if (data && data.everyday !== undefined) {
            applyAutoStatus("autoStatusMain", "everyday", data.everyday);
            applyAutoStatus("autoStatusReview", "everydayReview", data.everyday_review);
          } else {
            applyAutoStatus("autoStatusMain", "everyday", data);
            applyAutoStatus("autoStatusReview", "everydayReview", { state: "none", message: "disabled" });
          }
        } catch (e) {
          // ignore
        }
      }
      
      boot().catch(e => {
        console.error(e);
        setStatus("err", "load error");
      });
      setInterval(pollAutoStatus, 5000);
      pollAutoStatus();
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


def _parse_status_tags(raw: Any) -> List[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        values = raw
    else:
        text = str(raw).strip()
        try:
            loaded = json.loads(text)
            values = loaded if isinstance(loaded, list) else [text]
        except json.JSONDecodeError:
            values = [part.strip() for part in text.replace(",", "|").split("|") if part.strip()]
    return [TAG_LABELS.get(str(item), str(item)) for item in values]


def _display_rows(model: str, columns: List[str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    mapping = DISPLAY_COLUMNS.get(model, {})
    display_columns = [mapping.get(col, col) for col in columns]
    display_rows: List[Dict[str, Any]] = []
    for row in rows:
        item: Dict[str, Any] = {}
        for col in columns:
            value = row.get(col)
            item[mapping.get(col, col)] = _parse_status_tags(value) if col == "status_tags" else value
        display_rows.append(item)
    return {"columns": display_columns, "rows": display_rows, "meta": {"rows": len(display_rows), "empty_hint": "0 rows"}}


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
            day_rows = conn.execute(sql_text("SELECT COUNT(*) FROM stock_daily WHERE date = :d"), {"d": date_value}).scalar()
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
            "has_stock_daily": int(day_rows or 0) > 0,
            "laowang_rows": pool_counts.get("laowang", 0),
            "ywcx_rows": pool_counts.get("ywcx", 0),
            "fhkq_rows": pool_counts.get("fhkq", 0),
            "stwg_rows": pool_counts.get("stwg", 0),
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
            translated = _display_rows(key, columns, [])
            translated["meta"] = {"rows": 0, "empty_hint": f"{cfg['table']} unavailable: {exc}"}
            return translated
        translated = _display_rows(key, columns, rows)
        translated["trade_date"] = date_value
        return translated

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
            if path == "/api/auto-status":
                _send_json(self, {"everyday": {"state": "disabled", "message": "未集成自动任务"}, "everyday_review": {"state": "disabled", "message": "未集成自动任务"}})
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
