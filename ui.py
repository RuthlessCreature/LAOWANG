#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui.py

简化版本地 Web UI：
- 仅从数据库读取 model_laowang_pool / model_ywcx_pool / model_stwg_pool / model_fhkq
- 不再触发任何计算任务
- 表格展示 status_tags 徽章
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
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import everyday
import everydayReview
import mock_backtest as backtest_model
import relay_strategy_model as relay_model


DEFAULT_DB = "data/stock.db"
FAVICON_PATH = Path(__file__).with_name("favicon.ico")

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

LAOWANG_COLS: Sequence[Tuple[str, str]] = [
    ("rank_no", "排名"),
    ("stock_code", "代码"),
    ("stock_name", "名称"),
    ("close", "收盘价"),
    ("support_level", "支撑位"),
    ("resistance_level", "压力位"),
    ("total_score", "总分"),
    ("status_tags", "标签"),
]

FHKQ_COLS: Sequence[Tuple[str, str]] = [
    ("stock_code", "代码"),
    ("stock_name", "名称"),
    ("consecutive_limit_down", "连板天数"),
    ("last_limit_down", "前一日跌停"),
    ("volume_ratio", "量能比"),
    ("amount_ratio", "成交额比"),
    ("open_board_flag", "开板标记"),
    ("liquidity_exhaust", "流动性衰竭"),
    ("fhkq_score", "FHKQ得分"),
    ("fhkq_level", "等级"),
]

YWCX_COLS: Sequence[Tuple[str, str]] = [
    ("rank_no", "排名"),
    ("stock_code", "代码"),
    ("stock_name", "名称"),
    ("close", "收盘价"),
    ("total_score", "总分"),
    ("weak_position_score", "位置衰弱"),
    ("volume_dry_score", "缩量枯竭"),
    ("low_volatility_score", "极弱波动"),
    ("status_tags", "标签"),
]

STWG_COLS: Sequence[Tuple[str, str]] = [
    ("rank_no", "排名"),
    ("stock_code", "代码"),
    ("stock_name", "名称"),
    ("close", "收盘价"),
    ("total_score", "总分"),
    ("stageB_compression_score", "缩量压缩"),
    ("breakout_confirmation_score", "突破确认"),
    ("status_tags", "标签"),
]

RELAY_DEFAULT_START_DATE = "2020-01-01"
RELAY_DEFAULT_BUFFER_DAYS = 45
RELAY_DEFAULT_TARGET_PRECISION = 0.90
RELAY_DEFAULT_MIN_SELECTED = 60
RELAY_DEFAULT_MAX_PER_DAY = 6
BACKTEST_DEFAULT_START_DATE = "2025-01-01"

def _build_backtest_variant_specs() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for profile in ("base", "expanded", "aggressive"):
        for spec in backtest_model.list_variant_specs(profile):
            name = str(spec.get("name") or "").strip()
            if not name:
                continue
            out[name] = {
                "hidden": float(spec["hidden"]),
                "epochs": float(spec["epochs"]),
                "lr": float(spec["lr"]),
                "weight_decay": float(spec["weight_decay"]),
                "target_gt": float(spec["target_gt"]),
                "seed": float(spec["seed"]),
            }
    return out


BACKTEST_VARIANT_SPECS: Dict[str, Dict[str, float]] = _build_backtest_variant_specs()
BACKTEST_RISK_PROFILES: Dict[str, Tuple[Optional[float], Optional[float], Optional[int], Optional[float]]] = {
    "off": (None, None, None, None),
    "balanced": (0.45, 0.24, 20, 0.09),
    "strict": (0.40, 0.28, 15, 0.08),
}
BEST_BACKTEST_CSV_CANDIDATES: Sequence[Path] = (
    Path("reports/iter_rounds/best/best_model_backtest_records.csv"),
    Path("best_model_backtest_records.csv"),
)
BEST_MODEL_META_PATH = Path("reports/iter_rounds/best/best_model.json")


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
      .relay-summary {
        padding: 12px 14px;
        border-bottom: 1px solid rgba(0,229,255,0.15);
        color: rgba(219,231,255,0.92);
        font-family: var(--mono);
        font-size: 12px;
        line-height: 1.7;
        white-space: normal;
        overflow-wrap: anywhere;
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="topbar">
        <div class="brand">
          <div class="title">爆头先锋</div>
          <div class="sub">老王 · 阳痿次新 · 粪海狂蛆 · 缩头乌龟 · 次日接力</div>
        </div>
        <div class="controls">
          <label for="tradeDate">最新交易日</label>
          <select id="tradeDate"></select>
          <label for="relayMinProb">接力阈值(分位)</label>
          <input id="relayMinProb" type="number" min="0.50" max="0.99" step="0.01" placeholder="自动" style="width:86px;background: rgba(10,14,22,0.95);border: 1px solid var(--line);color: var(--text);border-radius: 10px;padding: 8px 10px;font-family: var(--mono);" />
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

      <div class="panel">
        <div class="panel-header">
          <div>次日接力方案（排除一字板）</div>
          <div id="metaRelay"></div>
        </div>
        <div class="relay-summary" id="relaySummary">loading...</div>
        <div class="table-wrap">
          <table id="tableRelay">
            <thead></thead>
            <tbody></tbody>
          </table>
          <div class="empty" id="emptyRelay" style="display:none;"></div>
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

      function renderRelaySummary(payload) {
        const el = $("relaySummary");
        if (!el) return;
        if (!payload || payload.error) {
          el.textContent = payload && payload.error ? `接力模型异常: ${payload.error}` : "接力模型暂无数据";
          return;
        }
        const s = payload.summary || {};
        const parts = [
          `明日市场=${s.market_stage || "N/A"}`,
          `仓位建议=${s.position || "N/A"}`,
          `聚焦=${s.focus || "N/A"}`,
          `涨停=${s.limit_up_count ?? "N/A"}`,
          `跌停=${s.limit_down_count ?? "N/A"}`,
          `炸板率=${s.broken_rate ?? "N/A"}`,
          `最高连板=${s.max_board ?? "N/A"}`,
          `阈值=${s.threshold ?? "N/A"}`,
          `模型=${s.model_version || "N/A"}`,
          `过滤规则=${s.filter_rule || "N/A"}`,
          `入选=${s.selected_code || "-"}`,
          `原因=${s.selected_reason || "N/A"}`,
        ];
        el.textContent = parts.join(" | ");
      }


      async function apiGet(path) {
        const resp = await fetch(path, { cache: "no-store" });
        if (!resp.ok) throw new Error(await resp.text());
        return await resp.json();
      }

      function currentRelayMinProb() {
        const el = $("relayMinProb");
        if (!el) return "";
        const raw = (el.value || "").trim();
        if (!raw) return "";
        const v = Number(raw);
        if (!Number.isFinite(v)) return "";
        return String(Math.max(0.5, Math.min(0.99, v)).toFixed(2));
      }

      async function loadDate(dateStr) {
        if (!dateStr) return;
        setStatus("warn", "loading");
        $("metaYwcx").textContent = "";
        $("metaStwg").textContent = "";
        $("metaLaowang").textContent = "";
        $("metaFhkq").textContent = "";
        $("metaRelay").textContent = "";
        const st = await apiGet(`/api/status?trade_date=${encodeURIComponent(dateStr)}`);
        if (!st.has_stock_daily) {
          setStatus("err", "该日无K线数据");
          renderTable("tableLaowang", "emptyLaowang", { columns: [], rows: [], meta: { empty_hint: "no data" }});
          renderTable("tableYwcx", "emptyYwcx", { columns: [], rows: [], meta: { empty_hint: "no data" }});
          renderTable("tableFhkq", "emptyFhkq", { columns: [], rows: [], meta: { empty_hint: "no data" }});
          renderTable("tableStwg", "emptyStwg", { columns: [], rows: [], meta: { empty_hint: "no data" }});
          renderTable("tableRelay", "emptyRelay", { columns: [], rows: [], meta: { empty_hint: "no data" }});
          renderRelaySummary({ error: "该日无K线数据" });
          return;
        }
        const ok = st.laowang_rows > 0 || st.ywcx_rows > 0 || st.fhkq_rows > 0 || st.stwg_rows > 0;
        setStatus(
          ok ? "ok" : "warn",
          `老王:${st.laowang_rows} 阳痿次新:${st.ywcx_rows} 粪海狂蛆:${st.fhkq_rows} 缩头乌龟:${st.stwg_rows}`
        );

        const relayMinProb = currentRelayMinProb();
        const relayPath = relayMinProb
          ? `/api/relay/plan?trade_date=${encodeURIComponent(dateStr)}&min_prob=${encodeURIComponent(relayMinProb)}`
          : `/api/relay/plan?trade_date=${encodeURIComponent(dateStr)}`;
        const [lw, yw, fk, stwg, relay] = await Promise.all([
          apiGet(`/api/model/laowang?trade_date=${encodeURIComponent(dateStr)}`),
          apiGet(`/api/model/ywcx?trade_date=${encodeURIComponent(dateStr)}`),
          apiGet(`/api/model/fhkq?trade_date=${encodeURIComponent(dateStr)}`),
          apiGet(`/api/model/stwg?trade_date=${encodeURIComponent(dateStr)}`),
          apiGet(relayPath),
        ]);
        $("metaLaowang").textContent = `rows=${lw.meta && lw.meta.rows ? lw.meta.rows : lw.rows.length}`;
        $("metaYwcx").textContent = `rows=${yw.meta && yw.meta.rows ? yw.meta.rows : yw.rows.length}`;
        $("metaFhkq").textContent = `rows=${fk.meta && fk.meta.rows ? fk.meta.rows : fk.rows.length}`;
        $("metaStwg").textContent = `rows=${stwg.meta && stwg.meta.rows ? stwg.meta.rows : stwg.rows.length}`;
        $("metaRelay").textContent = `rows=${relay.meta && relay.meta.rows ? relay.meta.rows : (relay.rows ? relay.rows.length : 0)}`;
        renderTable("tableLaowang", "emptyLaowang", lw);
        renderTable("tableYwcx", "emptyYwcx", yw);
        renderTable("tableFhkq", "emptyFhkq", fk);
        renderTable("tableStwg", "emptyStwg", stwg);
        renderTable("tableRelay", "emptyRelay", relay);
        renderRelaySummary(relay);
      }

      async function boot() {
        const datesPayload = await apiGet("/api/dates");
        const sel = $("tradeDate");
        const relayInput = $("relayMinProb");
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
        if (relayInput) {
          relayInput.addEventListener("change", async () => {
            if (sel.value) await loadDate(sel.value);
          });
        }
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


def _translate_rows(rows: List[Dict[str, Any]], mapping: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    translated: List[Dict[str, Any]] = []
    for row in rows:
        new_row: Dict[str, Any] = {}
        for en, cn in mapping:
            val = row.get(en)
            if cn == "标签" and isinstance(val, list):
                new_row[cn] = val
            else:
                new_row[cn] = val
        translated.append(new_row)
    return translated


def _parse_status_tags(raw: Any) -> List[str]:
    values: List[str]
    if raw is None:
        values = []
    elif isinstance(raw, list):
        values = [str(x) for x in raw if str(x).strip()]
    elif isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:  # noqa: BLE001
            data = None
        if isinstance(data, list):
            values = [str(x) for x in data if str(x).strip()]
        else:
            cleaned = raw.strip()
            values = [cleaned] if cleaned else []
    else:
        values = []
    return [TAG_LABELS.get(v, v) for v in values]


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


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(z)
    denom = ex.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    return ex / denom


def _train_multiclass_mlp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = x.shape
    y_idx = y.astype(int)
    y_oh = np.eye(int(n_classes), dtype=float)[y_idx]

    w1 = rng.normal(scale=0.12, size=(d, hidden_size))
    b1 = np.zeros(hidden_size, dtype=float)
    w2 = rng.normal(scale=0.12, size=(hidden_size, n_classes))
    b2 = np.zeros(n_classes, dtype=float)

    for _ in range(max(1, int(epochs))):
        z1 = x @ w1 + b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ w2 + b2
        p = _softmax_np(z2)

        dz2 = (p - y_oh) / float(n)
        dw2 = a1.T @ dz2 + 2.0 * weight_decay * w2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ w2.T
        dz1 = da1 * (z1 > 0.0)
        dw1 = x.T @ dz1 + 2.0 * weight_decay * w1
        db1 = dz1.sum(axis=0)

        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2
    return w1, b1, w2, b2


def _predict_multiclass_mlp(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    h = np.maximum(x @ w1 + b1, 0.0)
    z = h @ w2 + b2
    p = _softmax_np(z)
    pred = np.argmax(p, axis=1).astype(np.int8)
    return pred, p


def _market_stage_label(row: pd.Series) -> int:
    lu = float(row.get("limit_up_count", 0.0) or 0.0)
    br = float(row.get("broken_rate", np.nan))
    mb = float(row.get("max_board", 0.0) or 0.0)
    if not math.isfinite(br):
        br = 0.5
    if lu >= 60 and br <= 0.30 and mb >= 4:
        return 2
    if lu >= 35 and br <= 0.42 and mb >= 2:
        return 1
    return 0


def _market_stage_from_label(label: int) -> Tuple[str, str, str]:
    x = int(label)
    if x >= 2:
        return "强趋势", "60%-80%", "优先2-4板换手票，允许2-3只并行。"
    if x == 1:
        return "修复/震荡", "30%-50%", "只做分时回封确认，单票不超过15%。"
    return "弱势轮动", "10%-20%", "减少出手，仅做高胜率首板/二板回封。"




def _safe_opt_float(v: Any) -> Optional[float]:
    s = str(v).strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return None
    try:
        x = float(s)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return None


def _safe_opt_int(v: Any) -> Optional[int]:
    s = str(v).strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _parse_best_cfg(cfg_text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "variant": "mlp_h56_e320_gt1",
        "alpha": 10,
        "sell_rule": "strong_hold_t3",
        "top_k": 3,
        "gap_min": None,
        "gap_max": None,
        "max_board": 3,
        "risk_profile": "off",
        "alloc_pct": 1.0,
    }
    parts = [p.strip() for p in str(cfg_text or "").split("/") if str(p).strip()]
    if parts:
        out["variant"] = parts[0]
    for p in parts[1:]:
        if p.startswith("a"):
            try:
                out["alpha"] = int(p[1:])
            except Exception:
                pass
            continue
        if p.startswith("k"):
            try:
                out["top_k"] = max(1, int(p[1:]))
            except Exception:
                pass
            continue
        if p.startswith("gap[") and p.endswith("]"):
            body = p[4:-1]
            left, right = (body.split(",", 1) + [""])[:2]
            out["gap_min"] = _safe_opt_float(left)
            out["gap_max"] = _safe_opt_float(right)
            continue
        if p.startswith("board<="):
            out["max_board"] = _safe_opt_int(p.split("<=", 1)[1])
            continue
        if p.startswith("risk="):
            out["risk_profile"] = p.split("=", 1)[1].strip() or "off"
            continue
        if p.startswith("alloc="):
            v = _safe_opt_float(p.split("=", 1)[1])
            if v is not None:
                out["alloc_pct"] = max(0.0, min(1.0, float(v)))
            continue
        out["sell_rule"] = p
    rp = BACKTEST_RISK_PROFILES.get(str(out["risk_profile"]), BACKTEST_RISK_PROFILES["off"])
    out["max_broken_rate"] = rp[0]
    out["min_red_rate"] = rp[1]
    out["max_limit_down"] = rp[2]
    out["max_pullback"] = rp[3]
    return out


def _choose_best_threshold_from_meta(meta: Dict[str, Any]) -> float:
    metrics = meta.get("metrics_by_threshold")
    if not isinstance(metrics, dict) or not metrics:
        return 0.75
    best_th = 0.75
    best_ret = float("-inf")
    for k, v in metrics.items():
        try:
            th = float(k)
            ret = float((v or {}).get("ret_pct", float("-inf")))
            if ret > best_ret:
                best_ret = ret
                best_th = th
        except Exception:
            continue
    return max(0.50, min(0.99, float(best_th)))


def _reason_to_text(reason: str) -> str:
    mp = {
        "ok": "通过过滤",
        "preopen_ok": "通过盘前过滤（开盘后再确认跳空与可成交性）",
        "threshold_blocked": "分位分数低于阈值",
        "board_blocked": "连板高度超限",
        "risk_blocked": "风险过滤未通过",
        "gap_blocked": "次日开盘跳空超限",
        "rule_blocked": "次日一字/涨停按规则不可买",
        "bad_buy_quote": "缺失次日开盘价/交易日（仅回测可见）",
        "no_candidate": "无候选",
        "missing_t1_trade_date": "缺少T+1交易日（仅回测可见）",
        "missing_t1_open": "缺少T+1开盘价（仅回测可见）",
        "missing_signal_close": "缺少信号日收盘价",
        "gap_below_min": "开盘跳空低于下限",
        "gap_above_max": "开盘跳空高于上限",
        "limit_up_unfilled": "涨停开盘按规则买不到",
        "broken_rate_high": "炸板率过高",
        "red_rate_low": "红盘率过低",
        "limit_down_high": "跌停数过高",
        "pullback_high": "日内回落过大",
    }
    return mp.get(str(reason or ""), str(reason or ""))

class AppContext:
    def __init__(
        self,
        engine: Engine,
        min_trade_date: Optional[str],
        job_runner: Optional["DailyJobRunner"],
        review_job_runner: Optional["ReviewJobRunner"] = None,
        *,
        relay_start_date: str = RELAY_DEFAULT_START_DATE,
        relay_buffer_days: int = RELAY_DEFAULT_BUFFER_DAYS,
        relay_target_precision: float = RELAY_DEFAULT_TARGET_PRECISION,
        relay_min_selected: int = RELAY_DEFAULT_MIN_SELECTED,
        relay_max_per_day: int = RELAY_DEFAULT_MAX_PER_DAY,
    ) -> None:
        self.engine = engine
        self.min_trade_date = min_trade_date  # YYYY-MM-DD
        self.job_runner = job_runner
        self.review_job_runner = review_job_runner
        self.relay_start_date = _normalize_iso_date(relay_start_date) or RELAY_DEFAULT_START_DATE
        self.relay_buffer_days = max(10, int(relay_buffer_days))
        self.relay_target_precision = float(relay_target_precision)
        self.relay_min_selected = max(10, int(relay_min_selected))
        self.relay_max_per_day = max(1, int(relay_max_per_day))
        self._relay_lock = threading.Lock()
        self._relay_cache: Optional[Dict[str, Any]] = None
        self._market_snapshot_cache: Dict[str, Dict[str, Any]] = {}

    def _latest_trade_date(self) -> Optional[str]:
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT MAX(date) FROM stock_daily")).fetchone()
        if not row or not row[0]:
            return None
        return str(row[0])

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
                "ywcx_rows": 0,
                "stwg_rows": 0,
                "fhkq_rows": 0,
            }
        with self.engine.connect() as conn:
            has_daily = conn.execute(text("SELECT 1 FROM stock_daily WHERE date = :d LIMIT 1"), {"d": trade_date}).fetchone()
            lw = conn.execute(text("SELECT COUNT(*) FROM model_laowang_pool WHERE trade_date = :d"), {"d": trade_date}).fetchone()[0]
            yw = conn.execute(text("SELECT COUNT(*) FROM model_ywcx_pool WHERE trade_date = :d"), {"d": trade_date}).fetchone()[0]
            stwg = conn.execute(text("SELECT COUNT(*) FROM model_stwg_pool WHERE trade_date = :d"), {"d": trade_date}).fetchone()[0]
            fk = conn.execute(text("SELECT COUNT(*) FROM model_fhkq WHERE trade_date = :d"), {"d": trade_date}).fetchone()[0]
        return {
            "trade_date": trade_date,
            "has_stock_daily": bool(has_daily),
            "laowang_rows": int(lw or 0),
            "ywcx_rows": int(yw or 0),
            "stwg_rows": int(stwg or 0),
            "fhkq_rows": int(fk or 0),
        }

    def fetch_laowang(self, trade_date: str) -> Dict[str, Any]:
        en_cols = [c for c, _ in LAOWANG_COLS]
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
        raw_rows: List[Dict[str, Any]] = []
        for row in rows:
            data = {}
            for idx, col in enumerate(en_cols):
                val = row[idx] if idx < len(row) else None
                if col == "status_tags":
                    data[col] = _parse_status_tags(val)
                else:
                    data[col] = val
            raw_rows.append(data)
        cn_rows = _translate_rows(raw_rows, LAOWANG_COLS)
        return {
            "columns": [cn for _, cn in LAOWANG_COLS],
            "rows": cn_rows,
            "meta": {"rows": len(cn_rows), "empty_hint": "0 rows"},
        }

    def fetch_fhkq(self, trade_date: str) -> Dict[str, Any]:
        en_cols = [c for c, _ in FHKQ_COLS]
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
        raw_rows = [{en_cols[idx]: row[idx] if idx < len(row) else None for idx in range(len(en_cols))} for row in rows]
        cn_rows = _translate_rows(raw_rows, FHKQ_COLS)
        return {
            "columns": [cn for _, cn in FHKQ_COLS],
            "rows": cn_rows,
            "meta": {"rows": len(cn_rows), "empty_hint": "0 rows"},
        }

    def fetch_ywcx(self, trade_date: str) -> Dict[str, Any]:
        en_cols = [c for c, _ in YWCX_COLS]
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT rank_no, stock_code, stock_name, close, total_score,
                           weak_position_score, volume_dry_score, low_volatility_score, status_tags
                    FROM model_ywcx_pool
                    WHERE trade_date = :d
                    ORDER BY rank_no ASC
                    """
                ),
                {"d": trade_date},
            ).fetchall()
        raw_rows: List[Dict[str, Any]] = []
        for row in rows:
            data = {}
            for idx, col in enumerate(en_cols):
                val = row[idx] if idx < len(row) else None
                if col == "status_tags":
                    data[col] = _parse_status_tags(val)
                else:
                    data[col] = val
            raw_rows.append(data)
        cn_rows = _translate_rows(raw_rows, YWCX_COLS)
        return {
            "columns": [cn for _, cn in YWCX_COLS],
            "rows": cn_rows,
            "meta": {"rows": len(cn_rows), "empty_hint": "0 rows"},
        }

    def fetch_stwg(self, trade_date: str) -> Dict[str, Any]:
        en_cols = [c for c, _ in STWG_COLS]
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT rank_no, stock_code, stock_name, close, total_score,
                           stageB_compression_score, breakout_confirmation_score, status_tags
                    FROM model_stwg_pool
                    WHERE trade_date = :d
                    ORDER BY rank_no ASC
                    """
                ),
                {"d": trade_date},
            ).fetchall()
        raw_rows: List[Dict[str, Any]] = []
        for row in rows:
            data = {}
            for idx, col in enumerate(en_cols):
                val = row[idx] if idx < len(row) else None
                if col == "status_tags":
                    data[col] = _parse_status_tags(val)
                else:
                    data[col] = val
            raw_rows.append(data)
        cn_rows = _translate_rows(raw_rows, STWG_COLS)
        return {
            "columns": [cn for _, cn in STWG_COLS],
            "rows": cn_rows,
            "meta": {"rows": len(cn_rows), "empty_hint": "0 rows"},
        }

    def _build_market_tomorrow_model(self, market_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if market_df is None or market_df.empty:
            return pd.DataFrame(), {"market_model_acc": 0.0, "market_samples": 0}

        mk = market_df.sort_values("date").copy()
        mk["stage_label"] = mk.apply(_market_stage_label, axis=1).astype(np.int8)
        mk["next_stage_label"] = mk["stage_label"].shift(-1)
        mk["next_gap"] = (pd.to_datetime(mk["date"].shift(-1)) - pd.to_datetime(mk["date"])).dt.days

        feat_cols = [
            "limit_up_count",
            "limit_down_count",
            "broken_rate",
            "max_board",
            "red_rate",
            "broken_red_rate",
            "amount_change5",
            "limit_up_count_lag1",
            "limit_up_count_lag2",
            "limit_up_count_lag3",
            "limit_down_count_lag1",
            "limit_down_count_lag2",
            "limit_down_count_lag3",
            "broken_rate_lag1",
            "broken_rate_lag2",
            "broken_rate_lag3",
            "max_board_lag1",
            "max_board_lag2",
            "max_board_lag3",
            "red_rate_lag1",
            "red_rate_lag2",
            "red_rate_lag3",
            "broken_red_rate_lag1",
            "broken_red_rate_lag2",
            "broken_red_rate_lag3",
            "amount_change5_lag1",
            "amount_change5_lag2",
            "amount_change5_lag3",
        ]

        train_df = mk[(mk["next_stage_label"].notna()) & (mk["next_gap"] >= 1) & (mk["next_gap"] <= 7)].copy()
        if train_df.empty:
            out = mk[["date"]].copy()
            out["today_stage"] = "N/A"
            out["tomorrow_stage"] = "N/A"
            out["tomorrow_position"] = "N/A"
            out["tomorrow_focus"] = "N/A"
            out["tomorrow_prob"] = np.nan
            return out, {"market_model_acc": 0.0, "market_samples": 0}

        train_x_raw = train_df[feat_cols].to_numpy(dtype=float)
        all_x_raw = mk[feat_cols].to_numpy(dtype=float)
        train_y = train_df["next_stage_label"].to_numpy(dtype=np.int8)

        train_x, all_x, _, _ = relay_model.fill_and_scale(train_x_raw, all_x_raw)
        w1, b1, w2, b2 = _train_multiclass_mlp(
            train_x,
            train_y,
            n_classes=3,
            hidden_size=24,
            epochs=900,
            learning_rate=0.04,
            weight_decay=1e-4,
            seed=43,
        )
        train_pred, _ = _predict_multiclass_mlp(train_x, w1, b1, w2, b2)
        all_pred, all_prob = _predict_multiclass_mlp(all_x, w1, b1, w2, b2)

        out = mk[["date", "stage_label"]].copy()
        out["today_stage"] = out["stage_label"].apply(lambda x: _market_stage_from_label(int(x))[0])
        out["tomorrow_stage_label_pred"] = all_pred.astype(np.int8)
        out["tomorrow_prob"] = all_prob.max(axis=1)
        out["tomorrow_stage"] = out["tomorrow_stage_label_pred"].apply(lambda x: _market_stage_from_label(int(x))[0])
        out["tomorrow_position"] = out["tomorrow_stage_label_pred"].apply(lambda x: _market_stage_from_label(int(x))[1])
        out["tomorrow_focus"] = out["tomorrow_stage_label_pred"].apply(lambda x: _market_stage_from_label(int(x))[2])

        return out[["date", "today_stage", "tomorrow_stage", "tomorrow_position", "tomorrow_focus", "tomorrow_prob"]], {
            "market_model_acc": float((train_pred == train_y).mean()),
            "market_samples": int(len(train_y)),
        }

    def _market_pred_for_date(self, cache: Dict[str, Any], trade_date: str) -> Dict[str, Any]:
        mdf = cache.get("market_pred")
        if mdf is None or len(mdf) == 0:
            return {
                "today_stage": "N/A",
                "tomorrow_stage": "N/A",
                "tomorrow_position": "N/A",
                "tomorrow_focus": "N/A",
                "tomorrow_prob": np.nan,
            }
        sub = mdf[mdf["date"] == trade_date]
        if sub.empty:
            return {
                "today_stage": "N/A",
                "tomorrow_stage": "N/A",
                "tomorrow_position": "N/A",
                "tomorrow_focus": "N/A",
                "tomorrow_prob": np.nan,
            }
        row = sub.iloc[0]
        return {
            "today_stage": str(row.get("today_stage", "N/A") or "N/A"),
            "tomorrow_stage": str(row.get("tomorrow_stage", "N/A") or "N/A"),
            "tomorrow_position": str(row.get("tomorrow_position", "N/A") or "N/A"),
            "tomorrow_focus": str(row.get("tomorrow_focus", "N/A") or "N/A"),
            "tomorrow_prob": float(row.get("tomorrow_prob", np.nan)),
        }

    def _relay_gate_for_day(
        self,
        *,
        base_threshold: float,
        market_pred: Dict[str, Any],
        manual_override: bool,
    ) -> Tuple[float, int, str]:
        # Dual-filter: layer-1 market-state gate + layer-2 stock confidence.
        if manual_override:
            return float(base_threshold), int(self.relay_max_per_day), "手动阈值：仅执行个股过滤"

        stage = str(market_pred.get("tomorrow_stage", "") or "")
        if stage == "弱势轮动":
            return min(0.99, float(base_threshold) + 0.05), max(1, self.relay_max_per_day // 2), "明日弱势：收紧阈值+减半出手"
        if stage == "强趋势":
            return max(0.50, float(base_threshold) - 0.02), int(self.relay_max_per_day), "明日强势：适度放宽阈值"
        return float(base_threshold), int(self.relay_max_per_day), "明日中性：维持基准阈值"

    def _build_relay_cache(self) -> Dict[str, Any]:
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT MAX(date) FROM stock_daily")).fetchone()
        end_date = str(row[0]) if row and row[0] else None
        if not end_date:
            raise ValueError("stock_daily is empty")

        best_meta = self._best_backtest_meta()
        best_start = _normalize_iso_date(str(best_meta.get("start_date", BACKTEST_DEFAULT_START_DATE))) or BACKTEST_DEFAULT_START_DATE
        try:
            seed_offset = int(float(best_meta.get("seed_offset", 0) or 0))
        except Exception:
            seed_offset = 0

        threshold_cfg: Dict[float, Dict[str, Any]] = {}
        metrics = best_meta.get("metrics_by_threshold") if isinstance(best_meta.get("metrics_by_threshold"), dict) else {}
        for k, v in metrics.items():
            try:
                th = max(0.50, min(0.99, float(k)))
            except Exception:
                continue
            cfg_text = str((v or {}).get("cfg") or "")
            parsed = _parse_best_cfg(cfg_text)
            threshold_cfg[float(th)] = parsed

        if not threshold_cfg:
            threshold_cfg[0.75] = _parse_best_cfg("mlp_h56_e320_gt1/a10/strong_hold_t3/k3/gap[None,None]/board<=3/risk=off/alloc=1.00")

        default_threshold = _choose_best_threshold_from_meta(best_meta)
        if threshold_cfg:
            default_threshold = min(threshold_cfg.keys(), key=lambda x: abs(float(x) - float(default_threshold)))

        train_start = RELAY_DEFAULT_START_DATE
        train_start_dt = dt.datetime.strptime(train_start, "%Y-%m-%d").date()
        load_start = (train_start_dt - dt.timedelta(days=60)).strftime("%Y-%m-%d")

        base = relay_model.load_base_frame(self.engine, load_start, end_date)
        full = relay_model.compute_flags_and_stock_features(base)
        market = relay_model.compute_market_features(full)
        market_pred, market_meta = self._build_market_tomorrow_model(market)

        full_pool, _, _ = relay_model.build_sample_frames(
            full_df=full,
            market_df=market,
            start_iso=train_start,
            end_iso=end_date,
            tp=0.02,
            sl=-0.01,
        )
        if full_pool.empty:
            raise ValueError("relay samples are empty")

        full_pool = full_pool.copy()
        full_pool["date"] = full_pool["date"].astype(str)
        full_pool["stock_code"] = full_pool["stock_code"].astype(str)
        full_pool = backtest_model._attach_future_fields(self.engine, full_pool, train_start)

        train_df = full_pool.copy()
        train_df = train_df[(train_df["next_open"].notna()) & (train_df["next2_close"].notna())].copy()
        train_df["next_open"] = pd.to_numeric(train_df["next_open"], errors="coerce")
        train_df["next2_close"] = pd.to_numeric(train_df["next2_close"], errors="coerce")
        train_df = train_df[(train_df["next_open"] > 0) & (train_df["next2_close"] > 0)].copy()
        train_df["t1_hold_ret"] = train_df["next2_close"] / train_df["next_open"] - 1.0
        if train_df.empty:
            raise ValueError("no train samples after T+1/T+2 target")

        feat_cols = backtest_model._feature_cols()
        x_train_raw = train_df[feat_cols].to_numpy(dtype=float)
        x_all_raw = full_pool[feat_cols].to_numpy(dtype=float)
        x_train, x_all, _, _ = relay_model.fill_and_scale(x_train_raw, x_all_raw)

        needed_variants = {str(cfg.get("variant")) for cfg in threshold_cfg.values()}
        if not needed_variants:
            needed_variants = {"mlp_h56_e320_gt1"}

        variant_probs: Dict[str, pd.Series] = {}
        variant_train_acc: Dict[str, float] = {}

        for variant in sorted(needed_variants):
            spec = BACKTEST_VARIANT_SPECS.get(variant)
            if not spec:
                continue
            y = (train_df["t1_hold_ret"] > float(spec["target_gt"])).astype(np.int8).to_numpy(np.int8)
            w1, b1, w2, b2 = relay_model.train_binary_mlp(
                x=x_train,
                y=y,
                hidden_size=int(spec["hidden"]),
                epochs=int(spec["epochs"]),
                learning_rate=float(spec["lr"]),
                weight_decay=float(spec["weight_decay"]),
                seed=int(spec["seed"]) + int(seed_offset),
            )
            train_pred, _ = relay_model.predict_binary_mlp(x_train, w1, b1, w2, b2)
            _, all_prob = relay_model.predict_binary_mlp(x_all, w1, b1, w2, b2)
            variant_probs[variant] = pd.Series(all_prob, index=full_pool.index)
            variant_train_acc[variant] = float((train_pred == y).mean()) if len(y) else 0.0

        if not variant_probs:
            raise ValueError("no valid backtest variant trained")

        base_pool = full_pool[full_pool["date"] >= best_start].copy()
        if base_pool.empty:
            base_pool = full_pool.copy()

        scored_by_key: Dict[str, pd.DataFrame] = {}
        ranked_by_key: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for th, cfg in threshold_cfg.items():
            variant = str(cfg.get("variant") or "")
            if variant not in variant_probs:
                # fallback to one available variant
                variant = next(iter(variant_probs.keys()))
                cfg["variant"] = variant
            alpha = int(cfg.get("alpha") or 10)
            key = f"{variant}/a{alpha}"
            cfg["score_key"] = key
            if key in scored_by_key:
                continue

            prob_series = variant_probs[variant]
            base_prob = prob_series.reindex(base_pool.index).to_numpy(dtype=float)
            score = backtest_model._score_from_raw_prob(base_prob, alpha=alpha)
            scored = base_pool.copy()
            scored["model_prob"] = base_prob
            scored["model_score"] = score
            # Keep both naming styles for downstream compatibility.
            scored["raw_prob"] = scored["model_prob"]
            scored["score"] = scored["model_score"]

            scored_by_key[key] = scored
            ranked_by_key[key] = backtest_model._build_ranked_by_date(scored)

        if not scored_by_key:
            raise ValueError("no scored relay pool generated")

        if market_pred is not None and not market_pred.empty:
            for k in list(scored_by_key.keys()):
                scored_by_key[k] = scored_by_key[k].merge(
                    market_pred[["date", "today_stage", "tomorrow_stage", "tomorrow_position", "tomorrow_focus", "tomorrow_prob"]],
                    on="date",
                    how="left",
                )
                ranked_by_key[k] = backtest_model._build_ranked_by_date(scored_by_key[k])

        rules = backtest_model._load_execution_rules("rules.txt")

        default_cfg = threshold_cfg.get(float(default_threshold)) or next(iter(threshold_cfg.values()))
        default_variant = str(default_cfg.get("variant") or "")
        fit_acc = float(variant_train_acc.get(default_variant, 0.0))

        return {
            "threshold_cfg": threshold_cfg,
            "default_threshold": float(default_threshold),
            "scored_by_key": scored_by_key,
            "ranked_by_key": ranked_by_key,
            "rules": rules,
            "fit_acc": fit_acc,
            "sample_rows": int(len(train_df)),
            "market_pred": market_pred,
            "market_model_acc": float(market_meta.get("market_model_acc", 0.0) or 0.0),
            "market_model_samples": int(market_meta.get("market_samples", 0) or 0),
            "score_mode": "mock_backtest_rank_score",
            "start_date": best_start,
            "end_date": end_date,
            "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _best_backtest_meta(self) -> Dict[str, Any]:
        # Legacy stub kept for compatibility with old relay-cache pipeline.
        return {}


    def _relay_cache_data(self) -> Dict[str, Any]:
        with self._relay_lock:
            latest = self._latest_trade_date()
            need_rebuild = self._relay_cache is None
            if self._relay_cache is not None and latest and self._relay_cache.get("end_date") != latest:
                need_rebuild = True
            if need_rebuild:
                self._relay_cache = self._build_relay_cache()
            return self._relay_cache

    def _fmt_pct(self, value: Any) -> str:
        try:
            f = float(value)
            if not math.isfinite(f):
                return "N/A"
            return f"{f * 100.0:.2f}%"
        except Exception:
            return "N/A"

    def _market_snapshot_from_stock_daily(self, trade_date: str) -> Optional[Dict[str, Any]]:
        cached = self._market_snapshot_cache.get(trade_date)
        if cached is not None:
            return dict(cached)

        try:
            d = dt.datetime.strptime(str(trade_date), "%Y-%m-%d").date()
        except Exception:
            return None

        lookback_start = (d - dt.timedelta(days=90)).strftime("%Y-%m-%d")
        try:
            base = relay_model.load_base_frame(self.engine, lookback_start, trade_date)
            if base is None or base.empty:
                return None
            full = relay_model.compute_flags_and_stock_features(base)
            market = relay_model.compute_market_features(full)
            if market is None or market.empty:
                return None
            market = market.copy()
            market["date"] = market["date"].astype(str)
            sub = market[market["date"] == trade_date]
            if sub.empty:
                return None
            row = sub.iloc[0]
        except Exception:
            return None

        limit_up_count = int(float(row.get("limit_up_count", 0.0) or 0.0))
        limit_down_count = int(float(row.get("limit_down_count", 0.0) or 0.0))
        broken_count = int(float(row.get("broken_count", 0.0) or 0.0))
        max_board = int(float(row.get("max_board", 0.0) or 0.0))
        broken_rate = float(row.get("broken_rate", np.nan))
        red_rate = float(row.get("red_rate", np.nan))
        broken_red_rate = float(row.get("broken_red_rate", np.nan))
        amount_change5 = float(row.get("amount_change5", np.nan))

        stage_row = pd.Series(
            {
                "limit_up_count": limit_up_count,
                "broken_rate": broken_rate,
                "max_board": max_board,
            }
        )
        stage_label = _market_stage_label(stage_row)
        stage_text, position, focus = _market_stage_from_label(stage_label)

        out = {
            "market_stage": stage_text,
            "position": position,
            "focus": focus,
            "limit_up_count": limit_up_count,
            "limit_down_count": limit_down_count,
            "broken_count": broken_count,
            "broken_rate": self._fmt_pct(broken_rate),
            "max_board": max_board,
            "red_rate": self._fmt_pct(red_rate),
            "broken_red_rate": self._fmt_pct(broken_red_rate),
            "amount_change5": self._fmt_pct(amount_change5),
        }
        self._market_snapshot_cache[trade_date] = dict(out)
        return out

    def _stage_from_day(self, day_df) -> Tuple[str, str, str]:
        if day_df is None or day_df.empty:
            return "N/A", "N/A", "N/A"
        return relay_model.market_stage(day_df.iloc[0])

    def _relay_trade_advices(self, prob: float, threshold: float, board_count: int) -> Dict[str, str]:
        premium = float(prob - threshold)
        board = int(board_count or 0)

        if board >= 4:
            open_plan = "高开>2%不追；仅平开~小低开观察"
            trigger = "10:00前首封回封且封单稳定≥5分钟才轻仓"
            risk = "回封后再炸板或跌破昨收-3%直接撤"
        elif premium >= 0.08:
            open_plan = "平开~高开3%可优先关注，>5%等回封"
            trigger = "分时放量过前高或回封后不炸板再介入"
            risk = "跌破昨收-3%止损；首封失败不恋战"
        elif premium >= 0.03:
            open_plan = "平开~高开2%可小仓试错"
            trigger = "分时二次放量转强或回封确认"
            risk = "弱于预期则先减仓，跌破昨收-3%止损"
        else:
            open_plan = "仅观察，不主动追高"
            trigger = "只有超预期强转强再考虑"
            risk = "不满足触发条件不出手"

        exit_plan = "T+1冲高2%-4%分批减仓；最晚T+2不转强清仓"
        return {
            "open_plan": open_plan,
            "trigger": trigger,
            "risk": risk,
            "exit": exit_plan,
        }

    def _relay_threshold(self, override: Optional[float] = None, *, default_threshold: float = 0.75) -> float:
        base = float(default_threshold)
        if override is None:
            return base
        try:
            v = float(override)
            if not math.isfinite(v):
                return base
            return max(0.50, min(0.99, v))
        except Exception:
            return base

    def _relay_live_reason(
        self,
        row: Dict[str, Any],
        *,
        threshold: float,
        max_board_filter: Optional[int],
        max_broken_rate: Optional[float],
        min_red_rate: Optional[float],
        max_limit_down: Optional[int],
        max_pullback: Optional[float],
    ) -> str:
        score = float(row.get("score", 0.0) or 0.0)
        if score < float(threshold):
            return "threshold_blocked"
        board = int(float(row.get("board_count", 0.0) or 0.0))
        if max_board_filter is not None and board > int(max_board_filter):
            return "board_blocked"

        ok_risk, risk_reason = backtest_model._passes_risk_filters(
            row=row,
            max_broken_rate=max_broken_rate,
            min_red_rate=min_red_rate,
            max_limit_down=max_limit_down,
            max_pullback=max_pullback,
        )
        if not ok_risk:
            return str(risk_reason or "risk_blocked")
        return "preopen_ok"

    def _select_relay_live_candidate(
        self,
        *,
        rows: Sequence[Dict[str, Any]],
        threshold: float,
        top_k: int,
        max_board_filter: Optional[int],
        max_broken_rate: Optional[float],
        min_red_rate: Optional[float],
        max_limit_down: Optional[int],
        max_pullback: Optional[float],
    ) -> Tuple[Optional[Dict[str, Any]], str, int]:
        if not rows:
            return None, "no_candidate", 0

        cap = min(len(rows), max(1, int(top_k)))
        has_above_threshold = False
        last_reason = "threshold_blocked"
        for idx, row in enumerate(rows[:cap], 1):
            reason = self._relay_live_reason(
                row,
                threshold=threshold,
                max_board_filter=max_board_filter,
                max_broken_rate=max_broken_rate,
                min_red_rate=min_red_rate,
                max_limit_down=max_limit_down,
                max_pullback=max_pullback,
            )
            if reason == "threshold_blocked":
                continue
            has_above_threshold = True
            if reason != "preopen_ok":
                last_reason = reason
                continue
            return row, reason, idx

        if not has_above_threshold:
            return None, "threshold_blocked", 0
        return None, last_reason, 0

    def _load_relay_rows(self, trade_date: str) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            db_rows = conn.execute(
                text(
                    """
                    SELECT trade_date, rank_no, stock_code, stock_name, model_prob, model_score, board_count, ret1, close,
                           next_trade_date, next_open, is_st, broken_rate, red_rate, limit_down_count, pullback,
                           limit_up_count, max_board, broken_count, amount_change5, model_version, alpha,
                           default_threshold, top_k, max_board_filter, gap_min, gap_max,
                           max_broken_rate_filter, min_red_rate_filter, max_limit_down_filter, max_pullback_filter,
                           risk_profile, updated_at
                    FROM model_relay_pool
                    WHERE trade_date = :d
                    ORDER BY rank_no ASC, model_score DESC
                    """
                ),
                {"d": trade_date},
            ).fetchall()

        rows: List[Dict[str, Any]] = []
        for row in db_rows:
            rows.append(
                {
                    "date": str(row[0]),
                    "rank_no": int(float(row[1] or 0.0)),
                    "stock_code": str(row[2] or ""),
                    "stock_name": str(row[3] or ""),
                    "raw_prob": float(row[4]) if row[4] is not None else 0.0,
                    "score": float(row[5]) if row[5] is not None else 0.0,
                    "board_count": int(float(row[6] or 0.0)),
                    "ret1": float(row[7]) if row[7] is not None else float("nan"),
                    "close": float(row[8]) if row[8] is not None else float("nan"),
                    "next_date": str(row[9] or ""),
                    "next_open": float(row[10]) if row[10] is not None else float("nan"),
                    "is_st": bool(int(float(row[11] or 0.0))),
                    "broken_rate": float(row[12]) if row[12] is not None else float("nan"),
                    "red_rate": float(row[13]) if row[13] is not None else float("nan"),
                    "limit_down_count": int(float(row[14] or 0.0)),
                    "pullback": float(row[15]) if row[15] is not None else float("nan"),
                    "limit_up_count": int(float(row[16] or 0.0)),
                    "max_board": int(float(row[17] or 0.0)),
                    "broken_count": int(float(row[18] or 0.0)),
                    "amount_change5": float(row[19]) if row[19] is not None else float("nan"),
                    "model_version": str(row[20] or ""),
                    "alpha": int(float(row[21] or 10.0)),
                    "default_threshold": float(row[22]) if row[22] is not None else 0.75,
                    "top_k": int(float(row[23] or 3.0)),
                    "max_board_filter": int(float(row[24])) if row[24] is not None else None,
                    "gap_min": float(row[25]) if row[25] is not None else None,
                    "gap_max": float(row[26]) if row[26] is not None else None,
                    "max_broken_rate_filter": float(row[27]) if row[27] is not None else None,
                    "min_red_rate_filter": float(row[28]) if row[28] is not None else None,
                    "max_limit_down_filter": int(float(row[29])) if row[29] is not None else None,
                    "max_pullback_filter": float(row[30]) if row[30] is not None else None,
                    "risk_profile": str(row[31] or "off"),
                    "updated_at": str(row[32] or ""),
                }
            )
        return rows

    def _relay_cfg_from_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {
                "model_version": "N/A",
                "alpha": 10,
                "default_threshold": 0.75,
                "top_k": 3,
                "max_board_filter": None,
                "gap_min": None,
                "gap_max": None,
                "max_broken_rate_filter": None,
                "min_red_rate_filter": None,
                "max_limit_down_filter": None,
                "max_pullback_filter": None,
                "risk_profile": "off",
                "updated_at": "",
            }
        r = rows[0]
        return {
            "model_version": str(r.get("model_version") or "N/A"),
            "alpha": int(r.get("alpha") or 10),
            "default_threshold": float(r.get("default_threshold") or 0.75),
            "top_k": max(1, int(r.get("top_k") or 1)),
            "max_board_filter": r.get("max_board_filter"),
            "gap_min": r.get("gap_min"),
            "gap_max": r.get("gap_max"),
            "max_broken_rate_filter": r.get("max_broken_rate_filter"),
            "min_red_rate_filter": r.get("min_red_rate_filter"),
            "max_limit_down_filter": r.get("max_limit_down_filter"),
            "max_pullback_filter": r.get("max_pullback_filter"),
            "risk_profile": str(r.get("risk_profile") or "off"),
            "updated_at": str(r.get("updated_at") or ""),
        }

    def _market_snapshot_for_day(self, trade_date: str) -> Dict[str, Any]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT pool_type, COUNT(*)
                    FROM daily_review_akshare_pool
                    WHERE trade_date = :d AND (is_st IS NULL OR is_st = 0)
                    GROUP BY pool_type
                    """
                ),
                {"d": trade_date},
            ).fetchall()
            cnt = {str(r[0]): int(r[1] or 0) for r in rows}

            mx_row = conn.execute(
                text(
                    """
                    SELECT MAX(limit_up_days)
                    FROM daily_review_xgb_limit_up
                    WHERE trade_date = :d AND (is_st IS NULL OR is_st = 0)
                    """
                ),
                {"d": trade_date},
            ).fetchone()
            max_board = int(float(mx_row[0] or 0.0)) if mx_row and mx_row[0] is not None else 0
            if max_board <= 0:
                fb_row = conn.execute(
                    text("SELECT MAX(board_count) FROM model_relay_pool WHERE trade_date = :d"),
                    {"d": trade_date},
                ).fetchone()
                max_board = int(float(fb_row[0] or 0.0)) if fb_row and fb_row[0] is not None else 0

            prev_row = conn.execute(
                text("SELECT MAX(date) FROM stock_daily WHERE date < :d"),
                {"d": trade_date},
            ).fetchone()
            prev_date = str(prev_row[0]) if prev_row and prev_row[0] else None

            red_rate = float("nan")
            broken_red_rate = float("nan")
            if prev_date:
                rr = conn.execute(
                    text(
                        """
                        SELECT
                            SUM(CASE WHEN cur.close > prev.close THEN 1 ELSE 0 END) AS red_cnt,
                            COUNT(*) AS total_cnt
                        FROM daily_review_akshare_pool p
                        JOIN stock_daily prev
                          ON prev.stock_code = p.stock_code AND prev.date = :prev_date
                        JOIN stock_daily cur
                          ON cur.stock_code = p.stock_code AND cur.date = :trade_date
                        WHERE p.trade_date = :prev_date
                          AND p.pool_type = 'limit_up'
                          AND (p.is_st IS NULL OR p.is_st = 0)
                        """
                    ),
                    {"prev_date": prev_date, "trade_date": trade_date},
                ).fetchone()
                if rr and rr[1]:
                    red_rate = float(rr[0] or 0.0) / float(rr[1])

                br = conn.execute(
                    text(
                        """
                        SELECT
                            SUM(CASE WHEN cur.close > prev.close THEN 1 ELSE 0 END) AS red_cnt,
                            COUNT(*) AS total_cnt
                        FROM daily_review_akshare_pool p
                        JOIN stock_daily prev
                          ON prev.stock_code = p.stock_code AND prev.date = :prev_date
                        JOIN stock_daily cur
                          ON cur.stock_code = p.stock_code AND cur.date = :trade_date
                        WHERE p.trade_date = :prev_date
                          AND p.pool_type = 'broken'
                          AND (p.is_st IS NULL OR p.is_st = 0)
                        """
                    ),
                    {"prev_date": prev_date, "trade_date": trade_date},
                ).fetchone()
                if br and br[1]:
                    broken_red_rate = float(br[0] or 0.0) / float(br[1])

            amt_rows = conn.execute(
                text(
                    """
                    SELECT date, SUM(amount) AS amt
                    FROM stock_daily
                    WHERE date <= :d
                    GROUP BY date
                    ORDER BY date DESC
                    LIMIT 6
                    """
                ),
                {"d": trade_date},
            ).fetchall()

        amount_change5 = float("nan")
        if amt_rows:
            amounts = [float(r[1] or 0.0) for r in amt_rows]
            if len(amounts) >= 2:
                today_amt = amounts[0]
                hist = amounts[1:6]
                denom = float(np.mean(hist)) if hist else float("nan")
                if math.isfinite(denom) and denom > 0:
                    amount_change5 = today_amt / denom - 1.0

        limit_up_count = int(cnt.get("limit_up", 0))
        limit_down_count = int(cnt.get("limit_down", 0))
        broken_count = int(cnt.get("broken", 0))

        if not rows and max_board <= 0 and limit_up_count == 0 and limit_down_count == 0 and broken_count == 0:
            fallback = self._market_snapshot_from_stock_daily(trade_date)
            if fallback is not None:
                return fallback

        broken_rate = float("nan")
        denom = limit_up_count + broken_count
        if denom > 0:
            broken_rate = float(broken_count) / float(denom)

        stage_row = pd.Series(
            {
                "limit_up_count": limit_up_count,
                "broken_rate": broken_rate,
                "max_board": max_board,
            }
        )
        stage_label = _market_stage_label(stage_row)
        stage_text, position, focus = _market_stage_from_label(stage_label)

        return {
            "market_stage": stage_text,
            "position": position,
            "focus": focus,
            "limit_up_count": limit_up_count,
            "limit_down_count": limit_down_count,
            "broken_count": broken_count,
            "broken_rate": self._fmt_pct(broken_rate),
            "max_board": max_board,
            "red_rate": self._fmt_pct(red_rate),
            "broken_red_rate": self._fmt_pct(broken_red_rate),
            "amount_change5": self._fmt_pct(amount_change5),
        }

    def fetch_relay_plan(self, trade_date: str, min_prob: Optional[float] = None) -> Dict[str, Any]:
        day_rows = self._load_relay_rows(trade_date)
        market = self._market_snapshot_for_day(trade_date)
        cfg = self._relay_cfg_from_rows(day_rows)

        threshold = self._relay_threshold(min_prob, default_threshold=float(cfg.get("default_threshold") or 0.75))
        selected_row, selected_reason, selected_rank = self._select_relay_live_candidate(
            rows=list(day_rows),
            threshold=float(threshold),
            top_k=int(cfg.get("top_k") or 1),
            max_board_filter=cfg.get("max_board_filter"),
            max_broken_rate=cfg.get("max_broken_rate_filter"),
            min_red_rate=cfg.get("min_red_rate_filter"),
            max_limit_down=cfg.get("max_limit_down_filter"),
            max_pullback=cfg.get("max_pullback_filter"),
        )

        summary = {
            **market,
            "threshold": f"{threshold:.2f}",
            "model_version": str(cfg.get("model_version") or "N/A"),
            "filter_rule": (
                f"topk={cfg.get('top_k')}, board<={cfg.get('max_board_filter')}, "
                f"gap=[{cfg.get('gap_min')},{cfg.get('gap_max')}] (open-check), risk={cfg.get('risk_profile')}"
            ),
            "selected_rank": int(selected_rank or 0),
            "selected_reason": _reason_to_text(selected_reason),
            "selected_code": str((selected_row or {}).get("stock_code", "")),
            "updated_at": cfg.get("updated_at"),
        }

        columns = ["代码", "名称", "评分", "原始概率", "连板", "当日涨跌", "是否入选", "过滤原因", "开盘预案", "买点触发", "不及预期", "止盈/离场"]
        if not day_rows:
            return {
                "summary": summary,
                "columns": columns,
                "rows": [],
                "meta": {"rows": 0, "empty_hint": "当日无接力候选（或尚未跑 everydayReview/scoring_relay）"},
            }

        cap = min(len(day_rows), max(int(self.relay_max_per_day), int(cfg.get("top_k") or 1)))
        rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(day_rows[:cap], 1):
            score = float(row.get("score", 0.0) or 0.0)
            raw_prob = float(row.get("raw_prob", 0.0) or 0.0)
            board = int(float(row.get("board_count", 0) or 0))

            reason_code = self._relay_live_reason(
                row,
                threshold=float(threshold),
                max_board_filter=cfg.get("max_board_filter"),
                max_broken_rate=cfg.get("max_broken_rate_filter"),
                min_red_rate=cfg.get("min_red_rate_filter"),
                max_limit_down=cfg.get("max_limit_down_filter"),
                max_pullback=cfg.get("max_pullback_filter"),
            )

            is_selected = bool(selected_row) and idx == int(selected_rank or 0)
            if is_selected:
                reason_text = "盘前预选入选：TopK 内首个通过过滤（开盘后再确认）"
            else:
                reason_text = _reason_to_text(reason_code)

            adv = self._relay_trade_advices(score, threshold, board)
            rows.append(
                {
                    "代码": str(row.get("stock_code", "")),
                    "名称": str(row.get("stock_name", "")),
                    "评分": self._fmt_pct(score),
                    "原始概率": self._fmt_pct(raw_prob),
                    "连板": board,
                    "当日涨跌": self._fmt_pct(row.get("ret1")),
                    "是否入选": "是" if is_selected else "否",
                    "过滤原因": reason_text,
                    "开盘预案": adv["open_plan"],
                    "买点触发": adv["trigger"],
                    "不及预期": adv["risk"],
                    "止盈/离场": adv["exit"],
                }
            )

        return {
            "summary": summary,
            "columns": columns,
            "rows": rows,
            "meta": {
                "rows": len(rows),
                "pool_rows": int(len(day_rows)),
                "empty_hint": "当日无高置信接力候选",
            },
        }

    def fetch_relay_holding(self, trade_date: str, codes: Sequence[str], min_prob: Optional[float] = None) -> Dict[str, Any]:
        if not codes:
            return {
                "columns": ["代码", "建议", "说明"],
                "rows": [],
                "meta": {"rows": 0, "empty_hint": "请传入 codes=000001,000002"},
            }

        day_rows = self._load_relay_rows(trade_date)
        cfg = self._relay_cfg_from_rows(day_rows)
        threshold = self._relay_threshold(min_prob, default_threshold=float(cfg.get("default_threshold") or 0.75))
        by_code = {str(r.get("stock_code")): r for r in day_rows}

        rows: List[Dict[str, Any]] = []
        for code in sorted({str(c).strip() for c in codes if str(c).strip()}):
            r = by_code.get(code)
            if not r:
                rows.append({"代码": code, "建议": "减仓/观察", "说明": "不在当日接力候选池（或已被一字板排除）"})
                continue

            score = float(r.get("score", 0.0) or 0.0)
            reason = self._relay_live_reason(
                r,
                threshold=float(threshold),
                max_board_filter=cfg.get("max_board_filter"),
                max_broken_rate=cfg.get("max_broken_rate_filter"),
                min_red_rate=cfg.get("min_red_rate_filter"),
                max_limit_down=cfg.get("max_limit_down_filter"),
                max_pullback=cfg.get("max_pullback_filter"),
            )

            if reason == "preopen_ok" and score >= threshold + 0.03:
                advice = "持有优先"
                note = "盘前信号仍强，开盘后确认不弱转再按计划分批止盈"
            elif reason == "preopen_ok":
                advice = "谨慎持有"
                note = "盘前可持有，开盘走弱或量价不符先减仓"
            else:
                advice = "优先减仓"
                note = f"过滤未通过：{_reason_to_text(reason)}"

            rows.append({
                "代码": code,
                "建议": advice,
                "说明": f"评分={self._fmt_pct(score)}，阈值={self._fmt_pct(threshold)}；{note}",
            })

        return {
            "columns": ["代码", "建议", "说明"],
            "rows": rows,
            "meta": {"rows": len(rows), "empty_hint": "0 rows"},
        }


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
        if path == "/favicon.ico":
            if FAVICON_PATH.exists():
                data = FAVICON_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/x-icon")
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                _text(self, "", status=404)
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
        if path == "/api/auto-status":
            everyday_status = self.app.job_runner.status() if self.app.job_runner else {"state": "none", "message": "disabled"}
            review_status = self.app.review_job_runner.status() if self.app.review_job_runner else {"state": "none", "message": "disabled"}
            _json(self, {"everyday": everyday_status, "everyday_review": review_status})
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
            if name == "ywcx":
                _json(self, self.app.fetch_ywcx(trade_date))
                return
            if name == "stwg":
                _json(self, self.app.fetch_stwg(trade_date))
                return
            if name == "fhkq":
                _json(self, self.app.fetch_fhkq(trade_date))
                return
            _json(self, {"error": "unknown model"}, status=404)
            return
        if path == "/api/relay/plan":
            trade_date = (q.get("trade_date") or [""])[0]
            trade_date = str(trade_date).strip()
            if not trade_date:
                _json(self, {"error": "trade_date required"}, status=400)
                return
            min_prob_raw = (q.get("min_prob") or [""])[0]
            try:
                min_prob = float(min_prob_raw) if str(min_prob_raw).strip() else None
            except Exception:
                min_prob = None
            _json(self, self.app.fetch_relay_plan(trade_date, min_prob=min_prob))
            return
        if path == "/api/relay/holding":
            trade_date = (q.get("trade_date") or [""])[0]
            trade_date = str(trade_date).strip()
            if not trade_date:
                _json(self, {"error": "trade_date required"}, status=400)
                return
            raw_codes = (q.get("codes") or [""])[0]
            codes = [s.strip() for s in str(raw_codes).split(",") if s.strip()]
            min_prob_raw = (q.get("min_prob") or [""])[0]
            try:
                min_prob = float(min_prob_raw) if str(min_prob_raw).strip() else None
            except Exception:
                min_prob = None
            _json(self, self.app.fetch_relay_holding(trade_date, codes, min_prob=min_prob))
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
        ywcx_workers: int,
        stwg_workers: int,
        fhkq_workers: int,
        laowang_top: int,
        laowang_min_score: float,
        ywcx_top: int,
        ywcx_min_score: float,
        stwg_top: int,
        stwg_min_score: float,
        review_job_runner: Optional["ReviewJobRunner"] = None,
        run_review_after_everyday: bool = True,
    ) -> None:
        self.config = config
        self.db_url = db_url
        self.db = db
        self.initial_start = initial_start
        self.get_workers = int(get_workers)
        self.laowang_workers = int(laowang_workers)
        self.ywcx_workers = int(ywcx_workers)
        self.stwg_workers = int(stwg_workers)
        self.fhkq_workers = int(fhkq_workers)
        self.laowang_top = int(laowang_top)
        self.laowang_min_score = float(laowang_min_score)
        self.ywcx_top = int(ywcx_top)
        self.ywcx_min_score = float(ywcx_min_score)
        self.stwg_top = int(stwg_top)
        self.stwg_min_score = float(stwg_min_score)
        self.review_job_runner = review_job_runner
        self.run_review_after_everyday = bool(run_review_after_everyday)
        self.hour, self.minute = self._parse_time(auto_time)
        now = dt.datetime.now()
        target = now.replace(hour=self.hour, minute=self.minute, second=0, microsecond=0)
        self.last_run_date: Optional[dt.date] = now.date() if now >= target else None
        self.state = "idle"
        self.message = "等待自动更新…"
        self.thread = threading.Thread(target=self._loop, name="DailyJobRunner", daemon=True)
        self.thread.start()

    def bind_review_runner(self, review_job_runner: Optional["ReviewJobRunner"]) -> None:
        self.review_job_runner = review_job_runner

    def _parse_time(self, s: str) -> Tuple[int, int]:
        try:
            parts = str(s or "17:35").split(":")
            h = max(0, min(23, int(parts[0])))
            m = max(0, min(59, int(parts[1]) if len(parts) > 1 else 0))
            return h, m
        except Exception:
            return 17, 35

    def _is_trading_day(self, day: dt.date) -> bool:
        return day.weekday() < 5

    def _run_job(self) -> None:
        self.state = "running"
        self.message = f"{dt.date.today().strftime('%Y%m%d')} 数据更新中…"
        logging.info("[auto] everyday start")
        try:
            everyday.run_once(
                config=self.config,
                db_url=self.db_url,
                db=self.db,
                initial_start_date=self.initial_start,
                getdata_workers=self.get_workers,
                laowang_workers=self.laowang_workers,
                ywcx_workers=self.ywcx_workers,
                stwg_workers=self.stwg_workers,
                fhkq_workers=self.fhkq_workers,
                laowang_top=self.laowang_top,
                laowang_min_score=self.laowang_min_score,
                ywcx_top=self.ywcx_top,
                ywcx_min_score=self.ywcx_min_score,
                stwg_top=self.stwg_top,
                stwg_min_score=self.stwg_min_score,
            )
            logging.info("[auto] everyday finished")
            self.state = "ok"
            self.message = f"{dt.date.today().strftime('%Y%m%d')} 数据更新完毕"
            if self.run_review_after_everyday and self.review_job_runner is not None:
                logging.info("[auto] chain trigger everydayReview after everyday")
                review_ok = self.review_job_runner.run_now(trigger_date=dt.date.today(), source="after_everyday")
                if review_ok:
                    self.message = f"{dt.date.today().strftime('%Y%m%d')} 数据更新完毕，relay 同步完成"
                else:
                    self.message = f"{dt.date.today().strftime('%Y%m%d')} 数据更新完毕，relay 同步失败"
        except Exception:
            logging.exception("[auto] everyday failed")
            self.state = "fail"
            self.message = f"{dt.date.today().strftime('%Y%m%d')} 数据更新失败"

    def _loop(self) -> None:
        while True:
            now = dt.datetime.now()
            target = now.replace(hour=self.hour, minute=self.minute, second=0, microsecond=0)
            if now >= target:
                today = now.date()
                if self.last_run_date != today:
                    if self._is_trading_day(today):
                        self._run_job()
                    else:
                        self.state = "idle"
                        self.message = f"{today.strftime('%Y%m%d')} 非交易日，自动任务跳过"
                    self.last_run_date = today
                target = target + dt.timedelta(days=1)
            sleep_sec = max(30.0, min(300.0, (target - now).total_seconds()))
            time.sleep(sleep_sec)

    def status(self) -> Dict[str, str]:
        return {"state": self.state, "message": self.message}


class ReviewJobRunner:
    def __init__(
        self,
        *,
        auto_time: str,
        config: Optional[str],
        db_url: Optional[str],
        db: Optional[str],
        initial_start: str,
        model_file: str,
        data_workers: int,
        xgb_timeout: int,
    ) -> None:
        self.config = config
        self.db_url = db_url
        self.db = db
        self.initial_start = initial_start
        self.model_file = model_file
        self.data_workers = int(data_workers)
        self.xgb_timeout = int(xgb_timeout)
        self.hour, self.minute = self._parse_time(auto_time)
        now = dt.datetime.now()
        target = now.replace(hour=self.hour, minute=self.minute, second=0, microsecond=0)
        self.last_run_date: Optional[dt.date] = now.date() if now >= target else None
        self.state = "idle"
        self.message = "等待 nightly relay 更新…"
        self._run_lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, name="ReviewJobRunner", daemon=True)
        self.thread.start()

    def _parse_time(self, s: str) -> Tuple[int, int]:
        try:
            parts = str(s or "21:00").split(":")
            h = max(0, min(23, int(parts[0])))
            m = max(0, min(59, int(parts[1]) if len(parts) > 1 else 0))
            return h, m
        except Exception:
            return 21, 0

    def _is_trading_day(self, day: dt.date) -> bool:
        return day.weekday() < 5

    def _run_job(self) -> bool:
        self.state = "running"
        self.message = f"{dt.date.today().strftime('%Y%m%d')} relay 更新中…"
        logging.info("[auto] everydayReview start")
        try:
            everydayReview.run_once(
                config=self.config,
                db_url=self.db_url,
                db=self.db,
                initial_start_date=self.initial_start,
                model_file=self.model_file,
                data_workers=self.data_workers,
                xgb_timeout=self.xgb_timeout,
                full_rebuild=False,
            )
            logging.info("[auto] everydayReview finished")
            self.state = "ok"
            self.message = f"{dt.date.today().strftime('%Y%m%d')} relay 更新完成"
            return True
        except Exception:
            logging.exception("[auto] everydayReview failed")
            self.state = "fail"
            self.message = f"{dt.date.today().strftime('%Y%m%d')} relay 更新失败"
            return False

    def run_now(self, *, trigger_date: Optional[dt.date] = None, source: str = "manual") -> bool:
        with self._run_lock:
            if self.state == "running":
                logging.info("[auto] everydayReview already running, skip source=%s", source)
                return False
            ok = self._run_job()
            if trigger_date is not None:
                self.last_run_date = trigger_date
            return ok

    def _loop(self) -> None:
        while True:
            now = dt.datetime.now()
            target = now.replace(hour=self.hour, minute=self.minute, second=0, microsecond=0)
            if now >= target:
                today = now.date()
                if self.last_run_date != today:
                    if self._is_trading_day(today):
                        self.run_now(trigger_date=today, source="schedule")
                    else:
                        self.state = "idle"
                        self.message = f"{today.strftime('%Y%m%d')} 非交易日，relay 任务跳过"
                        self.last_run_date = today
                target = target + dt.timedelta(days=1)
            sleep_sec = max(30.0, min(300.0, (target - now).total_seconds()))
            time.sleep(sleep_sec)

    def status(self) -> Dict[str, str]:
        return {"state": self.state, "message": self.message}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LAOWANG/FHKQ 浏览 UI（只读 + 自动任务）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--start-date", default=None, help="仅显示该日期及之后的交易日 (YYYYMMDD 或 YYYY-MM-DD)")
    parser.add_argument("--disable-auto-update", action="store_true", help="禁用 17:35 自动执行 everyday.py（仅交易日）")
    parser.add_argument("--auto-time", default="17:35", help="HH:MM（默认 17:35）")
    parser.add_argument("--auto-init-start-date", default="2000-01-01")
    parser.add_argument("--auto-getdata-workers", type=int, default=1)
    parser.add_argument("--auto-laowang-workers", type=int, default=16)
    parser.add_argument("--auto-ywcx-workers", type=int, default=16)
    parser.add_argument("--auto-stwg-workers", type=int, default=16)
    parser.add_argument("--auto-fhkq-workers", type=int, default=8)
    parser.add_argument("--auto-laowang-top", type=int, default=200)
    parser.add_argument("--auto-laowang-min-score", type=float, default=60.0)
    parser.add_argument("--auto-ywcx-top", type=int, default=120)
    parser.add_argument("--auto-ywcx-min-score", type=float, default=55.0)
    parser.add_argument("--auto-stwg-top", type=int, default=150)
    parser.add_argument("--auto-stwg-min-score", type=float, default=55.0)
    parser.add_argument("--disable-auto-review-update", action="store_true", help="禁用 21:00 自动执行 everydayReview.py（仅交易日）")
    parser.add_argument("--auto-review-time", default="21:00", help="HH:MM（默认 21:00）")
    parser.add_argument("--auto-review-init-start-date", default="2025-01-01")
    parser.add_argument("--auto-review-model-file", default="models/relay_model_active.npz")
    parser.add_argument("--auto-review-data-workers", type=int, default=1)
    parser.add_argument("--auto-review-xgb-timeout", type=int, default=10)
    parser.add_argument("--relay-start-date", default=RELAY_DEFAULT_START_DATE, help="接力模块起始日 YYYYMMDD / YYYY-MM-DD")
    parser.add_argument("--relay-buffer-days", type=int, default=RELAY_DEFAULT_BUFFER_DAYS, help="接力模块特征缓冲天数")
    parser.add_argument("--relay-target-precision", type=float, default=RELAY_DEFAULT_TARGET_PRECISION, help="接力模块目标命中率")
    parser.add_argument("--relay-min-selected", type=int, default=RELAY_DEFAULT_MIN_SELECTED, help="接力模块阈值最小样本数")
    parser.add_argument("--relay-max-per-day", type=int, default=RELAY_DEFAULT_MAX_PER_DAY, help="页面每个交易日最多展示接力票")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    db_target = resolve_db_target(args)
    engine = make_engine(db_target)

    scheduler: Optional[DailyJobRunner] = None
    review_scheduler: Optional[ReviewJobRunner] = None
    if not args.disable_auto_update:
        scheduler = DailyJobRunner(
            auto_time=args.auto_time,
            config=args.config,
            db_url=args.db_url,
            db=args.db,
            initial_start=args.auto_init_start_date,
            get_workers=int(args.auto_getdata_workers),
            laowang_workers=int(args.auto_laowang_workers),
            ywcx_workers=int(args.auto_ywcx_workers),
            stwg_workers=int(args.auto_stwg_workers),
            fhkq_workers=int(args.auto_fhkq_workers),
            laowang_top=int(args.auto_laowang_top),
            laowang_min_score=float(args.auto_laowang_min_score),
            ywcx_top=int(args.auto_ywcx_top),
            ywcx_min_score=float(args.auto_ywcx_min_score),
            stwg_top=int(args.auto_stwg_top),
            stwg_min_score=float(args.auto_stwg_min_score),
            review_job_runner=None,
            run_review_after_everyday=True,
        )
    if not args.disable_auto_review_update:
        review_scheduler = ReviewJobRunner(
            auto_time=args.auto_review_time,
            config=args.config,
            db_url=args.db_url,
            db=args.db,
            initial_start=args.auto_review_init_start_date,
            model_file=args.auto_review_model_file,
            data_workers=int(args.auto_review_data_workers),
            xgb_timeout=int(args.auto_review_xgb_timeout),
        )
    if scheduler and review_scheduler:
        scheduler.bind_review_runner(review_scheduler)

    min_date_iso = _normalize_iso_date(args.start_date)
    app = AppContext(
        engine,
        min_trade_date=min_date_iso,
        job_runner=scheduler,
        review_job_runner=review_scheduler,
        relay_start_date=args.relay_start_date,
        relay_buffer_days=int(args.relay_buffer_days),
        relay_target_precision=float(args.relay_target_precision),
        relay_min_selected=int(args.relay_min_selected),
        relay_max_per_day=int(args.relay_max_per_day),
    )
    httpd = ThreadingHTTPServer((str(args.host), int(args.port)), Handler)
    httpd.app = app  # type: ignore[attr-defined]
    url = f"http://{args.host}:{int(args.port)}"
    logging.info("UI running: %s", url)
    if scheduler:
        logging.info("everyday 自动任务每个交易日 %s 运行", args.auto_time)
    if review_scheduler:
        logging.info("everydayReview 自动任务每个交易日 %s 运行", args.auto_review_time)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
