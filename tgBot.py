#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tgBot.py

Telegram 机器人（长轮询 + 推送助手）。
功能：
- 轮询消息：随时查询“老王/阳痿次新/缩头乌龟/粪海狂蛆”或“全部/all”，返回当日最新股票池。
- 推送：`everyday.py` 完成后可调用 push 接口，向所有订阅者推送当日四个股票池。

说明：
- 机器人 Token 读取 `--bot-token` 或 `TG_BOT_TOKEN` 环境变量；缺失时直接退出。
- 订阅者列表保存在 `data/tg_subscribers.json`，只要主动和 bot 对话一次即可加入。
- CLI：`python tgBot.py --mode serve`（默认）运行轮询服务；`--mode push` 手动推送一次。
"""

from __future__ import annotations

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine


DEFAULT_DB = "data/stock.db"
DEFAULT_PROXY = "http://127.0.0.1:7890"
SUBSCRIBERS_PATH = Path("data/tg_subscribers.json")


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


def _default_token(token: Optional[str]) -> Optional[str]:
    if token and token.strip():
        return token.strip()
    env = os.getenv("TG_BOT_TOKEN")
    if env and env.strip():
        return env.strip()
    return None


def _resolve_proxy(value: Optional[str]) -> Optional[str]:
    candidate: Optional[str]
    if value is None:
        env = os.getenv("TG_BOT_PROXY")
        candidate = env if env is not None else DEFAULT_PROXY
    else:
        candidate = value
    s = str(candidate or "").strip()
    if not s or s.lower() in {"none", "no", "off"}:
        return None
    return s


class SubscriberStore:
    def __init__(self, path: Path = SUBSCRIBERS_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[int]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            logging.warning("[tg] 订阅者文件损坏，重置：%s", self.path)
            return []
        ids: List[int] = []
        for raw in payload:
            try:
                ids.append(int(raw))
            except (TypeError, ValueError):
                continue
        return sorted(set(ids))

    def save(self, ids: Iterable[int]) -> None:
        uniq = sorted({int(i) for i in ids})
        self.path.write_text(json.dumps(uniq, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, chat_id: int) -> None:
        ids = self.load()
        if chat_id in ids:
            return
        ids.append(chat_id)
        self.save(ids)
        logging.info("[tg] 新增订阅者：%s", chat_id)


def _fmt_float(val: Optional[float], digits: int = 2) -> str:
    if val is None:
        return "--"
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def _fmt_int(val: Optional[int]) -> str:
    if val is None:
        return "--"
    try:
        return str(int(val))
    except (TypeError, ValueError):
        return "--"


def _fmt_tags(tags: Optional[str]) -> str:
    if not tags:
        return ""
    text = str(tags).strip()
    if not text:
        return ""
    items: List[str] = []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            items = [str(s) for s in obj if s]
    except Exception:
        if "," in text:
            items = [part.strip() for part in text.split(",") if part.strip()]
        else:
            items = [text]
    if not items:
        return ""
    return f" |{'/'.join(items[:3])}|"


def _format_laowang(row: Dict[str, Any], idx: int) -> str:
    return (
        f"{idx:02d}. {row.get('stock_code','')} {row.get('stock_name','')}"
        f" 收:{_fmt_float(row.get('close'))}"
        f" 分:{_fmt_float(row.get('total_score'))}"
        f" 支撑:{_fmt_float(row.get('support_level'))}"
        f" 压力:{_fmt_float(row.get('resistance_level'))}"
        f"{_fmt_tags(row.get('status_tags'))}"
    )


def _format_ywcx(row: Dict[str, Any], idx: int) -> str:
    return (
        f"{idx:02d}. {row.get('stock_code','')} {row.get('stock_name','')}"
        f" 收:{_fmt_float(row.get('close'))}"
        f" 分:{_fmt_float(row.get('total_score'))}"
        f" 位置:{_fmt_float(row.get('weak_position_score'))}"
        f" 枯竭:{_fmt_float(row.get('volume_dry_score'))}"
        f"{_fmt_tags(row.get('status_tags'))}"
    )


def _format_stwg(row: Dict[str, Any], idx: int) -> str:
    return (
        f"{idx:02d}. {row.get('stock_code','')} {row.get('stock_name','')}"
        f" 收:{_fmt_float(row.get('close'))}"
        f" 分:{_fmt_float(row.get('total_score'))}"
        f" 压缩:{_fmt_float(row.get('stageB_compression_score'))}"
        f" 突破:{_fmt_float(row.get('breakout_confirmation_score'))}"
        f"{_fmt_tags(row.get('status_tags'))}"
    )


def _format_fhkq(row: Dict[str, Any], idx: int) -> str:
    return (
        f"{idx:02d}. {row.get('stock_code','')} {row.get('stock_name','')}"
        f" 连板:{_fmt_int(row.get('consecutive_limit_down'))}"
        f" 得分:{_fmt_int(row.get('fhkq_score'))}"
        f" 等级:{row.get('fhkq_level') or '--'}"
        f" 开板:{'是' if row.get('open_board_flag') else '否'}"
    )


PoolFormatter = Callable[[Dict[str, Any], int], str]


class PoolConfig:
    def __init__(
        self,
        *,
        key: str,
        title: str,
        table: str,
        formatter: PoolFormatter,
        order_by: str,
    ) -> None:
        self.key = key
        self.title = title
        self.table = table
        self.formatter = formatter
        self.order_by = order_by


POOL_CONFIGS: Dict[str, PoolConfig] = {
    "laowang": PoolConfig(
        key="laowang",
        title="老王股票池",
        table="model_laowang_pool",
        formatter=_format_laowang,
        order_by="rank_no ASC, total_score DESC",
    ),
    "ywcx": PoolConfig(
        key="ywcx",
        title="阳痿次新",
        table="model_ywcx_pool",
        formatter=_format_ywcx,
        order_by="rank_no ASC, total_score DESC",
    ),
    "stwg": PoolConfig(
        key="stwg",
        title="缩头乌龟",
        table="model_stwg_pool",
        formatter=_format_stwg,
        order_by="rank_no ASC, total_score DESC",
    ),
    "fhkq": PoolConfig(
        key="fhkq",
        title="粪海狂蛆",
        table="model_fhkq",
        formatter=_format_fhkq,
        order_by="fhkq_score DESC, stock_code ASC",
    ),
}

POOL_ALIASES: Dict[str, str] = {
    "老王": "laowang",
    "lw": "laowang",
    "laowang": "laowang",
    "粪海狂蛆": "fhkq",
    "狂蛆": "fhkq",
    "fhkq": "fhkq",
    "阳痿次新": "ywcx",
    "阳痿": "ywcx",
    "次新": "ywcx",
    "ywcx": "ywcx",
    "缩头乌龟": "stwg",
    "乌龟": "stwg",
    "stwg": "stwg",
    "plan": "__plan__",
    "计划": "__plan__",
    "计划单": "__plan__",
    "positions": "__positions__",
    "持仓": "__positions__",
    "positions": "__positions__",
}


def fetch_latest_pool(engine: Engine, cfg: PoolConfig, limit: int) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    limit = max(1, min(int(limit), 30))
    with engine.connect() as conn:
        latest = conn.execute(text(f"SELECT trade_date FROM {cfg.table} ORDER BY trade_date DESC LIMIT 1")).fetchone()
        if not latest or not latest[0]:
            return None, []
        trade_date = str(latest[0])
        query = text(
            f"SELECT * FROM {cfg.table} WHERE trade_date = :date ORDER BY {cfg.order_by} LIMIT {limit}"
        )
        rows = conn.execute(query, {"date": trade_date}).mappings().all()
        payload = [dict(row) for row in rows]
    return trade_date, payload


def build_pool_block(cfg: PoolConfig, trade_date: Optional[str], rows: List[Dict[str, Any]]) -> str:
    if not trade_date:
        return f"{cfg.title}\n暂无数据"
    if not rows:
        return f"{cfg.title}（{trade_date}）\n无结果"
    lines = [cfg.formatter(row, idx + 1) for idx, row in enumerate(rows)]
    return f"{cfg.title}（{trade_date}）\n" + "\n".join(lines)


def build_summary_text(engine: Engine, keys: Optional[Sequence[str]] = None, *, limit: int = 20) -> str:
    keys = list(keys) if keys else list(POOL_CONFIGS.keys())
    blocks = []
    for key in keys:
        cfg = POOL_CONFIGS[key]
        trade_date, rows = fetch_latest_pool(engine, cfg, limit)
        blocks.append(build_pool_block(cfg, trade_date, rows))
    return "\n\n".join(blocks)


def fetch_latest_trade_date(engine: Engine) -> Optional[str]:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT MAX(date) FROM stock_daily")).fetchone()
    return str(row[0]) if row and row[0] else None


def fetch_signal_plan(engine: Engine, trade_date: str) -> Dict[str, Any]:
    ordinary_query = """
        SELECT
            signal_date, model, stock_code, stock_name, score, model_version,
            action_plan_json,
            CASE
                WHEN json_extract(action_plan_json, '$.skip_reason') IS NOT NULL
                 AND json_extract(action_plan_json, '$.skip_reason') != '[]'
                THEN 'skip'
                ELSE 'pending_t1_buy_check'
            END AS action_state,
            ROUND(COALESCE(json_extract(action_plan_json, '$.position_pct'), 0), 4) AS planned_position_pct
        FROM strategy_signal_log
        WHERE signal_date = :td
          AND model IN ('laowang', 'stwg', 'ywcx', 'fhkq')
        ORDER BY model, planned_position_pct DESC
    """
    with engine.connect() as conn:
        ordinary_rows = [dict(r) for r in conn.execute(text(ordinary_query), {"td": trade_date}).mappings().all()]
    return {"ordinary": ordinary_rows}


def fetch_positions(engine: Engine, limit: int = 50) -> Dict[str, Any]:
    query = """
        SELECT
            trade_id, signal_date, model, stock_code, stock_name,
            buy_date, buy_price, buy_shares, buy_amount,
            sell_date, sell_price, pnl, pnl_pct,
            exit_reason, trade_status
        FROM strategy_trade_journal
        ORDER BY COALESCE(buy_date, signal_date) DESC
        LIMIT :lim
    """
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(text(query), {"lim": limit}).mappings().all()]
    open_pos = [r for r in rows if r.get("trade_status") != "CLOSED"]
    closed = [r for r in rows if r.get("trade_status") == "CLOSED"]
    pcts = [float(r["pnl_pct"]) for r in closed if r.get("pnl_pct") is not None]
    wins = sum(1 for p in pcts if p > 0)
    return {
        "open": open_pos,
        "closed": closed,
        "open_count": len(open_pos),
        "closed_count": len(closed),
        "wins": wins,
        "losses": len(pcts) - wins,
        "win_rate": wins / len(pcts) if pcts else 0.0,
        "net_pnl": round(sum(float(r["pnl"]) for r in closed if r.get("pnl") is not None), 2),
    }


def build_plan_text(engine: Engine) -> str:
    trade_date = fetch_latest_trade_date(engine)
    if not trade_date:
        return "No trade dates in stock_daily; run everyday.py first."
    data = fetch_signal_plan(engine, trade_date)
    ordinary = data.get("ordinary", [])
    if not ordinary:
        return f"({trade_date}) no plan rows; run generate_trade_plan.py first."

    lines = [f"Plan {trade_date}", ""]
    pending = [r for r in ordinary if r.get("action_state") == "pending_t1_buy_check"]
    skipped = [r for r in ordinary if r.get("action_state") == "skip"]
    if pending:
        lines.append(f"T+1 conditional buys ({len(pending)}):")
        for r in pending[:12]:
            pct = float(r.get("planned_position_pct") or 0)
            lines.append(
                f"  {r.get('model','').upper()} {r.get('stock_code')} {r.get('stock_name')} "
                f"score:{_fmt_float(r.get('score'))} pos:{pct*100:.1f}%"
            )
    if skipped:
        lines.append(f"Skipped ({len(skipped)}):")
        for r in skipped[:5]:
            lines.append(f"  {r.get('model','').upper()} {r.get('stock_code')} {r.get('stock_name')}")
        if len(skipped) > 5:
            lines.append(f"  ... {len(skipped)-5} more")
    return "\n".join(lines)

def build_positions_text(engine: Engine) -> str:
    data = fetch_positions(engine)
    open_pos = data.get("open", [])
    closed = data.get("closed", [])
    if not open_pos and not closed:
        return "暂无持仓记录，请先运行 record_trade.py 录入交易"
    lines = ["📊 持仓状态", ""]
    if open_pos:
        lines.append(f"🔵 持仓中（{data['open_count']} 条）：")
        for r in open_pos:
            lines.append(
                f"  {r.get('model','').upper()} {r.get('stock_code')} {r.get('stock_name')} "
                f"买:{_fmt_float(r.get('buy_price'))} 日期:{r.get('buy_date')}"
            )
    else:
        lines.append("🔵 持仓中：0 条")
    if closed:
        wr = data["win_rate"]
        net = data["net_pnl"]
        lines.append(
            f"📪 已平仓 {data['closed_count']} 条 "
            f"胜率:{wr*100:.0f}% 净盈亏:¥{net}"
        )
        for r in closed[:5]:
            pct = r.get("pnl_pct") or 0
            emoji = "✅" if pct > 0 else "❌"
            lines.append(
                f"  {emoji} {r.get('model','').upper()} {r.get('stock_code')} "
                f"买:{_fmt_float(r.get('buy_price'))} 卖:{_fmt_float(r.get('sell_price'))} "
                f"{pct*100:+.1f}% 原因:{r.get('exit_reason','')}"
            )
    return "\n".join(lines)


class TelegramClient:
    def __init__(self, token: str, *, proxy: Optional[str]) -> None:
        self.token = token
        self.base = f"https://api.telegram.org/bot{token}"
        self.session = requests.Session()
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})
            logging.info("[tg] 使用代理：%s", proxy)

    def send_message(self, chat_id: int, text: str, *, retry: int = 2) -> None:
        payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
        for attempt in range(retry + 1):
            try:
                resp = self.session.post(f"{self.base}/sendMessage", json=payload, timeout=20)
                data = resp.json()
                if not data.get("ok"):
                    raise RuntimeError(f"sendMessage failed: {data}")
                return
            except Exception as exc:  # noqa: BLE001
                if attempt >= retry:
                    logging.error("[tg] sendMessage 失败：chat=%s err=%s", chat_id, exc)
                    return
                logging.warning("[tg] sendMessage 重试中：chat=%s err=%s", chat_id, exc)
                time.sleep(2)

    def get_updates(self, offset: Optional[int], timeout: int = 30) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset is not None:
            params["offset"] = offset
        resp = self.session.get(f"{self.base}/getUpdates", params=params, timeout=timeout + 10)
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"getUpdates failed: {data}")
        return data.get("result", [])


class BotService:
    def __init__(
        self,
        *,
        engine: Engine,
        token: str,
        proxy: Optional[str],
        store: SubscriberStore,
        limit: int,
        poll_timeout: int = 30,
    ) -> None:
        self.engine = engine
        self.client = TelegramClient(token, proxy=proxy)
        self.store = store
        self.limit = max(1, min(limit, 20))
        self.poll_timeout = poll_timeout
        self.help_text = (
            "可用命令：\n"
            "- 老王 / 阳痿次新 / 缩头乌龟 / 粪海狂蛆\n"
            "- all / 全部 —— 返回四个股票池\n"
            "- plan / 计划 —— 查看次日交易计划\n"
            "- positions / 持仓 —— 查看当前持仓\n"
            "- /start —— 订阅推送"
        )

    def run(self) -> None:
        logging.info("[tg] BotService 启动（长轮询）")
        offset: Optional[int] = None
        while True:
            try:
                updates = self.client.get_updates(offset, timeout=self.poll_timeout)
            except Exception:
                logging.exception("[tg] getUpdates 失败，等待 5 秒")
                time.sleep(5)
                continue
            for upd in updates:
                offset = upd["update_id"] + 1
                self._handle_update(upd)

    def _handle_update(self, update: Dict[str, Any]) -> None:
        message = update.get("message") or update.get("edited_message")
        if not message:
            return
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None:
            return
        text = (message.get("text") or "").strip()
        if not text:
            return
        logging.info("[tg] 收到消息 chat=%s text=%s", chat_id, text)
        self.store.add(int(chat_id))
        response = self._handle_text(text)
        if response:
            self.client.send_message(chat_id=int(chat_id), text=response)

    def _handle_text(self, text: str) -> str:
        normalized = text.strip().lower().lstrip("/")
        if normalized in {"start", "help", "开始", "帮助"}:
            return self.help_text
        if normalized in {"all", "全部"}:
            return build_summary_text(self.engine, limit=self.limit)
        if normalized == "__plan__":
            return build_plan_text(self.engine)
        if normalized == "__positions__":
            return build_positions_text(self.engine)
        if normalized in POOL_ALIASES:
            key = POOL_ALIASES[normalized]
            if key.startswith("__"):
                return "未知指令"
            cfg = POOL_CONFIGS[key]
            trade_date, rows = fetch_latest_pool(self.engine, cfg, self.limit)
            return build_pool_block(cfg, trade_date, rows)
        return "未知指令，可输入“老王/阳痿次新/缩头乌龟/粪海狂蛆”或 “all/全部”。"


def push_latest_pools(
    *,
    engine: Engine,
    token: Optional[str],
    store: SubscriberStore,
    limit: int = 15,
    proxy: Optional[str] = None,
) -> None:
    real_token = _default_token(token)
    if not real_token:
        logging.info("[tg] 未配置机器人 Token，跳过推送；请设置 --bot-token 或 TG_BOT_TOKEN")
        return
    subscribers = store.load()
    if not subscribers:
        logging.info("[tg] 没有订阅者，跳过推送")
        return
    text = build_summary_text(engine, limit=limit)
    proxy_url = _resolve_proxy(proxy)
    client = TelegramClient(real_token, proxy=proxy_url)
    for chat_id in subscribers:
        client.send_message(chat_id=chat_id, text=text)
    logging.info("[tg] 推送完成，覆盖 %d 个订阅者", len(subscribers))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telegram Bot（股票池推送与查询）")
    parser.add_argument("--config", default=None, help="config.ini 路径")
    parser.add_argument("--db-url", default=None, help="自定义 SQLAlchemy URL")
    parser.add_argument("--db", default=None, help="SQLite 文件路径")
    parser.add_argument("--bot-token", default=None, help="覆盖 TG_BOT_TOKEN")
    parser.add_argument("--subscriber-file", default=str(SUBSCRIBERS_PATH), help="订阅者文件（JSON）")
    parser.add_argument("--limit", type=int, default=15, help="每个股票池返回条数")
    parser.add_argument("--mode", choices=["serve", "push"], default="serve", help="serve=长轮询, push=立即推送一次")
    parser.add_argument("--poll-timeout", type=int, default=30, help="getUpdates 超时时间（秒）")
    parser.add_argument("--proxy", default=DEFAULT_PROXY, help="HTTP/HTTPS 代理，例如 http://127.0.0.1:7890；传空字符串关闭")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db_target = resolve_db_target(args)
    engine = make_engine(db_target)
    token = _default_token(args.bot_token)
    if not token:
        logging.error("未提供 bot token，可用 --bot-token 或 TG_BOT_TOKEN")
        return 1
    proxy_url = _resolve_proxy(args.proxy)
    store = SubscriberStore(Path(args.subscriber_file))

    if args.mode == "push":
        push_latest_pools(engine=engine, token=token, store=store, limit=int(args.limit), proxy=proxy_url)
        return 0

    srv = BotService(
        engine=engine,
        token=token,
        proxy=proxy_url,
        store=store,
        limit=int(args.limit),
        poll_timeout=int(args.poll_timeout),
    )
    srv.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
