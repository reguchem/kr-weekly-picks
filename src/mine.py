"""사용자가 페이지에서 기록한 실제 매매(docs/data/my_trades.json) → 보유·손익 계산"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

from . import config as C

KST = timezone(timedelta(hours=9))
TRADES = C.DOCS_DATA / "my_trades.json"
PORT = C.DOCS_DATA / "my_portfolio.json"
MAX_HOLD = 6

# 실제 체결가를 입력받으므로 슬리피지는 빼고 수수료·세금만 반영
BUY_FEE = C.FEE
SELL_FEE_STOCK = C.FEE + C.SELL_TAX
SELL_FEE_ETF = C.FEE


def load_trades() -> list[dict]:
    if not TRADES.exists():
        return []
    try:
        return json.loads(TRADES.read_text()).get("trades", [])
    except Exception:
        return []


def positions(trades: list[dict]):
    """평균단가 방식. 반환: pos{code:{...}}, realized, cash, invested_total"""
    pos, realized, cash = {}, 0.0, float(C.SEED)
    for t in sorted(trades, key=lambda x: (x.get("date", ""), x.get("ts", ""))):
        c, sh, px = t["code"], int(t["shares"]), float(t["price"])
        etf = t.get("type") == "etf"
        if t["side"] == "BUY":
            cost = sh * px * (1 + BUY_FEE)
            cash -= cost
            p = pos.setdefault(c, {"code": c, "name": t.get("name", c), "type": t.get("type", "stock"),
                                   "shares": 0, "cost": 0.0, "first": t["date"]})
            p["shares"] += sh
            p["cost"] += cost
        else:
            p = pos.get(c)
            if not p:
                continue
            sh = min(sh, p["shares"])
            avg = p["cost"] / p["shares"]
            proceeds = sh * px * (1 - (SELL_FEE_ETF if etf or p["type"] == "etf" else SELL_FEE_STOCK))
            realized += proceeds - avg * sh
            cash += proceeds
            p["shares"] -= sh
            p["cost"] -= avg * sh
            if p["shares"] <= 0:
                pos.pop(c)
    return pos, realized, cash


def write_portfolio(pos: dict, realized: float, cash: float, last_px: dict, asof: str):
    rows, value = [], 0.0
    for c, p in pos.items():
        last = last_px.get(c)
        avg = p["cost"] / p["shares"]
        v = p["shares"] * last if last else None
        value += v or p["cost"]
        rows.append({"code": c, "name": p["name"], "type": p["type"], "shares": p["shares"], "first": p["first"],
                     "avg": round(avg, 2), "last": last, "value": round(v) if v else None,
                     "pl": round(v - p["cost"]) if v else None,
                     "pl_pct": round(v / p["cost"] - 1, 4) if v else None})
    old = json.loads(PORT.read_text()) if PORT.exists() else {}
    hist = [h for h in old.get("history", []) if h["date"] != asof]
    total = value + cash
    hist.append({"date": asof, "total": round(total)})
    out = {"asof": asof, "updated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"), "seed": C.SEED,
           "positions": rows, "cash": round(cash), "stock_value": round(value), "total": round(total),
           "realized": round(realized), "ret": round(total / C.SEED - 1, 4), "history": hist[-400:]}
    PORT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out
