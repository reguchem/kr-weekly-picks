"""매 거래일 15:33(KST) 실행: (매매 요일이면) 이번 주 3종목 추천 + 전체 성과 갱신.

매매 요일은 백테스트가 고른 요일(docs/data/strategy.json의 weekday). 그 요일이 휴장이면 다음 거래일.
추천은 당일 종가 기준이며, 15:30~16:00 시간외 종가 매매로 '당일 종가'에 즉시 체결하는 것을 가정합니다.

실행: python -m src.weekly            # 매매 요일에만 신규 추천
      FORCE=1 python -m src.weekly    # 요일 무관하게 이번 주 추천 생성(이번 주 추천이 없을 때)
산출물: docs/data/recommendations.json, docs/data/performance.json
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from . import config as C
from .data import (get_universe, get_price_panel, get_ohlcv, get_fundamentals, get_sectors,
                   get_flow_panel, get_theme_etfs, SYNTHETIC)
from .etf import EtfFactors, etf_select, market_ok, ETF_SELL, ETF_BUY
from .notify import send_telegram, send_warnings
from . import llm
from .strategy import (STRATEGIES, Factors, score, select, select_cost_aware, expected, simulate,
                       sector_strength, KEEP_SECTORS, ROUND_TRIP)

KST = timezone(timedelta(hours=9))
REC_PATH = C.DOCS_DATA / "recommendations.json"
PERF_PATH = C.DOCS_DATA / "performance.json"
STRAT_PATH = C.DOCS_DATA / "strategy.json"
WARN_PATH = C.DOCS_DATA / "warnings.json"
BRIEF_PATH = C.DOCS_DATA / "brief.json"
SECTOR_CACHE = C.CACHE / "sectors.json"


def _load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _iso_week(d: pd.Timestamp):
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def _f(x, nd=4):
    return None if x is None or pd.isna(x) else round(float(x), nd)



# ───────────────────────── 시장 데이터 ─────────────────────────
def load_sectors(refresh: bool) -> dict:
    C.CACHE.mkdir(parents=True, exist_ok=True)
    if SECTOR_CACHE.exists() and not refresh:
        return json.loads(SECTOR_CACHE.read_text())
    m = get_sectors()
    if m:
        SECTOR_CACHE.write_text(json.dumps(m, ensure_ascii=False))
        return m
    return json.loads(SECTOR_CACHE.read_text()) if SECTOR_CACHE.exists() else {}


def _prev_picks(recs, kind):
    w = recs["weeks"][-1] if recs["weeks"] else None
    return [p for p in (w["picks"] if w else []) if p.get("type", "stock") == kind]


def build_market(recs: dict, last_day: pd.Timestamp, refresh_sectors: bool) -> dict:
    # 업종 분류(FDR→KRX 비로그인)는 KRX 로그인 조회보다 먼저: 비로그인 요청이 로그인 세션을 끊는 것으로 보임
    sectors = load_sectors(refresh_sectors)
    kospi = get_ohlcv("KOSPI", 300)["close"].loc[:last_day]
    uni = get_universe(C.UNIVERSE_SIZE)
    names = dict(zip(uni["code"], uni["name"]))
    for p in _prev_picks(recs, "stock"):
        names.setdefault(p["code"], p["name"])
    px = get_price_panel(list(names), 300)
    close = px["close"].loc[:last_day]
    # 테마 ETF
    etfs = get_theme_etfs(80)
    enames = dict(zip(etfs["code"], etfs["name"]))
    for p in _prev_picks(recs, "etf"):
        enames.setdefault(p["code"], p["name"])
    epx = get_price_panel(list(enames), 300)
    E = EtfFactors(epx["close"].loc[:last_day], epx["volume"].loc[:last_day])
    # 수급 (KRX 로그인 → 실패 시 네이버)
    start = str((last_day - pd.Timedelta(days=60)).date())
    flow = get_flow_panel(list(names), start, str(last_day.date()))
    if flow is None:
        flow = get_flow_panel(list(names), start, str(last_day.date()), recent_only=True)
    F = Factors(close, px["volume"].loc[:last_day], flow, sectors)
    t = close.index[-1]
    f = F.at(t)
    sec = sector_strength(f) if f["sector"].notna().any() else pd.DataFrame()
    ma = kospi.rolling(C.MARKET_MA).mean().iloc[-1] if C.MARKET_MA else np.nan
    return {"names": names, "enames": enames, "close": close, "F": F, "f": f, "t": t, "flow": flow,
            "sectors": sectors, "sector_table": sec, "E": E, "eclose": E.close, "kospi": kospi,
            "market_ok": market_ok(kospi, C.MARKET_MA, t),
            "kospi_now": float(kospi.iloc[-1]), "kospi_ma": None if np.isnan(ma) else float(ma)}


def _sector_rows(sec: pd.DataFrame, n=12):
    return [{"sector": s, "rank": int(r["rank"]), "ret_21": float(r["ret_21"]),
             "foreign_flow": None if pd.isna(r["flow"]) else round(float(r["flow"]), 3),
             "tilt": int(r.get("tilt", 0) or 0), "n": int(r["n"])} for s, r in sec.head(n).iterrows()]


# ───────────────────────── 신규 추천 ─────────────────────────
def make_recommendation(recs: dict, M: dict) -> dict | None:
    st = _load(STRAT_PATH, {})
    strat = st.get("strategy") if C.STRATEGY_MODE == "auto" and st.get("strategy") in STRATEGIES \
        else C.DEFAULT_STRATEGY
    fn, label, desc, buffer = STRATEGIES[strat]
    cal = st.get("calibration") if st.get("strategy") == strat else None
    names, enames, f, t = M["names"], M["enames"], M["f"], M["t"]
    prev_s = [p["code"] for p in _prev_picks(recs, "stock")]
    prev_e = [p["code"] for p in _prev_picks(recs, "etf")]
    prev_all = {p["code"]: p for k in ("stock", "etf") for p in _prev_picks(recs, k)}
    fund = get_fundamentals(t)

    # 1) 뉴스·정세 브리핑 (Claude API, 선택) → 업종 전망 보정
    brief = None
    if len(M["sector_table"]):
        hold_info = [{"name": prev_all[c]["name"], "sector": M["sectors"].get(c) or "ETF"} for c in prev_all]
        brief = llm.market_brief(str(t.date()), _sector_rows(M["sector_table"]), hold_info)
    ctx = {"sector_tilt": (brief or {}).get("sector_outlook") or {}}
    s = score(strat, f, fund, ctx)
    sec = ctx.get("sectors_table", M["sector_table"])
    sec_rank = {k: int(v) for k, v in sec["rank"].items()} if len(sec) else {}

    # 2) 종목당 금액·가격 상한
    equity = (_load(PERF_PATH, {}).get("value") or C.SEED)
    slot = equity / C.SLOTS
    rank, erank = {}, {}
    if not M["market_ok"]:
        stock_picks, etf_picks = [], []                       # 시장 필터: 전부 현금
    else:
        stock_picks = []
        if s is not None and len(s) >= C.N_PICKS:
            afford = set(s.index[f["close"].reindex(s.index) <= slot * C.MAX_PRICE_FRACTION])
            if cal and buffer > C.N_PICKS:
                stock_picks, rank = select_cost_aware(s, prev_s, C.N_PICKS, cal, ctx.get("new_ok"), afford)
            else:
                new_ok = afford if ctx.get("new_ok") is None else (ctx["new_ok"] & afford)
                stock_picks, rank = select(s, prev_s, C.N_PICKS, buffer, new_ok)
        esc = M["E"].scores(t)
        erank = {c: i + 1 for i, c in enumerate(esc.index)}
        eaff = set(esc.index[M["eclose"].loc[t].reindex(esc.index) <= slot * C.MAX_PRICE_FRACTION])
        etf_picks = etf_select(esc, prev_e, C.N_ETFS, C.N_ETFS * 2, M["E"], t, eaff) if len(esc) else []
    picks = stock_picks + etf_picks
    holdings = prev_s + prev_e
    n_new = sum(1 for c in picks if c not in holdings)
    prev_hold = {h["code"]: h for h in _load(PERF_PATH, {}).get("holdings", [])}
    cash_now = _load(PERF_PATH, {}).get("cash", C.SEED if not holdings else 0) or 0
    last_px = pd.concat([M["close"].iloc[-1], M["eclose"].loc[t]])
    freed = cash_now + sum((prev_hold.get(c, {}).get("shares") or 0) * float(last_px.get(c, 0) or 0)
                           * (1 - (ETF_SELL if c in prev_e else C.SELL_COST)) for c in holdings if c not in picks)
    per_new = (freed / n_new) if n_new else 0

    def shares_for(c, px):
        is_new = c not in holdings
        sh = int(per_new // (px * 1.002)) if (is_new and px) else (prev_hold.get(c, {}).get("shares"))
        return sh, (_f(sh * px, 0) if (sh and px) else None)

    def detail(c):
        row = f.loc[c] if c in f.index else None
        fd = fund.loc[c] if fund is not None and c in fund.index else None
        g = (lambda k, nd=4: _f(row[k], nd) if row is not None else None)
        sector = M["sectors"].get(c)
        px = float(row["close"]) if row is not None else None
        sh, amt = shares_for(c, px)
        return {
            "type": "stock", "code": c, "name": names.get(c, c), "rank": rank.get(c), "score": _f(s.get(c), 3),
            "action": "HOLD" if c in holdings else "BUY",
            "exp_week": _f(expected(cal, rank.get(c))) if cal else None, "shares": sh, "amount": amt,
            "sector": sector, "sector_rank": sec_rank.get(sector),
            "close": g("close", 0), "ret_21": g("ret_21"), "ret_63": g("ret_63"), "mom_12_1": g("mom_12_1"),
            "near_high": _f(row["close"] / row["high_252"]) if row is not None else None,
            "tv_ratio": _f(row["tv20"] / row["tv120"], 2) if row is not None else None,
            "flow_f": g("flow_f"), "flow_i": g("flow_i"),
            "per": _f(fd["PER"], 2) if fd is not None else None,
            "pbr": _f(fd["PBR"], 2) if fd is not None else None,
        }

    def edetail(c):
        r = M["E"].table(t).loc[c] if c in M["E"].close.columns else None
        px = float(r["close"]) if r is not None else None
        sh, amt = shares_for(c, px)
        return {"type": "etf", "code": c, "name": enames.get(c, c), "rank": erank.get(c),
                "action": "HOLD" if c in holdings else "BUY", "shares": sh, "amount": amt,
                "close": _f(px, 0), "ret_21": _f(r["r21"]) if r is not None else None,
                "ret_63": _f(r["r63"]) if r is not None else None,
                "near_high": _f(r["close"] / r["hi"]) if r is not None else None}

    nm = lambda c: names.get(c) or enames.get(c) or prev_all.get(c, {}).get("name", c)
    rec = {
        "week": _iso_week(t), "signal_date": str(t.date()),
        "strategy": strat, "strategy_label": label + f" + 테마 ETF {C.N_ETFS}개",
        "market_ok": M["market_ok"], "kospi_level": _f(M["kospi_now"], 2), "kospi_ma": _f(M["kospi_ma"], 2),
        "market_ma": C.MARKET_MA,
        "picks": [detail(c) for c in stock_picks] + [edetail(c) for c in etf_picks],
        "sells": [{"code": c, "name": nm(c), "type": "etf" if c in prev_e else "stock",
                   "rank": (erank if c in prev_e else rank).get(c), "sector": M["sectors"].get(c)}
                  for c in holdings if c not in picks],
        "top10": [{"code": c, "name": names.get(c, c), "score": _f(s[c], 3),
                   "sector": M["sectors"].get(c)} for c in (s.index[:10] if s is not None else [])],
        "sectors": _sector_rows(sec) if len(sec) else [],
        "llm_used": brief is not None, "equity": _f(equity, 0), "round_trip": ROUND_TRIP,
        "created_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
    }
    if brief:
        BRIEF_PATH.write_text(json.dumps({"date": str(t.date()), **brief}, ensure_ascii=False, indent=1))
    print(f"[추천] {rec['signal_date']} 시장필터 {'통과' if M['market_ok'] else '현금'}: " +
          ", ".join(f"{p['name']}({p['action']})" for p in rec["picks"]) +
          (" / 매도: " + ", ".join(x["name"] for x in rec["sells"]) if rec["sells"] else ""))
    return rec


# ───────────────────────── 매일 보유 종목 점검 ─────────────────────────
def check_warnings(recs: dict, M: dict) -> dict:
    """현재 보유(최근 추천) 종목·ETF의 하락 위험 신호. 매매는 하지 않고 정보만 표시."""
    out = {"date": str(M["t"].date()), "items": [], "market_ok": M["market_ok"],
           "kospi": _f(M["kospi_now"], 2), "kospi_ma": _f(M["kospi_ma"], 2), "market_ma": C.MARKET_MA}
    if not recs["weeks"]:
        return out
    weeks = recs["weeks"]
    F, E, t = M["F"], M["E"], M["t"]
    sec = M["sector_table"]
    sec_rank = {k: int(v) for k, v in sec["rank"].items()} if len(sec) else {}
    esc = E.scores(t)
    erank = {c: i + 1 for i, c in enumerate(esc.index)}
    items = []
    for p in weeks[-1]["picks"]:
        c, kind = p["code"], p.get("type", "stock")
        close = M["eclose"] if kind == "etf" else M["close"]
        buy_w = weeks[-1]
        for w in reversed(weeks):
            if any(q["code"] == c for q in w["picks"]):
                buy_w = w
            else:
                break
        bd = pd.Timestamp(buy_w["signal_date"])
        if c not in close or bd not in close.index:
            continue
        ser = close[c].loc[bd:].dropna()
        if ser.empty:
            continue
        last, buy, peak = float(ser.iloc[-1]), float(ser.iloc[0]), float(ser.max())
        dd, ret = last / peak - 1, last / buy - 1
        sig, strong = [], False
        if dd <= -0.10:
            sig.append(f"보유 중 고점 대비 {dd:.1%}")
            strong |= dd <= -0.15
        if kind == "etf":
            ma60 = E.ma60[c].loc[t] if c in E.ma60 else np.nan
            if not pd.isna(ma60) and last < ma60:
                sig.append(f"60일선 이탈 (60일선 {ma60:,.0f}원)")
            r = erank.get(c)
            if r is None or r > C.N_ETFS * 2:
                sig.append("테마 강도 순위 하락" + (f" ({r}위)" if r else " (후보 조건 탈락)"))
        else:
            ma20 = F.ma20[c].loc[t] if c in F.ma20 else np.nan
            if not pd.isna(ma20) and last < ma20:
                sig.append(f"20일선 이탈 (20일선 {ma20:,.0f}원)")
            if F.flow_f is not None and c in F.flow_f:
                ff = F.flow_f[c].loc[t]
                raw = M["flow"]["foreign"][c].dropna().tail(5) if c in M["flow"]["foreign"] else pd.Series(dtype=float)
                if len(raw) == 5 and (raw < 0).all():
                    sig.append("외국인 5일 연속 순매도")
                elif not pd.isna(ff) and ff < -0.05:
                    sig.append(f"외국인 20일 순매도 우위 ({ff:+.1%})")
            sr = sec_rank.get(M["sectors"].get(c))
            if sr and sr > KEEP_SECTORS:
                sig.append(f"업종({M['sectors'].get(c)}) 강도 {sr}위로 하락")
        items.append({"code": c, "name": p["name"], "type": kind, "sector": M["sectors"].get(c) or ("ETF" if kind == "etf" else None),
                      "buy_date": str(bd.date()), "buy": _f(buy, 0), "last": _f(last, 0), "ret": _f(ret), "dd": _f(dd),
                      "signals": sig, "strong": strong})
    news = llm.holdings_news(out["date"], items) if items else None
    for it in items:
        n = (news or {}).get(it["code"])
        if n and n["risk"] >= 1:
            it["signals"].append("뉴스: " + n["reason"])
            it["strong"] |= n["risk"] >= 2
        it["news"] = n
        if not M["market_ok"]:
            it["signals"].append(f"시장 필터: KOSPI가 {C.MARKET_MA}일선 아래 (다음 화요일 전량 현금화 대상)")
            it["strong"] = True
        k = len(it["signals"])
        it["level"] = "경고" if (it.pop("strong") or k >= 2) else ("주의" if k == 1 else "정상")
    out["items"] = items
    out["news_checked"] = news is not None
    return out


# ───────────────────────── 성과 계산 ─────────────────────────
def update_performance(recs: dict):
    weeks = recs["weeks"]
    if not weeks:
        PERF_PATH.write_text(json.dumps({"updated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
                                         "empty": True}, ensure_ascii=False))
        return
    first = pd.Timestamp(weeks[0]["signal_date"])
    count = int(np.busday_count(np.datetime64(first.date()), np.datetime64("today"))) + 90
    codes = sorted({p["code"] for w in weeks for p in w["picks"]} | {s["code"] for w in weeks for s in w["sells"]})
    px = get_price_panel(codes, count)
    k = get_ohlcv("KOSPI", count)
    close, open_ = px["close"], px["open"]
    days = close.index.union(k.index)
    close, open_ = close.reindex(days), open_.reindex(days)
    kc, ko = k["close"].reindex(days).ffill(), k["open"].reindex(days)
    ko = ko.where(ko > 0, kc)
    last = days[-1]

    def ret_from(i0, series_close, entry):
        out = {}
        for key, h in C.HORIZONS.items():
            j = i0 + h  # 진입일 종가 → h거래일 뒤 종가
            out[key] = _f(series_close.iloc[j] / entry - 1) if j < len(days) and entry else None
        out["now"] = _f(series_close.ffill().iloc[-1] / entry - 1) if entry else None
        return out

    for w in weeks:
        sig = pd.Timestamp(w["signal_date"])
        i0 = days.searchsorted(sig, side="left")   # 추천일 당일 종가 체결
        if i0 >= len(days):
            w["entry_date"] = None
            for p in w["picks"]:
                p["entry"], p["ret"] = None, None
            w["avg"], w["kospi"] = None, None
            continue
        w["entry_date"] = str(days[i0].date())
        for p in w["picks"]:
            c = p["code"]
            entry = close[c].iloc[i0] if c in close else np.nan
            entry = None if pd.isna(entry) else float(entry)
            p["entry"] = _f(entry, 0)
            p["ret"] = ret_from(i0, close[c], entry) if c in close else None
        kentry = float(kc.iloc[i0]) if not pd.isna(kc.iloc[i0]) else None
        w["kospi"] = ret_from(i0, kc, kentry)
        w["avg"] = {key: _f(np.mean(v)) if (v := [p["ret"][key] for p in w["picks"]
                                                  if p["ret"] and p["ret"][key] is not None]) else None
                    for key in list(C.HORIZONS) + ["now"]}

    # 구간별 요약 (추천 3종목 평균 vs KOSPI, 비용 제외)
    summary = {}
    for key in list(C.HORIZONS):
        pairs = [(w["avg"][key], w["kospi"][key]) for w in weeks
                 if w.get("avg") and w["avg"][key] is not None and w["kospi"][key] is not None]
        if pairs:
            a = np.array(pairs)
            summary[key] = {"n": len(a), "avg": _f(a[:, 0].mean()), "kospi_avg": _f(a[:, 1].mean()),
                            "win_rate": _f((a[:, 0] > 0).mean()), "beat_rate": _f((a[:, 0] > a[:, 1]).mean())}

    # 실전 포트폴리오: 추천대로 매주 교체 매매(비용 포함)
    sched = [(pd.Timestamp(w["signal_date"]), [p["code"] for p in w["picks"]]) for w in weeks]
    etf_codes = {p["code"] for w in weeks for p in w["picks"] if p.get("type") == "etf"} | \
                {x["code"] for w in weeks for x in w["sells"] if x.get("type") == "etf"}
    eq, trades = simulate(sched, open_, close, etf_codes=etf_codes, etf_buy=ETF_BUY, etf_sell=ETF_SELL)
    perf = {"updated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"), "last_date": str(last.date()),
            "summary": summary, "empty": eq.empty}
    if not eq.empty:
        b = kc.loc[eq.index[0]:] / kc.loc[eq.index[0]]
        perf["curve"] = {"dates": [str(d.date()) for d in eq.index],
                         "portfolio": [_f(x / C.SEED) for x in eq], "kospi": [_f(x) for x in b.reindex(eq.index)]}
        perf["total"] = _f(eq.iloc[-1] / C.SEED - 1)
        perf["value"] = _f(eq.iloc[-1], 0)
        perf["seed"] = C.SEED
        perf["kospi_total"] = _f(b.iloc[-1] - 1)
        perf["mdd"] = _f((eq / eq.cummax() - 1).min())
        perf["trades"] = [{**t, "date": str(t["date"].date()), "signal": str(t["signal"].date())} for t in trades]
        # 현재 보유
        held = {}
        for t in trades:
            if t["side"] == "BUY":
                held[t["code"]] = t
            else:
                held.pop(t["code"], None)
        names = {p["code"]: p["name"] for w in weeks for p in w["picks"]}
        cl = close.ffill().iloc[-1]
        perf["holdings"] = [{"code": c, "name": names.get(c, c), "buy_date": str(t["date"].date()),
                             "buy_price": _f(t["price"], 0), "last": _f(cl.get(c), 0),
                             "ret": _f(cl.get(c) / t["price"] - 1), "shares": t.get("shares"),
                             "value": _f(t.get("shares", 0) * cl.get(c, 0), 0)} for c, t in held.items()]
        perf["cash"] = _f(eq.iloc[-1] - sum(h["value"] or 0 for h in perf["holdings"]), 0)
    PERF_PATH.write_text(json.dumps(perf, ensure_ascii=False, indent=1))
    print(f"[성과] 누적 {perf.get('total')}, KOSPI {perf.get('kospi_total')}, 기준일 {perf['last_date']}")




def _wait_today_bar(today: pd.Timestamp, minutes=12) -> pd.Timestamp:
    """오늘 일봉(종가)이 올라올 때까지 대기. 끝까지 없으면 휴장으로 보고 마지막 거래일 반환."""
    deadline = time.time() + minutes * 60
    while True:
        last = get_ohlcv("KOSPI", 10).index[-1]
        if last.normalize() >= today or SYNTHETIC or time.time() > deadline:
            return last
        print(f"오늘({today.date()}) 종가 대기 중… 마지막 {last.date()}")
        time.sleep(60)


LEVEL = {"정상": 0, "주의": 1, "경고": 2}


def run():
    C.DOCS_DATA.mkdir(parents=True, exist_ok=True)
    recs = _load(REC_PATH, {"weeks": []})
    trade_wd = C.TRADE_WEEKDAY
    now = datetime.now(KST)
    today = pd.Timestamp(now.date())
    force = os.environ.get("FORCE") == "1"
    done = {w["week"] for w in recs["weeks"]}
    want = (now.weekday() >= trade_wd or force) and _iso_week(today) not in done
    last_day = _wait_today_bar(today)
    is_trading_day = last_day.normalize() == today
    if not is_trading_day and not force:
        print(f"오늘은 휴장 — 성과만 갱신 (마지막 거래일 {last_day.date()})")
        update_performance(recs)
        return

    M = build_market(recs, last_day, refresh_sectors=want)
    if want:
        rec = make_recommendation(recs, M)
        if rec:
            recs["weeks"].append(rec)
            send_telegram(rec)
    else:
        print(f"신규 추천 없음 (오늘 {C.WEEKDAYS_KR[now.weekday()] if now.weekday() < 5 else '주말'}, "
              f"매매요일 {C.WEEKDAYS_KR[trade_wd]}, 이번 주 추천 {'있음' if _iso_week(today) in done else '없음'})")

    # 매일 보유 종목 점검 → 새로 생기거나 심해진 경고만 알림
    prev = {i["code"]: i["level"] for i in _load(WARN_PATH, {}).get("items", [])}
    warn = check_warnings(recs, M)
    hist = _load(WARN_PATH, {}).get("history", [])
    changed = [i for i in warn["items"] if LEVEL[i["level"]] > LEVEL.get(prev.get(i["code"], "정상"), 0)]
    for i in changed:
        hist.append({"date": warn["date"], "code": i["code"], "name": i["name"], "level": i["level"],
                     "signals": i["signals"]})
    warn["history"] = hist[-200:]
    WARN_PATH.write_text(json.dumps(warn, ensure_ascii=False, indent=1))
    if changed:
        send_warnings(warn["date"], changed)
    for i in warn["items"]:
        print(f"[점검] {i['name']}: {i['level']} {i['signals']}")

    update_performance(recs)
    recs["updated_at"] = now.strftime("%Y-%m-%d %H:%M")
    REC_PATH.write_text(json.dumps(recs, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    run()
