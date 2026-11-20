"""개선안 비교 실험 (5개 요일 평균 + 2022년 이후 검증)

실행: python -m src.experiments   → docs/data/experiments.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from . import config as C
from .data import get_universe, get_price_panel, get_ohlcv, get_sectors, get_theme_etfs, check_krx_login
from .backtest import load_flow, load_fundamentals
from .etf import (EtfFactors, etf_select, market_ok, ETF_BUY, ETF_SELL, kospi_rets, bear_ok_stocks,
                  bear_ok_etfs)
from .strategy import (Factors, score, select_cost_aware, fit_calibration, simulate, weekly_signal_dates,
                       signal_dates, metrics)

KST = timezone(timedelta(hours=9))


def run():
    t0 = pd.Timestamp(C.BACKTEST_START)
    is_end = pd.Timestamp(C.IN_SAMPLE_END)
    count = int(np.busday_count(np.datetime64(C.BACKTEST_START), np.datetime64("today"))) + 320
    # 업종 분류(FDR→KRX 비로그인 요청)를 KRX 로그인 '전'에 끝냄: 비로그인 요청이 로그인 세션을 끊는 것으로 보임
    sectors = get_sectors()
    kospi = get_ohlcv("KOSPI", count)["close"]
    uni = get_universe(C.UNIVERSE_SIZE)
    codes = uni["code"].tolist()
    px = get_price_panel(codes, count)
    etfs = get_theme_etfs(80)
    epx = get_price_panel(etfs["code"].tolist(), count)
    check_krx_login()
    flow = load_flow(codes, str(px["close"].index[0].date()), str(px["close"].index[-1].date()))
    fund_dates = [d for d in weekly_signal_dates(px["close"].index) if d >= t0 - pd.Timedelta(days=10)]
    fund = load_fundamentals(fund_dates, codes)
    fund_keys = pd.DatetimeIndex(sorted(fund))
    print(f"[data] 수급 {'있음 ' + str(flow['foreign'].shape[1]) + '종목' if flow is not None else '없음'}, "
          f"PER/PBR {len(fund)}개 날짜")

    def fund_asof(d):
        if not len(fund_keys):
            return None
        i = fund_keys.searchsorted(d, side="right") - 1
        return fund[fund_keys[i]] if i >= 0 and (d - fund_keys[i]).days <= 10 else None

    F = Factors(px["close"], px["volume"], flow, sectors)
    E = EtfFactors(epx["close"], epx["volume"])
    enames = dict(zip(etfs["code"], etfs["name"]))
    close_s, close_e = px["close"], epx["close"]
    fcache, scache = {}, {}

    def stock_scored(strategy, wd):
        key = (strategy, wd)
        if key in scache:
            return scache[key]
        out = []
        for d in [d for d in signal_dates(close_s.index) if d >= t0]:
            if d not in fcache:
                fcache[d] = F.at(d)
            ctx = {}
            sc = score(strategy, fcache[d], fund_asof(d), ctx)
            if sc is not None and len(sc) >= C.N_PICKS:
                out.append((d, sc, ctx.get("new_ok")))
        samples = []
        for (d, sc, _), (d2, _, _) in zip(out, out[1:]):
            if d2 > is_end:
                break
            rk = {c: i + 1 for i, c in enumerate(sc.index)}
            fwd = close_s.loc[d2] / close_s.loc[d] - 1
            samples += [(rk.get(c), fwd.get(c, np.nan)) for c in fcache[d].index]
        scache[key] = (out, fit_calibration(samples))
        return scache[key]

    def stock_sched(strategy, wd, ma, n, slots, bear=False):
        scored, cal = stock_scored(strategy, wd)
        afford_px = C.SEED / slots * C.MAX_PRICE_FRACTION
        hold, sched, bear_dates = [], {}, set()
        for d, sc, new_ok in scored:
            aff = set(sc.index[close_s.loc[d].reindex(sc.index) <= afford_px])
            if market_ok(kospi, ma, d):
                hold, _ = select_cost_aware(sc, hold, n, cal, new_ok, aff)
            elif bear:
                k21, k63 = kospi_rets(kospi, d)
                ok = bear_ok_stocks(fcache[d], k21, k63)
                hold = [h for h in hold if h in ok]
                nk = (new_ok & ok) if new_ok is not None else ok
                hold, _ = select_cost_aware(sc, hold, C.BEAR_STOCKS, cal, nk, aff & ok)
                bear_dates.add(d)
            else:
                hold = []
            sched[d] = list(hold)
        return sched, bear_dates

    def etf_sched(wd, ma, n, dates, bear=False):
        hold, sched, bear_dates = [], {}, set()
        for d in dates:
            if d not in close_e.index:
                sched[d] = list(hold)
                continue
            sc = E.scores(d)
            if market_ok(kospi, ma, d):
                hold = etf_select(sc, hold, n, n * 2, E, d) if len(sc) else []
            elif bear:
                k21, k63 = kospi_rets(kospi, d)
                ok = bear_ok_etfs(E, d, k21, k63)
                sc = sc[sc.index.isin(ok)]
                hold = [h for h in hold if h in ok]
                hold = etf_select(sc, hold, C.BEAR_ETFS, C.BEAR_ETFS * 2, E, d) if len(sc) else []
                bear_dates.add(d)
            else:
                hold = []
            sched[d] = list(hold)
        return sched, bear_dates

    import os
    variants_all = [
        ("H0", "종목3+ETF2, 시장필터 없음", {"stock": ("theme_flow_slow", 3), "etf": 2, "ma": 0}),
        ("H1", "종목3+ETF2, 하락장 전부 현금", {"stock": ("theme_flow_slow", 3), "etf": 2, "ma": 120}),
        ("B1", "종목3+ETF2, 하락장 엄격(주간 손절-7%)", {"stock": ("theme_flow_slow", 3), "etf": 2, "ma": 120,
                                                  "bear": True, "stop_daily": False}),
        ("B2", "종목3+ETF2, 하락장 엄격(매일 손절-7%)", {"stock": ("theme_flow_slow", 3), "etf": 2, "ma": 120,
                                                  "bear": True, "stop_daily": True}),
        ("S3", "종목 3개, 하락장 전부 현금", {"stock": ("theme_flow_slow", 3), "etf": 0, "ma": 120}),
        ("E4", "테마 ETF 3개, 하락장 전부 현금", {"stock": None, "etf": 3, "ma": 120}),
    ]
    B = {"etf": 2, "ma": 120, "bear": True, "stop_daily": False}
    variants_bo = [
        ("B1", "현재 전략 (테마·수급 느린교체, 하락장 엄격)", {"stock": ("theme_flow_slow", 3), **B}),
        ("X1", "급등초기: 거래대금 2배 + 20일 신고가", {"stock": ("breakout", 3), **B}),
        ("X2", "급등초기 + 외국인/기관 3일 연속 순매수", {"stock": ("breakout_flow", 3), **B}),
        ("X3", "급등초기 + 강한 업종(상위 3)", {"stock": ("breakout_theme", 3), **B}),
        ("X4", "급등초기 + 수급 + 강한 업종", {"stock": ("breakout_all", 3), **B}),
        ("X5", "현재 전략 + 신규매수는 돌파 신호 때만", {"stock": ("theme_breakout", 3), **B}),
        ("XS", "급등초기(X1) 종목만, ETF 없이", {"stock": ("breakout", 3), "etf": 0, "ma": 120, "bear": True,
                                           "stop_daily": False}),
        ("BS", "현재 전략 종목만, ETF 없이", {"stock": ("theme_flow_slow", 3), "etf": 0, "ma": 120, "bear": True,
                                        "stop_daily": False}),
    ]
    exp_set = os.environ.get("EXP_SET", "")
    variants = variants_bo if exp_set == "breakout" else variants_all
    # 신호 자체의 예측력: 신호가 뜬 종목의 이후 수익률 vs 전체 평균 (다음 거래일 시가 매수 기준)
    event = {}
    if exp_set == "breakout":
        op = px["open"]
        idx = close_s.index
        conds = {
            "거래대금 2배": lambda f: f["tv_surge"] >= 2,
            "20일 신고가": lambda f: f["close"] >= f["hi20_prev"],
            "거래대금 2배 + 20일 신고가": lambda f: (f["tv_surge"] >= 2) & (f["close"] >= f["hi20_prev"]),
            "+ 수급 3일 연속": lambda f: (f["tv_surge"] >= 2) & (f["close"] >= f["hi20_prev"]) & (f["flow_streak"] == 1),
            "지난주 +15% 이상 급등": lambda f: (f["close"] / F.close.shift(5).loc[f.name] .reindex(f.index) - 1) >= 0.15,
        }
        rows = {k: {h: [] for h in (5, 20, 60)} for k in conds}
        base = {h: [] for h in (5, 20, 60)}
        for d in [d for d in signal_dates(idx) if d >= t0]:
            i = idx.get_loc(d)
            if i + 61 >= len(idx):
                break
            f = fcache.get(d)
            if f is None:
                f = fcache[d] = F.at(d)
            f.name = d
            entry = op.iloc[i + 1].reindex(f.index)
            for h in (5, 20, 60):
                fwd = close_s.iloc[i + h].reindex(f.index) / entry - 1
                base[h].append(fwd.mean())
                for k, fn in conds.items():
                    m = fn(f).fillna(False)
                    if m.any():
                        rows[k][h] += list((fwd[m] - fwd.mean()).dropna())
        for k in conds:
            event[k] = {"n": len(rows[k][5])}
            for h in (5, 20, 60):
                x = np.array(rows[k][h])
                event[k][f"ex{h}"] = float(np.mean(x)) if len(x) else None
                event[k][f"win{h}"] = float(np.mean(x > 0)) if len(x) else None
            print(f"[신호] {k:22s} 건수 {event[k]['n']:6d} · 초과수익 1주 {event[k]['ex5'] or 0:+.2%} "
                  f"(이긴 비율 {event[k]['win5'] or 0:.0%}) · 1개월 {event[k]['ex20'] or 0:+.2%} "
                  f"({event[k]['win20'] or 0:.0%}) · 3개월 {event[k]['ex60'] or 0:+.2%} ({event[k]['win60'] or 0:.0%})")
    oc = pd.concat([close_s, close_e], axis=1)
    oo = pd.concat([px["open"], epx["open"]], axis=1)
    etf_set = set(close_e.columns)
    results, tue_curves, now_picks = {}, {}, {}
    for key, label, cfg in variants:
        per_wd = []
        for wd in range(1):                 # 주 2회(월·목 종가) 고정 일정
            slots = (cfg["stock"][1] if cfg["stock"] else 0) + cfg["etf"]
            dates = [d for d in signal_dates(close_s.index) if d >= t0]
            bear = cfg.get("bear", False)
            ss, b1 = stock_sched(cfg["stock"][0], wd, cfg["ma"], cfg["stock"][1], slots, bear) if cfg["stock"] else ({}, set())
            es, b2 = etf_sched(wd, cfg["ma"], cfg["etf"], dates, bear) if cfg["etf"] else ({}, set())
            sched = [(d, ss.get(d, []) + es.get(d, [])) for d in dates if d in ss or d in es]
            eq, trades = simulate(sched, oo, oc, etf_codes=etf_set, etf_buy=ETF_BUY, etf_sell=ETF_SELL,
                                  slots=slots, stop_dates=(b1 | b2) if bear else None,
                                  stop_loss=C.BEAR_STOP if bear else 0.0, stop_daily=cfg.get("stop_daily", False))
            eq = eq[eq.index >= t0]
            if eq.empty:
                continue
            full, oos = metrics(eq, kospi), metrics(eq[eq.index > is_end], kospi)
            buys = sum(1 for t in trades if t["side"] == "BUY")
            per_wd.append({"weekday": C.WEEKDAYS_KR[wd], "cagr": full["cagr"], "sharpe": full["sharpe"],
                           "mdd": full["mdd"], "oos_cagr": oos.get("cagr"), "oos_mdd": oos.get("mdd"),
                           "bench_cagr": full["bench_cagr"], "oos_bench": oos.get("bench_cagr"),
                           "buys_per_year": buys / max(full["years"], 0.1), "final_krw": float(eq.iloc[-1]),
                           "cash_weeks": float(np.mean([len(h) == 0 for _, h in sched])),
                           "stops_per_year": sum(1 for t in trades if t.get("stop")) / max(full["years"], 0.1)})
            if wd == 0:
                tue_curves[key] = eq
                now_picks[key] = [enames.get(c, c) for c in (sched[-1][1] if sched else [])]
        df = pd.DataFrame(per_wd)
        a = {k: float(df[k].mean()) for k in ("cagr", "sharpe", "mdd", "oos_cagr", "oos_mdd", "buys_per_year",
                                              "cash_weeks", "stops_per_year")}
        results[key] = {"label": label, "weekdays": per_wd, "avg": a,
                        "cagr_min": float(df["cagr"].min()), "cagr_max": float(df["cagr"].max()),
                        "bench_cagr": float(df["bench_cagr"].mean()), "oos_bench": float(df["oos_bench"].mean()),
                        "tue_final_krw": float(df["final_krw"].iloc[0])}
        print(f"{key} {label:28s} 평균CAGR {a['cagr']:+.1%} (요일범위 {results[key]['cagr_min']:+.1%}~"
              f"{results[key]['cagr_max']:+.1%}) 샤프 {a['sharpe']:.2f} MDD {a['mdd']:.1%} "
              f"검증CAGR {a['oos_cagr']:+.1%} 검증MDD {a['oos_mdd']:.1%} 매수/년 {a['buys_per_year']:.0f} "
              f"전액현금주 {a['cash_weeks']:.0%} 손절/년 {a['stops_per_year']:.1f} 100만원→ {results[key]['tue_final_krw']:,.0f}원")
    print(f"[유니버스] {'+'.join(C.UNIVERSE_MARKETS)} 시가총액 상위 {len(codes)}종목")
    k0 = next(iter(results))
    print(f"[참고] KOSPI 평균CAGR {results[k0]['bench_cagr']:+.1%}, 2022년 이후 {results[k0]['oos_bench']:+.1%}")
    print("[최근 화요일 선정]", now_picks)

    start = max(s.index[0] for s in tue_curves.values())
    wk = pd.DataFrame(tue_curves).loc[start:]
    wk = wk.groupby(pd.Grouper(freq="W-FRI")).tail(1).dropna(how="all")
    wk = wk / wk.iloc[0]
    b = kospi.reindex(wk.index, method="ffill")
    out = {"generated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
           "etf_universe": [{"code": c, "name": n} for c, n in zip(etfs["code"], etfs["name"])],
           "has_flow": flow is not None, "fund_dates": len(fund), "results": results, "event": event, "now_picks": now_picks,
           "curve": {"dates": [str(d.date()) for d in wk.index], "kospi": [round(float(x), 4) for x in b / b.iloc[0]],
                     **{k: [None if pd.isna(x) else round(float(x), 4) for x in wk[k]] for k in wk}}}
    C.DOCS_DATA.mkdir(parents=True, exist_ok=True)
    fn = "experiments_breakout.json" if exp_set == "breakout" else "experiments.json"
    (C.DOCS_DATA / fn).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    run()
