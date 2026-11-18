"""개선안 비교 실험: 테마 ETF 순환, 교체 축소, 시장 필터 (5개 요일 평균 + 2022년 이후 검증)

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
from .strategy import (Factors, score, select_cost_aware, fit_calibration, simulate,
                       weekly_signal_dates, metrics)

KST = timezone(timedelta(hours=9))
ETF_BUY = C.FEE + 0.0005          # ETF: 수수료 + 슬리피지 0.05%
ETF_SELL = C.FEE + 0.0005         # ETF 매도는 증권거래세 없음
ETF_MIN_TV = 3e8                  # 20일 평균 거래대금 3억 이상
ETF_MIN_HIST = 130
CORR_MAX = 0.85                   # 이미 고른 ETF와 상관이 이보다 높으면(같은 테마 중복) 건너뜀


class EtfFactors:
    def __init__(self, close, volume):
        c = close
        self.close = c
        self.ret = c.pct_change(fill_method=None)
        self.ret_21 = c / c.shift(21) - 1
        self.ret_63 = c / c.shift(63) - 1
        self.ma60 = c.rolling(60, min_periods=55).mean()
        self.high_126 = c.rolling(126, min_periods=100).max()
        self.tv20 = (c * volume.reindex_like(c)).rolling(20, min_periods=15).mean()
        self.hist = c.notna().cumsum()

    def scores(self, t):
        f = pd.DataFrame({"close": self.close.loc[t], "r21": self.ret_21.loc[t], "r63": self.ret_63.loc[t],
                          "ma60": self.ma60.loc[t], "hi": self.high_126.loc[t], "tv": self.tv20.loc[t],
                          "hist": self.hist.loc[t]})
        f = f[(f["hist"] >= ETF_MIN_HIST) & (f["tv"] >= ETF_MIN_TV) & f["r63"].notna() & f["ma60"].notna()]
        f = f[f["close"] > f["ma60"]]                         # 절대 모멘텀: 60일선 위만
        if f.empty:
            return pd.Series(dtype=float)
        r = lambda x: x.rank(pct=True)
        s = 0.4 * r(f["r21"]) + 0.4 * r(f["r63"]) + 0.2 * r(f["close"] / f["hi"])
        return s.sort_values(ascending=False)


def etf_select(sc: pd.Series, holdings, n, buffer, E: EtfFactors, t):
    rank = {c: i + 1 for i, c in enumerate(sc.index)}
    picks = [h for h in holdings if rank.get(h, 10 ** 9) <= buffer][:n]
    window = E.ret.loc[:t].tail(120)
    for c in sc.index:
        if len(picks) >= n:
            break
        if c in picks:
            continue
        if picks:
            cr = window[picks + [c]].corr()[c].drop(c)
            if (cr > CORR_MAX).any():
                continue
        picks.append(c)
    return picks


def market_ok(kospi, ma_n, t):
    if not ma_n:
        return True
    ma = kospi.rolling(ma_n, min_periods=int(ma_n * 0.9)).mean()
    k = kospi.loc[:t]
    return bool(k.iloc[-1] >= ma.loc[:t].iloc[-1]) if not np.isnan(ma.loc[:t].iloc[-1]) else True


def run():
    t0 = pd.Timestamp(C.BACKTEST_START)
    is_end = pd.Timestamp(C.IN_SAMPLE_END)
    count = int(np.busday_count(np.datetime64(C.BACKTEST_START), np.datetime64("today"))) + 320
    kospi = get_ohlcv("KOSPI", count)["close"]
    check_krx_login()

    # ── 개별 종목 (테마·수급) ──
    uni = get_universe(C.UNIVERSE_SIZE)
    px = get_price_panel(uni["code"].tolist(), count)
    sectors = get_sectors()
    codes = uni["code"].tolist()
    flow = load_flow(codes, str(px["close"].index[0].date()), str(px["close"].index[-1].date()))
    F = Factors(px["close"], px["volume"], flow, sectors)
    fund_dates = [d for d in weekly_signal_dates(px["close"].index) if d >= t0 - pd.Timedelta(days=10)]
    fund = load_fundamentals(fund_dates, codes)
    fund_keys = pd.DatetimeIndex(sorted(fund))

    def fund_asof(d):
        if not len(fund_keys):
            return None
        i = fund_keys.searchsorted(d, side="right") - 1
        return fund[fund_keys[i]] if i >= 0 and (d - fund_keys[i]).days <= 10 else None
    print(f"[data] 수급 {'있음' if flow is not None else '없음'}, PER/PBR {len(fund)}개 날짜")
    # ── 테마 ETF ──
    etfs = get_theme_etfs(80)
    epx = get_price_panel(etfs["code"].tolist(), count)
    E = EtfFactors(epx["close"], epx["volume"])
    enames = dict(zip(etfs["code"], etfs["name"]))

    # 최근 신호 점검 (지난 실험에서 마지막 화요일 ETF 선정이 비었던 원인 확인용)
    last_t = weekly_signal_dates(epx["close"].index, C.TRADE_WEEKDAY)[-1]
    ma120 = kospi.rolling(120).mean()
    sc_last = E.scores(last_t)
    tv_ok = (E.tv20.loc[last_t] >= ETF_MIN_TV).sum()
    above = (E.close.loc[last_t] > E.ma60.loc[last_t]).sum()
    print(f"[점검] {last_t.date()} KOSPI {kospi.loc[:last_t].iloc[-1]:,.0f} / 120일선 {ma120.loc[:last_t].iloc[-1]:,.0f} · "
          f"ETF 시세 {E.close.loc[last_t].notna().sum()}개, 거래대금 통과 {tv_ok}, 60일선 위 {above}, 최종 후보 {len(sc_last)}")
    print("[점검] 상위 후보:", [enames.get(c, c) for c in sc_last.index[:6]])

    variants = [
        ("S1", "테마·수급 종목 (현행)", "stock", {"strategy": "theme_flow", "ma": None}),
        ("S2", "테마·수급 종목 느린교체", "stock", {"strategy": "theme_flow_slow", "ma": None}),
        ("S3", "S2 + 시장필터(120일선)", "stock", {"strategy": "theme_flow_slow", "ma": 120}),
        ("C1", "복합(가치+저변동+모멘텀)", "stock", {"strategy": "composite", "ma": None}),
        ("E1", "테마 ETF 순환 4개", "etf", {"n": 4, "ma": None}),
        ("E2", "테마 ETF 4개 + 시장필터(120일선)", "etf", {"n": 4, "ma": 120}),
        ("E3", "테마 ETF 4개 + 시장필터(200일선)", "etf", {"n": 4, "ma": 200}),
        ("E4", "테마 ETF 3개 + 시장필터(120일선)", "etf", {"n": 3, "ma": 120}),
    ]
    fcache = {}
    results = {}
    tue_curves = {}
    for key, label, kind, cfg in variants:
        per_wd = []
        for wd in range(5):
            close = px["close"] if kind == "stock" else epx["close"]
            dates = [d for d in weekly_signal_dates(close.index, wd) if d >= t0]
            sched, holdings = [], []
            if kind == "stock":
                scored = []
                for d in dates:
                    if d not in fcache:
                        fcache[d] = F.at(d)
                    ctx = {}
                    sc = score(cfg["strategy"], fcache[d], fund_asof(d), ctx)
                    if sc is not None and len(sc) >= C.N_PICKS:
                        scored.append((d, sc, ctx.get("new_ok")))
                samples = []
                for (d, sc, _), (d2, _, _) in zip(scored, scored[1:]):
                    if d2 > is_end:
                        break
                    rk = {c: i + 1 for i, c in enumerate(sc.index)}
                    fwd = close.loc[d2] / close.loc[d] - 1
                    samples += [(rk.get(c), fwd.get(c, np.nan)) for c in fcache[d].index]
                cal = fit_calibration(samples)
                afford_px = C.SEED / C.N_PICKS * C.MAX_PRICE_FRACTION
                for d, sc, new_ok in scored:
                    if not market_ok(kospi, cfg["ma"], d):
                        holdings = []
                    else:
                        aff = set(sc.index[close.loc[d].reindex(sc.index) <= afford_px])
                        holdings, _ = select_cost_aware(sc, holdings, C.N_PICKS, cal, new_ok, aff)
                    sched.append((d, list(holdings)))
                eq, trades = simulate(sched, px["open"], close)
            else:
                for d in dates:
                    if not market_ok(kospi, cfg["ma"], d):
                        holdings = []
                    else:
                        sc = E.scores(d)
                        holdings = etf_select(sc, holdings, cfg["n"], cfg["n"] * 2, E, d) if len(sc) else []
                    sched.append((d, list(holdings)))
                eq, trades = simulate(sched, epx["open"], close, buy_cost=ETF_BUY, sell_cost=ETF_SELL)
            eq = eq[eq.index >= t0]
            if eq.empty:
                continue
            full = metrics(eq, kospi)
            oos = metrics(eq[eq.index > is_end], kospi)
            buys = sum(1 for t in trades if t["side"] == "BUY")
            cash_ratio = float(np.mean([len(h) == 0 for _, h in sched]))
            per_wd.append({"weekday": C.WEEKDAYS_KR[wd], "cagr": full["cagr"], "sharpe": full["sharpe"],
                           "mdd": full["mdd"], "oos_cagr": oos.get("cagr"), "oos_mdd": oos.get("mdd"),
                           "bench_cagr": full["bench_cagr"], "oos_bench": oos.get("bench_cagr"),
                           "buys_per_year": buys / max(full["years"], 0.1), "final_krw": float(eq.iloc[-1]),
                           "cash_weeks": cash_ratio, "start": str(eq.index[0].date())})
            if wd == C.TRADE_WEEKDAY:
                tue_curves[key] = eq
                if kind == "etf":
                    last = sched[-1][1] if sched else []
                    results.setdefault("_etf_now", {})[key] = [enames.get(c, c) for c in last]
        if not per_wd:
            continue
        df = pd.DataFrame(per_wd)
        results[key] = {"label": label, "kind": kind, "weekdays": per_wd,
                        "avg": {k: float(df[k].mean()) for k in
                                ("cagr", "sharpe", "mdd", "oos_cagr", "oos_mdd", "buys_per_year", "cash_weeks")},
                        "cagr_min": float(df["cagr"].min()), "cagr_max": float(df["cagr"].max()),
                        "bench_cagr": float(df["bench_cagr"].mean()), "oos_bench": float(df["oos_bench"].mean()),
                        "tue_final_krw": float(df.loc[df["weekday"] == "화", "final_krw"].iloc[0])
                        if (df["weekday"] == "화").any() else None, "start": per_wd[0]["start"]}
        a = results[key]["avg"]
        print(f"{key} {label:28s} 평균CAGR {a['cagr']:+.1%} (요일범위 {results[key]['cagr_min']:+.1%}~"
              f"{results[key]['cagr_max']:+.1%}) 샤프 {a['sharpe']:.2f} MDD {a['mdd']:.1%} "
              f"검증CAGR {a['oos_cagr']:+.1%} 검증MDD {a['oos_mdd']:.1%} 매수/년 {a['buys_per_year']:.0f} "
              f"현금주 {a['cash_weeks']:.0%} 화요일 최종 {results[key]['tue_final_krw']:,.0f}원")

    start = max(s.index[0] for s in tue_curves.values())
    wk = pd.DataFrame(tue_curves).loc[start:]
    wk = wk.groupby(pd.Grouper(freq="W-FRI")).tail(1).dropna(how="all")
    wk = wk / wk.iloc[0]
    b = kospi.reindex(wk.index, method="ffill")
    out = {"generated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
           "etf_universe": [{"code": c, "name": n} for c, n in zip(etfs["code"], etfs["name"])],
           "costs": {"stock_round_trip": C.BUY_COST + C.SELL_COST, "etf_round_trip": ETF_BUY + ETF_SELL},
           "results": results,
           "curve": {"dates": [str(d.date()) for d in wk.index], "kospi": [round(float(x), 4) for x in b / b.iloc[0]],
                     **{k: [None if pd.isna(x) else round(float(x), 4) for x in wk[k]] for k in wk}}}
    C.DOCS_DATA.mkdir(parents=True, exist_ok=True)
    (C.DOCS_DATA / "experiments.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    run()
