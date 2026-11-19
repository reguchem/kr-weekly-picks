"""전략 백테스트 → docs/data/backtest.json, docs/data/strategy.json

실행: python -m src.backtest
전략 선택은 IN_SAMPLE_END까지의 샤프지수로만 하고, 이후 구간은 '검증' 성과로 따로 보여줍니다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from . import config as C
from .data import get_universe, get_price_panel, get_ohlcv, get_fundamentals, get_sectors, get_flow_panel
from .strategy import (STRATEGIES, Factors, score, select, select_cost_aware, fit_calibration, simulate,
                       weekly_signal_dates, signal_dates, metrics)

KST = timezone(timedelta(hours=9))
FUND_CACHE = C.CACHE / "fundamentals.csv.gz"


def load_fundamentals(dates, codes):
    C.CACHE.mkdir(parents=True, exist_ok=True)
    cache = pd.read_csv(FUND_CACHE, dtype={"code": str}, parse_dates=["date"]) if FUND_CACHE.exists() \
        else pd.DataFrame(columns=["date", "code", "PER", "PBR"])
    have = set(pd.to_datetime(cache["date"]).dt.normalize()) if len(cache) else set()
    new, fails = [], 0
    for i, d in enumerate(dates):
        if d in have:
            continue
        if fails >= 5 and not new:
            print("[fundamental] 연속 실패 → 가치 팩터 없이 진행")
            break
        df = get_fundamentals(d)
        if df is None:
            fails += 1
            continue
        df = df.reindex(codes).dropna(how="all").reset_index(names="code")
        df["date"] = d
        new.append(df)
        if i % 50 == 0:
            print(f"[fundamental] {d.date()} ({i}/{len(dates)})")
            pd.concat([cache, *new], ignore_index=True).to_csv(FUND_CACHE, index=False, compression="gzip")
    if new:
        cache = pd.concat([cache, *new], ignore_index=True)
        cache.to_csv(FUND_CACHE, index=False, compression="gzip")
    out = {}
    if len(cache):
        cache["date"] = pd.to_datetime(cache["date"])
        for d, g in cache.groupby("date"):
            out[d] = g.set_index("code")[["PER", "PBR"]]
    return out


def load_flow(codes, start, end, batch=40):
    """외국인·기관 순매수 이력 (캐시 + 증분 수집). batch 종목마다 캐시에 저장해 중단돼도 이어받기 가능"""
    C.CACHE.mkdir(parents=True, exist_ok=True)
    paths = {k: C.CACHE / f"flow_{k}.csv.gz" for k in ("foreign", "inst")}
    cached = {k: pd.read_csv(p, index_col=0, parse_dates=True) for k, p in paths.items() if p.exists()}
    if len(cached) == 2:
        for k in cached:
            cached[k].columns = [str(c).zfill(6) for c in cached[k].columns]

    def merge_save(new):
        nonlocal cached
        out = {}
        for k in paths:
            parts = [x for x in (cached.get(k), new[k] if new else None) if x is not None]
            if not parts:
                return
            df = pd.concat(parts, axis=0).sort_index()
            df = df.T.groupby(level=0).last().T if df.columns.duplicated().any() else df
            df = df.groupby(level=0).last()
            df.columns = [str(c).zfill(6) for c in df.columns]
            df.to_csv(paths[k], compression="gzip", float_format="%.0f")
            out[k] = df
        cached = out

    have = set(cached["foreign"].columns) if len(cached) == 2 else set()
    missing = [c for c in codes if c not in have]
    if missing:
        print(f"[flow] 신규 종목 {len(missing)}개 전체 기간 수집 ({batch}개마다 저장)")
        for i in range(0, len(missing), batch):
            new = get_flow_panel(missing[i:i + batch], start, end)
            if new is None and i == 0 and not have:
                return None              # 첫 묶음부터 전부 실패 → KRX 문제로 보고 중단
            if new is not None:
                merge_save(new)
    if have and len(cached) == 2:
        last = cached["foreign"].index[-1]
        if last < pd.Timestamp(end):
            new = get_flow_panel([c for c in codes if c in have], str((last + pd.Timedelta(days=1)).date()), end)
            if new is not None:
                merge_save(new)
    return cached if len(cached) == 2 else None


def run():
    uni = get_universe(C.UNIVERSE_SIZE)
    codes = uni["code"].tolist()
    count = int(np.busday_count(np.datetime64(C.BACKTEST_START), np.datetime64("today"))) + 320
    px = get_price_panel(codes, count)
    kospi = get_ohlcv("KOSPI", count)["close"]
    close, open_ = px["close"], px["open"]
    sectors = get_sectors()
    flow = load_flow(codes, str(close.index[0].date()), str(close.index[-1].date()))
    F = Factors(close, px["volume"], flow, sectors)
    t0 = pd.Timestamp(C.BACKTEST_START)
    # PER/PBR은 주 1회(주 마지막 거래일)만 수집하고, 각 신호일엔 그 이전 가장 최근 값을 사용
    fund_dates = [d for d in weekly_signal_dates(close.index) if d >= t0 - pd.Timedelta(days=10)]
    fund = load_fundamentals(fund_dates, codes)
    fund_keys = pd.DatetimeIndex(sorted(fund))
    has_value = len(fund) > len(fund_dates) * 0.5

    def fund_asof(d):
        if not len(fund_keys):
            return None
        i = fund_keys.searchsorted(d, side="right") - 1
        return fund[fund_keys[i]] if i >= 0 and (d - fund_keys[i]).days <= 10 else None

    fcache = {}

    def fat(d):
        if d not in fcache:
            fcache[d] = F.at(d)
        return fcache[d]

    is_end = pd.Timestamp(C.IN_SAMPLE_END)
    names = [n for n in STRATEGIES if not (n == "value" and not has_value)
             and not (n == "theme_flow" and not sectors)]
    grid = {}   # (전략, 요일) → 결과
    cals = {}
    afford_px = C.SEED / C.N_PICKS * C.MAX_PRICE_FRACTION
    for wd in range(1):                     # 일정: 주 2회(월 종가·목 종가) 고정
        dates = [d for d in signal_dates(close.index) if d >= t0]
        for name in names:
            _, label, desc, buffer = STRATEGIES[name]
            # 1) 날짜별 점수 (보유와 무관하므로 한 번만 계산)
            scored = []
            for d in dates:
                ctx = {}
                sc = score(name, fat(d), fund_asof(d), ctx)
                if sc is not None and len(sc) >= C.N_PICKS:
                    scored.append((d, sc, ctx.get("new_ok")))
            # 2) 순위 → 다음 1주 수익률 보정 (선택 구간 데이터만 사용)
            samples = []
            for (d, sc, _), (d2, _, _) in zip(scored, scored[1:]):
                if d2 > is_end:
                    break
                rk = {c: i + 1 for i, c in enumerate(sc.index)}
                fwd = close.loc[d2] / close.loc[d] - 1
                for c in fat(d).index:
                    samples.append((rk.get(c), fwd.get(c, np.nan)))
            cal = fit_calibration(samples)
            cals[(name, wd)] = cal
            # 3) 선정: 매주 전량교체 변형은 단순 상위, 나머지는 비용 반영 교체
            holdings, sched = [], []
            for d, sc, new_ok in scored:
                aff = set(sc.index[close.loc[d].reindex(sc.index) <= afford_px])
                if buffer <= C.N_PICKS:
                    holdings, _ = select(sc, [], C.N_PICKS, buffer, new_ok)
                else:
                    holdings, _ = select_cost_aware(sc, holdings, C.N_PICKS, cal, new_ok, aff)
                sched.append((d, list(holdings)))
            eq, trades = simulate(sched, open_, close)
            if eq.empty:
                continue
            buys = sum(1 for t in trades if t["side"] == "BUY")
            full = metrics(eq, kospi)
            grid[(name, wd)] = {
                "full": full, "in_sample": metrics(eq[eq.index <= is_end], kospi),
                "out_sample": metrics(eq[eq.index > is_end], kospi),
                "buys_per_year": round(buys / max(full.get("years", 1), 0.1), 1),
                "final_krw": round(float(eq.iloc[-1])), "eq": eq}
            print(f"[월·목] {label:24s} CAGR {full['cagr']:+.1%}  Sharpe {full['sharpe']:.2f}  "
                  f"MDD {full['mdd']:.1%}  매수/년 {grid[(name, wd)]['buys_per_year']}")

    def is_sharpe(name, wd):
        g = grid.get((name, wd))
        return g["in_sample"].get("sharpe", np.nan) if g and g["in_sample"] else np.nan

    # 매매 요일은 고정(화요일). 다른 요일 결과는 참고용 비교표로만 표시
    weekday = C.TRADE_WEEKDAY
    avg_is = {n: float(np.nanmean([is_sharpe(n, wd) for wd in range(1)])) for n in names}
    # 전략 선택: 고정 요일 기준 인샘플 샤프 최고 (전량교체 비교용 변형은 제외)
    pool = {n: is_sharpe(n, weekday) for n in names if n != "composite_nobuffer"}
    pool = {n: v for n, v in pool.items() if not np.isnan(v)}
    best = max(pool, key=pool.get) if pool else C.DEFAULT_STRATEGY
    # STRATEGY_MODE="fixed"면 백테스트 결과와 무관하게 DEFAULT_STRATEGY 사용(비교표만 갱신)
    selected = best if C.STRATEGY_MODE == "auto" or C.DEFAULT_STRATEGY not in names else C.DEFAULT_STRATEGY
    print(f"선택: {selected} (주 2회 월·목 종가 신호, 다음날 시가 매매)")

    results = {}
    for n in names:
        g = grid.get((n, weekday))
        if not g:
            continue
        _, label, desc, buffer = STRATEGIES[n]
        results[n] = {"label": label, "desc": desc, "buffer": buffer, "avg_is_sharpe": round(avg_is[n], 3),
                      **{k: g[k] for k in ("full", "in_sample", "out_sample", "buys_per_year", "final_krw")}}
    weekdays = [{"weekday": wd, "label": C.WEEKDAYS_KR[wd],
                 **{k: grid[(selected, wd)][k] for k in ("full", "in_sample", "out_sample", "buys_per_year")}}
                for wd in range(1) if (selected, wd) in grid]

    # 요일별 KOSPI 일간 수익률 (참고)
    kr = kospi.pct_change().dropna()
    kr = kr[kr.index >= t0]
    kospi_wd = []
    for wd in range(5):
        a, b = kr[kr.index.weekday == wd], kr[(kr.index.weekday == wd) & (kr.index.year >= 2020)]
        kospi_wd.append({"label": C.WEEKDAYS_KR[wd], "mean": float(a.mean()), "up": float((a > 0).mean()),
                         "mean_2020": float(b.mean()), "up_2020": float((b > 0).mean()), "n": int(len(a))})

    curves = {n: grid[(n, weekday)]["eq"] for n in results}
    start = max(s.index[0] for s in curves.values())
    wk = lambda x: x.groupby(pd.Grouper(freq="W-FRI")).tail(1)  # 실제 거래일 날짜 유지
    weekly = wk(pd.DataFrame(curves).loc[start:]).dropna(how="all")
    weekly = weekly / weekly.iloc[0]
    bench = kospi.reindex(close.index).ffill().reindex(weekly.index)
    bench = bench / bench.iloc[0]

    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    out = {
        "generated_at": now,
        "period": [str(weekly.index[0].date()), str(weekly.index[-1].date())],
        "in_sample_end": C.IN_SAMPLE_END,
        "universe": f"{'+'.join(C.UNIVERSE_MARKETS)} 시가총액 상위 {len(codes)}종목 (현재 기준)",
        "has_value_factor": has_value,
        "has_flow": flow is not None, "has_sectors": bool(sectors),
        "costs": {"buy": C.BUY_COST, "sell": C.SELL_COST},
        "execute_at": C.EXECUTE_AT, "seed": C.SEED,
        "calibration": cals.get((selected, weekday)),
        "selected": selected, "best_in_sample": best, "strategy_mode": C.STRATEGY_MODE, "weekday": weekday, "weekday_label": C.WEEKDAYS_KR[weekday],
        "strategies": results,
        "weekdays": weekdays,
        "kospi_weekday": kospi_wd,
        "curve": {
            "dates": [str(d.date()) for d in weekly.index],
            "bench": [round(float(x), 4) for x in bench],
            **{k: [None if pd.isna(x) else round(float(x), 4) for x in weekly[k]] for k in weekly},
        },
    }
    C.DOCS_DATA.mkdir(parents=True, exist_ok=True)
    (C.DOCS_DATA / "backtest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (C.DOCS_DATA / "strategy.json").write_text(json.dumps(
        {"strategy": selected, "label": STRATEGIES[selected][1], "weekday": weekday,
         "calibration": cals.get((selected, weekday)),
         "weekday_label": C.WEEKDAYS_KR[weekday], "selected_at": now,
         "reason": ("고정 전략" if selected != best else f"{C.IN_SAMPLE_END} 이전 샤프지수 1위")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    run()
