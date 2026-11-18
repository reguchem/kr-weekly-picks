"""팩터 계산, 전략 점수, 종목 선정(버퍼 규칙), 포트폴리오 시뮬레이션."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ───────────────────────── 팩터 ─────────────────────────
class Factors:
    """가격 패널로부터 롤링 팩터를 한 번에 계산해 두고 날짜별로 꺼내 씀."""

    def __init__(self, close: pd.DataFrame, volume: pd.DataFrame, flow: dict | None = None,
                 sectors: dict | None = None):
        c = close
        ret = c.pct_change(fill_method=None)
        self.close = c
        self.sectors = sectors or {}
        self.ret_21 = c / c.shift(21) - 1
        self.ret_63 = c / c.shift(63) - 1
        self.ma20 = c.rolling(20, min_periods=18).mean()
        self.ma60 = c.rolling(60, min_periods=55).mean()
        self.high_252 = c.rolling(252, min_periods=200).max()
        tv = c * volume.reindex_like(c)
        self.tv120 = tv.rolling(120, min_periods=100).mean()
        # 외국인·기관 20일 순매수대금 / 20일 거래대금
        self.flow_f = self.flow_i = None
        if flow is not None:
            tv20s = tv.rolling(20, min_periods=15).sum()
            for k, attr in (("foreign", "flow_f"), ("inst", "flow_i")):
                fl = flow[k].reindex(index=c.index, columns=c.columns)
                setattr(self, attr, fl.rolling(20, min_periods=10).sum() / tv20s)
        self.mom_12_1 = c.shift(21) / c.shift(252) - 1     # 12개월 모멘텀(최근 1개월 제외)
        self.mom_6 = c / c.shift(126) - 1                  # 6개월 추세
        self.rev_1w = -(c / c.shift(5) - 1)                # 1주 반전 (많이 빠질수록 높음)
        self.vol_60 = ret.rolling(60, min_periods=50).std()
        self.ma200 = c.rolling(200, min_periods=180).mean()
        self.tv20 = (c * volume.reindex_like(c)).rolling(20, min_periods=15).mean()
        self.hist = c.notna().cumsum()

    def at(self, t) -> pd.DataFrame:
        f = pd.DataFrame({
            "close": self.close.loc[t],
            "mom_12_1": self.mom_12_1.loc[t],
            "mom_6": self.mom_6.loc[t],
            "rev_1w": self.rev_1w.loc[t],
            "vol_60": self.vol_60.loc[t],
            "ma200": self.ma200.loc[t],
            "tv20": self.tv20.loc[t],
            "hist": self.hist.loc[t],
            "ret_21": self.ret_21.loc[t], "ret_63": self.ret_63.loc[t],
            "ma20": self.ma20.loc[t], "ma60": self.ma60.loc[t], "high_252": self.high_252.loc[t], "tv120": self.tv120.loc[t],
            "flow_f": self.flow_f.loc[t] if self.flow_f is not None and t in self.flow_f.index else np.nan,
            "flow_i": self.flow_i.loc[t] if self.flow_i is not None and t in self.flow_i.index else np.nan,
        })
        f["sector"] = [self.sectors.get(c) for c in f.index]
        ok = (f["close"].notna() & (f["hist"] >= C.MIN_HISTORY_DAYS)
              & (f["tv20"] >= C.MIN_TRADING_VALUE) & f["vol_60"].notna() & f["mom_12_1"].notna())
        return f[ok]


def _r(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def _value(f, fund):
    if fund is None:
        return None
    fd = fund.reindex(f.index)
    if fd["PER"].notna().sum() < len(f) * 0.5:
        return None
    ey = (1 / fd["PER"]).where(fd["PER"] > 0, 0).fillna(0)
    by = (1 / fd["PBR"]).where(fd["PBR"] > 0, 0).fillna(0)
    return (_r(ey) + _r(by)) / 2


def s_composite(f, fund, ctx=None):
    parts = [_r(-f["vol_60"]), _r(f["mom_12_1"])]
    v = _value(f, fund)
    if v is not None:
        parts.append(v)
    return sum(parts) / len(parts)


def s_lowvol_trend(f, fund, ctx=None):
    s = (_r(-f["vol_60"]) + _r(f["mom_6"])) / 2
    return s.where(f["close"] > f["ma200"])


def s_momentum(f, fund, ctx=None):
    return _r(f["mom_12_1"])


def s_value(f, fund, ctx=None):
    return _value(f, fund)


def s_reversal(f, fund, ctx=None):
    return _r(f["rev_1w"])


TOP_SECTORS = 3        # 매수 후보 업종 수
KEEP_SECTORS = 5       # 보유 종목은 업종이 이 순위 안이면 유지
MIN_SECTOR_SIZE = 3


def sector_strength(f: pd.DataFrame, ctx=None) -> pd.DataFrame:
    """업종별 추세 강도: 소속 종목 1개월·3개월 수익률 순위 평균의 중앙값 (+ 외국인 수급, + 뉴스 전망 보정)"""
    g = f.dropna(subset=["sector"]).copy()
    g["m"] = (_r(g["ret_21"]) + _r(g["ret_63"])) / 2
    agg = g.groupby("sector").agg(n=("m", "size"), mom=("m", "median"), ret_21=("ret_21", "median"),
                                   flow=("flow_f", "median"))
    agg = agg[agg["n"] >= MIN_SECTOR_SIZE]
    sc = _r(agg["mom"])
    if agg["flow"].notna().sum() > len(agg) * 0.5:
        sc = 0.75 * sc + 0.25 * _r(agg["flow"]).fillna(0.5)
    tilt = (ctx or {}).get("sector_tilt") or {}
    agg["tilt"] = [tilt.get(s, 0) for s in agg.index]
    agg["score"] = sc + 0.15 * agg["tilt"]          # 뉴스·정세 전망은 ±0.15만 반영 (보정용)
    agg = agg.sort_values("score", ascending=False)
    agg["rank"] = np.arange(1, len(agg) + 1)
    return agg


def s_theme_flow_slow(f, fund, ctx=None):
    """테마·수급과 같지만 추세 이탈 기준을 60일선으로 완화 → 교체 횟수 축소"""
    return s_theme_flow(f, fund, ctx, trend_ma="ma60")


def s_theme_flow(f, fund, ctx=None, trend_ma="ma20"):
    """업종 추세 + 주도주 + 외국인 수급. 한국 시장의 테마 순환을 따라가되 과열·추세 이탈은 제외."""
    if f["sector"].notna().sum() < len(f) * 0.5:
        return None
    sec = sector_strength(f, ctx)
    if ctx is not None:
        ctx["sectors_table"] = sec
    keep = set(sec.index[:KEEP_SECTORS])
    g = f[f["sector"].isin(keep)].copy()
    ok = (g["close"] > g[trend_ma]) & (g["ret_21"] <= 0.40) & (g["close"] / g["ma20"] <= 1.25)
    g = g[ok]
    if g.empty:
        return None
    near_high = _r(g["close"] / g["high_252"])
    tv_up = _r(g["tv20"] / g["tv120"])
    parts = [(0.35, _r(g["sector"].map(sec["score"]))), (0.25, near_high), (0.15, tv_up)]
    if g["flow_f"].notna().sum() > len(g) * 0.5:
        flow = _r(g["flow_f"].fillna(g["flow_f"].median()) + 0.5 * g["flow_i"].fillna(0))
        parts.append((0.25, flow))
    w = sum(p[0] for p in parts)
    score = sum(wt * x for wt, x in parts) / w
    # 신규 매수는 상위 TOP_SECTORS 업종에서만, 4~5위 업종 종목은 '보유 유지'만 가능
    if ctx is not None:
        top = set(sec.index[:TOP_SECTORS])
        ctx["new_ok"] = set(g.index[g["sector"].isin(top)])
    return score


# name: (함수, 한글이름, 설명, 버퍼순위)
STRATEGIES = {
    "theme_flow": (s_theme_flow, "테마·수급(업종추세+주도주+외국인)",
                   "강한 업종 상위 3개 → 신고가 근접·거래대금 증가·외국인 순매수 종목, 과열·20일선 이탈 제외",
                   8),
    "theme_flow_slow": (s_theme_flow_slow, "테마·수급 느린교체(60일선)",
                        "테마·수급과 같으나 60일선 이탈 시에만 탈락", 8),
    "composite": (s_composite, "복합(가치+저변동+모멘텀)",
                  "저PER·저PBR, 낮은 변동성, 12-1개월 모멘텀 순위의 평균", C.BUFFER_RANK),
    "lowvol_trend": (s_lowvol_trend, "저변동+추세",
                     "200일선 위 종목 중 변동성 낮고 6개월 추세 강한 종목", C.BUFFER_RANK),
    "momentum": (s_momentum, "모멘텀", "12-1개월 수익률 상위", C.BUFFER_RANK),
    "value": (s_value, "가치", "저PER·저PBR 상위", C.BUFFER_RANK),
    "reversal": (s_reversal, "단기반전(매주 전량교체)", "지난주 낙폭 상위 — 매주 3종목 모두 교체", C.N_PICKS),
    "composite_nobuffer": (s_composite, "복합(매주 전량교체)",
                           "복합 점수 상위 3종목을 매주 새로 매수 — 버퍼 효과 비교용", C.N_PICKS),
}


def score(strategy: str, f: pd.DataFrame, fund, ctx=None) -> pd.Series | None:
    s = STRATEGIES[strategy][0](f, fund, ctx)
    return None if s is None else s.dropna().sort_values(ascending=False)


def select(scores: pd.Series, holdings: list[str], n: int, buffer: int,
           new_ok: set | None = None) -> tuple[list[str], dict]:
    """버퍼 규칙: 보유 종목이 buffer위 안이면 유지, 빈자리는 최상위 신규 종목으로(new_ok가 있으면 그 안에서)."""
    rank = {c: i + 1 for i, c in enumerate(scores.index)}
    keep = [h for h in holdings if rank.get(h, 10 ** 9) <= buffer][:n]
    picks = list(keep)
    for c in scores.index:
        if len(picks) >= n:
            break
        if c not in picks and (new_ok is None or c in new_ok):
            picks.append(c)
    return picks, rank


# ───────────────────────── 기대수익 · 비용 반영 선정 ─────────────────────────
RANK_BUCKETS = [(1, 3), (4, 10), (11, 20), (21, 40), (41, 10 ** 6)]
ROUND_TRIP = C.BUY_COST + C.SELL_COST


def bucket_of(rank: int | None) -> int:
    if rank is None:
        return len(RANK_BUCKETS)          # 점수 대상에서 빠짐(필터 탈락)
    for i, (lo, hi) in enumerate(RANK_BUCKETS):
        if lo <= rank <= hi:
            return i
    return len(RANK_BUCKETS) - 1


def fit_calibration(samples: list[tuple[int | None, float]], shrink: float = 0.5) -> dict:
    """samples: (점수 순위 or None, 다음 1주 수익률) → 순위 구간별 예상 주간 수익률.
    표본 잡음을 줄이기 위해 전체 평균 쪽으로 shrink 만큼 당김."""
    df = pd.DataFrame(samples, columns=["rank", "fwd"]).dropna(subset=["fwd"])
    df["fwd"] = df["fwd"].clip(-0.5, 0.5)
    df["b"] = [bucket_of(None if pd.isna(r) else int(r)) for r in df["rank"]]
    mu = float(df["fwd"].mean()) if len(df) else 0.0
    out = []
    for i in range(len(RANK_BUCKETS) + 1):
        g = df.loc[df["b"] == i, "fwd"]
        raw = float(g.mean()) if len(g) else mu
        lab = (f"{RANK_BUCKETS[i][0]}~{RANK_BUCKETS[i][1]}위" if i < len(RANK_BUCKETS) and RANK_BUCKETS[i][1] < 10 ** 6
               else (f"{RANK_BUCKETS[i][0]}위~" if i < len(RANK_BUCKETS) else "조건 탈락"))
        out.append({"bucket": i, "label": lab, "n": int(len(g)), "raw": raw,
                    "exp": mu + (1 - shrink) * (raw - mu), "win": float((g > 0).mean()) if len(g) else None})
    return {"mean": mu, "buckets": out, "round_trip": ROUND_TRIP}


def expected(cal: dict, rank: int | None) -> float:
    return cal["buckets"][bucket_of(rank)]["exp"]


def select_cost_aware(scores: pd.Series, holdings: list[str], n: int, cal: dict,
                      new_ok: set | None = None, affordable: set | None = None):
    """수수료를 고려한 선정: 빈자리는 예상수익 최고 종목으로 채우고,
    보유 종목은 '새 종목 예상수익 − 보유 종목 예상수익 > 왕복 거래비용'일 때만 교체."""
    rank = {c: i + 1 for i, c in enumerate(scores.index)}
    ok = lambda c: (new_ok is None or c in new_ok) and (affordable is None or c in affordable)
    cands = [c for c in scores.index if c not in holdings and ok(c)]
    # 전략 조건에서 탈락한 보유 종목(20일선 이탈·업종 약화·과열 등)은 무조건 교체
    picks = [h for h in holdings[:n] if h in rank or not cands]
    while len(picks) < n and cands:
        picks.append(cands.pop(0))
    # 교체 이득은 예상 보유기간 동안 이어진다고 보고, 그 합이 왕복 비용보다 클 때만 교체
    while cands and picks:
        worst = min(picks, key=lambda c: expected(cal, rank.get(c)))
        gain = (expected(cal, rank.get(cands[0])) - expected(cal, rank.get(worst))) * C.HOLD_WEEKS_EST
        if gain > ROUND_TRIP:
            picks[picks.index(worst)] = cands.pop(0)
        else:
            break
    return picks, rank


# ───────────────────────── 시뮬레이션 ─────────────────────────
def simulate(schedule: list[tuple[pd.Timestamp, list[str]]], open_: pd.DataFrame,
             close: pd.DataFrame, end=None, execute_at: str = C.EXECUTE_AT, capital: float = C.SEED,
             buy_cost: float = C.BUY_COST, sell_cost: float = C.SELL_COST, etf_codes: set | None = None,
             etf_buy: float = 0.00065, etf_sell: float = 0.00065,
             stop_dates: set | None = None, stop_loss: float = 0.0, stop_daily: bool = False,
             slots: int | None = None):
    """schedule: [(신호일, 목표 종목들)]
    execute_at="close": 신호일 종가에 매매(시간외 종가) / "next_open": 다음 거래일 시가에 매매.
    capital 원으로 시작, **정수 주식 수**로 매매 (남는 돈은 현금).

    반환: equity(일별 평가금액, 원), trades(list[dict])
    """
    days = close.index
    end = end or days[-1]
    close_f = close.ffill()
    cash, pos = float(capital), {}
    trades, eq = [], {}
    sched = {}
    for t, picks in schedule:
        i = days.searchsorted(t, side="left" if execute_at == "close" else "right")
        if i < len(days):
            sched[days[i]] = (t, picks)
    start = min(sched) if sched else None
    if start is None:
        return pd.Series(dtype=float), trades
    stopped_pos, stopped_today = {}, set()
    for d in days[days >= start]:
        if d > end:
            break
        stopped_today = set()
        # 손절: stop_dates(하락장 신호일)에 편입한 종목이 편입가 대비 stop_loss 이상 하락하면 종가 매도
        if stop_loss and stopped_pos:
            cl_d = close.loc[d]
            if stop_daily or d in sched:
                for c in [c for c in list(pos) if c in stopped_pos]:
                    px = cl_d.get(c, np.nan)
                    if not np.isnan(px) and px <= stopped_pos[c] * (1 - stop_loss):
                        sh = pos.pop(c)
                        stopped_pos.pop(c)
                        cash += sh * px * (1 - (etf_sell if etf_codes and c in etf_codes else sell_cost))
                        trades.append({"date": d, "signal": d, "code": c, "side": "SELL", "price": float(px),
                                       "shares": int(sh), "stop": True})
                        stopped_today.add(c)
        if d in sched:
            sig, target = sched[d]
            if stopped_today:
                target = [c for c in target if c not in stopped_today]
            o = close.loc[d] if execute_at == "close" else open_.loc[d]
            # 매도
            for c in [c for c in pos if c not in target]:
                px = o.get(c, np.nan)
                if np.isnan(px):
                    continue  # 거래정지 등 → 보유 유지
                sh = pos.pop(c)
                stopped_pos.pop(c, None)
                cash += sh * px * (1 - (etf_sell if etf_codes and c in etf_codes else sell_cost))
                trades.append({"date": d, "signal": sig, "code": c, "side": "SELL", "price": float(px),
                               "shares": int(sh)})
            # 매수: 현금을 신규 종목에 균등 배분, 정수 주식
            buys = [c for c in target if c not in pos and not np.isnan(o.get(c, np.nan))]
            if buys and cash > 0:
                per = cash / len(buys)
                if slots:   # 한 자리 금액 = 평가금액/자리수 (하락장에서 자리 일부만 채우면 나머지는 현금)
                    val = cash + sum(sh * close_f.loc[d].get(c, 0) for c, sh in pos.items())
                    per = min(per, val / slots)
                for c in buys:
                    px = float(o[c])
                    bc = etf_buy if etf_codes and c in etf_codes else buy_cost
                    sh = int(per // (px * (1 + bc)))
                    if sh <= 0:
                        continue
                    cash -= sh * px * (1 + bc)
                    pos[c] = sh
                    if stop_dates is not None and sig in stop_dates:
                        stopped_pos[c] = px
                    trades.append({"date": d, "signal": sig, "code": c, "side": "BUY", "price": px, "shares": sh})
        cl = close_f.loc[d]
        eq[d] = cash + sum(sh * cl.get(c, 0) for c, sh in pos.items())
    return pd.Series(eq), trades


def weekly_signal_dates(index: pd.DatetimeIndex, weekday: int | None = None) -> list[pd.Timestamp]:
    """매주 한 번의 신호일.
    weekday=None: 각 주의 마지막 거래일
    weekday=0~4: 그 주에서 해당 요일 이후 첫 거래일 (휴장이면 다음 거래일로 밀림 — 실전 규칙과 동일)"""
    s = pd.Series(index, index=index)
    iso = index.isocalendar()
    if weekday is not None:
        s = s[index.weekday >= weekday]
        iso = s.index.isocalendar()
        return list(s.groupby([iso.year.values, iso.week.values]).min().sort_values())
    return list(s.groupby([iso.year.values, iso.week.values]).max().sort_values())


def metrics(eq: pd.Series, bench: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 10:
        return {}
    b = bench.reindex(eq.index).ffill()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dr = eq.pct_change().dropna()
    wk = eq.resample("W-FRI").last().pct_change().dropna()
    bwk = b.resample("W-FRI").last().pct_change().dropna().reindex(wk.index)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    bcagr = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = dr.std() * np.sqrt(252)
    return {
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "cagr": float(cagr), "bench_cagr": float(bcagr), "excess_cagr": float(cagr - bcagr),
        "vol": float(vol), "sharpe": float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0,
        "mdd": float((eq / eq.cummax() - 1).min()),
        "weekly_win_rate": float((wk > 0).mean()),
        "weekly_beat_bench": float((wk > bwk).mean()),
        "years": round(yrs, 1),
    }
