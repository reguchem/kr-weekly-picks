"""테마 ETF 순환: 점수, 선정(같은 테마 중복 제외), 비용"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

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
        self.ma20 = c.rolling(20, min_periods=18).mean()
        self.high_126 = c.rolling(126, min_periods=100).max()
        self.tv20 = (c * volume.reindex_like(c)).rolling(20, min_periods=15).mean()
        self.hist = c.notna().cumsum()

    def table(self, t) -> pd.DataFrame:
        return pd.DataFrame({"close": self.close.loc[t], "r21": self.ret_21.loc[t], "r63": self.ret_63.loc[t],
                             "ma60": self.ma60.loc[t], "ma20": self.ma20.loc[t], "hi": self.high_126.loc[t], "tv": self.tv20.loc[t],
                             "hist": self.hist.loc[t]})

    def scores(self, t) -> pd.Series:
        f = self.table(t)
        f = f[(f["hist"] >= ETF_MIN_HIST) & (f["tv"] >= ETF_MIN_TV) & f["r63"].notna() & f["ma60"].notna()]
        f = f[f["close"] > f["ma60"]]                         # 절대 모멘텀: 60일선 위만
        if f.empty:
            return pd.Series(dtype=float)
        r = lambda x: x.rank(pct=True)
        s = 0.4 * r(f["r21"]) + 0.4 * r(f["r63"]) + 0.2 * r(f["close"] / f["hi"])
        return s.sort_values(ascending=False)


def etf_select(sc: pd.Series, holdings, n, buffer, E: EtfFactors, t, affordable=None):
    rank = {c: i + 1 for i, c in enumerate(sc.index)}
    picks = [h for h in holdings if rank.get(h, 10 ** 9) <= buffer][:n]
    window = E.ret.loc[:t].tail(120)
    for c in sc.index:
        if len(picks) >= n:
            break
        if c in picks or (affordable is not None and c not in affordable):
            continue
        if picks:
            cr = window[picks + [c]].corr()[c].drop(c)
            if (cr > CORR_MAX).any():
                continue
        picks.append(c)
    return picks


def market_ok(kospi: pd.Series, ma_n, t) -> bool:
    """KOSPI 종가가 ma_n일 이동평균 이상이면 True (투자), 아래면 False (현금)"""
    if not ma_n:
        return True
    k = kospi.loc[:t]
    ma = k.rolling(ma_n, min_periods=int(ma_n * 0.9)).mean().iloc[-1]
    return True if np.isnan(ma) else bool(k.iloc[-1] >= ma)


def kospi_rets(kospi: pd.Series, t):
    k = kospi.loc[:t]
    r = lambda n: float(k.iloc[-1] / k.iloc[-1 - n] - 1) if len(k) > n else 0.0
    return r(21), r(63)


def bear_ok_stocks(f: pd.DataFrame, k21: float, k63: float) -> set:
    """하락장 모드 종목 조건: 20·60일선 위, 3개월 수익률이 KOSPI+10%p 이상, 1개월도 KOSPI보다 강함"""
    ok = ((f["close"] > f["ma20"]) & (f["close"] > f["ma60"]) &
          (f["ret_63"] - k63 >= C.BEAR_RS_STOCK) & (f["ret_21"] > k21))
    return set(f.index[ok.fillna(False)])


def bear_ok_etfs(E: EtfFactors, t, k21: float, k63: float) -> set:
    f = E.table(t)
    ok = ((f["close"] > f["ma20"]) & (f["close"] > f["ma60"]) &
          (f["r63"] - k63 >= C.BEAR_RS_ETF) & (f["r21"] > k21) & (f["tv"] >= ETF_MIN_TV))
    return set(f.index[ok.fillna(False)])
