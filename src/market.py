"""무료 시장 흐름 (최근 2주): 수급·지표·뉴스 헤드라인 → docs/data/market.json

API 키 없이 동작합니다. 각 항목은 실패해도 나머지는 계속 만듭니다.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

from . import config as C
from .data import _get, get_ohlcv, SYNTHETIC

KST = timezone(timedelta(hours=9))
OUT = C.DOCS_DATA / "market.json"
WINDOW_DAYS = 14

# (키, 이름, 출처, 심볼, 단위)
INDICATORS = [
    ("kospi", "KOSPI", "naver", "KOSPI", "pt"),
    ("kosdaq", "KOSDAQ", "naver", "KOSDAQ", "pt"),
    ("usdkrw", "원/달러 환율", "yahoo", "KRW=X", "원"),
    ("spx", "S&P500", "yahoo", "^GSPC", "pt"),
    ("ndx", "나스닥", "yahoo", "^IXIC", "pt"),
    ("sox", "필라델피아 반도체", "yahoo", "^SOX", "pt"),
    ("us10y", "미국 10년물 금리", "yahoo", "^TNX", "%"),
    ("wti", "WTI 유가", "yahoo", "CL=F", "$"),
    ("vix", "VIX(공포지수)", "yahoo", "^VIX", "pt"),
]

RSS = [
    ("연합뉴스", "https://www.yna.co.kr/rss/economy.xml"),
    ("한국경제", "https://www.hankyung.com/feed/finance"),
    ("매일경제", "https://www.mk.co.kr/rss/50200011/"),
]
NEWS_KEYS = r"코스피|코스닥|증시|주가|외국인|기관|금리|환율|원[·/]?달러|반도체|연준|FOMC|수출|국채|유가|공매도|상장|실적|한은|기준금리|나스닥|S&P"


def _yahoo(symbol: str) -> pd.Series:
    r = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
             params={"range": "3mo", "interval": "1d"})
    res = r.json()["chart"]["result"][0]
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("Asia/Seoul").normalize().tz_localize(None)
    s = pd.Series(res["indicators"]["quote"][0]["close"], index=ts, dtype=float).dropna()
    s = s[~s.index.duplicated(keep="last")]
    if symbol == "^TNX" and s.median() > 20:      # 일부 응답은 ×10 단위
        s = s / 10
    return s


def _series(src, sym) -> pd.Series:
    if SYNTHETIC:
        rng = np.random.default_rng(abs(hash(sym)) % 1000)
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=60)
        return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))), index=idx)
    if src == "naver":
        return get_ohlcv(sym, 60)["close"]
    return _yahoo(sym)


def indicators(asof: pd.Timestamp):
    out = []
    start = asof - pd.Timedelta(days=WINDOW_DAYS)
    for key, name, src, sym, unit in INDICATORS:
        try:
            s = _series(src, sym).loc[:asof]
            if len(s) < 6:
                continue
            last = float(s.iloc[-1])
            w1 = s.loc[:asof - pd.Timedelta(days=7)]
            w2 = s.loc[:start]
            item = {"key": key, "name": name, "unit": unit, "last": round(last, 4), "date": str(s.index[-1].date()),
                    "chg_1w": (last / float(w1.iloc[-1]) - 1) if len(w1) else None,
                    "chg_2w": (last / float(w2.iloc[-1]) - 1) if len(w2) else None,
                    "spark": [round(float(x), 4) for x in s.loc[start:].tolist()]}
            if key == "us10y":   # 금리는 %p 변화로
                item["chg_1w"] = (last - float(w1.iloc[-1])) if len(w1) else None
                item["chg_2w"] = (last - float(w2.iloc[-1])) if len(w2) else None
                item["chg_unit"] = "%p"
            out.append(item)
        except Exception as e:  # noqa
            print(f"[market] {name} 실패: {e!r}")
    return out


def flows(asof: pd.Timestamp):
    """KOSPI 투자자별 일별 순매수 대금(억원), 최근 2주"""
    start = asof - pd.Timedelta(days=WINDOW_DAYS)
    if SYNTHETIC:
        idx = pd.bdate_range(start + pd.Timedelta(days=1), asof)
        rng = np.random.default_rng(3)
        df = pd.DataFrame({"foreign": rng.normal(-1500, 3000, len(idx)), "inst": rng.normal(300, 1500, len(idx)),
                           "indiv": rng.normal(1000, 2500, len(idx))}, index=idx)
    else:
        try:
            from pykrx import stock
            raw = stock.get_market_trading_value_by_date(start.strftime("%Y%m%d"), asof.strftime("%Y%m%d"), "KOSPI")
            df = pd.DataFrame({"foreign": raw["외국인합계"], "inst": raw["기관합계"], "indiv": raw["개인"]}) / 1e8
        except Exception as e:  # noqa
            print(f"[market] 수급 실패: {e!r}")
            return None
    df = df.loc[start + pd.Timedelta(days=1):asof]
    if df.empty:
        return None
    sign = np.sign(df["foreign"].values[::-1])
    streak = 0
    for v in sign:
        if v == sign[0] and v != 0:
            streak += 1
        else:
            break
    last5 = df.tail(5)
    return {"dates": [str(d.date()) for d in df.index],
            "foreign": [round(float(x)) for x in df["foreign"]], "inst": [round(float(x)) for x in df["inst"]],
            "indiv": [round(float(x)) for x in df["indiv"]],
            "sum_2w": {k: round(float(df[k].sum())) for k in ("foreign", "inst", "indiv")},
            "sum_1w": {k: round(float(last5[k].sum())) for k in ("foreign", "inst", "indiv")},
            "foreign_streak": int(streak * (1 if sign[0] > 0 else -1)) if len(sign) else 0}


def news(asof: pd.Timestamp, limit=10):
    if SYNTHETIC:
        return [{"title": f"(가상) 코스피 외국인 매도에 하락 마감 {i}", "link": "https://example.com", "source": "가상",
                 "date": str((asof - pd.Timedelta(days=i)).date())} for i in range(6)]
    start = asof - pd.Timedelta(days=WINDOW_DAYS)
    items, seen = [], set()
    for src, url in RSS:
        try:
            r = _get(url, timeout=15)
            root = ET.fromstring(r.content)
            for it in root.iter("item"):
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                pub = it.findtext("pubDate") or it.findtext("{http://purl.org/dc/elements/1.1/}date") or ""
                try:
                    d = parsedate_to_datetime(pub).astimezone(KST).replace(tzinfo=None)
                except Exception:
                    try:
                        d = pd.Timestamp(pub).tz_localize(None).to_pydatetime()
                    except Exception:
                        continue
                if not title or d < start.to_pydatetime() or not re.search(NEWS_KEYS, title):
                    continue
                key = re.sub(r"\W", "", title)[:30]
                if key in seen:
                    continue
                seen.add(key)
                items.append({"title": title, "link": link, "source": src, "date": d.strftime("%Y-%m-%d"),
                              "ts": d.isoformat()})
        except Exception as e:  # noqa
            print(f"[market] 뉴스 {src} 실패: {e!r}")
    items.sort(key=lambda x: x["ts"], reverse=True)
    return [{k: v for k, v in x.items() if k != "ts"} for x in items[:limit]]


def _won_eok(x):
    return f"{x / 1e4:+.1f}조" if abs(x) >= 1e4 else f"{x:+,.0f}억"


def summary(ind, fl):
    parts = []
    if fl:
        st = fl["foreign_streak"]
        parts.append(f"외국인 2주 순매수 {_won_eok(fl['sum_2w']['foreign'])}"
                     + (f"({abs(st)}일 연속 {'순매수' if st > 0 else '순매도'})" if abs(st) >= 3 else ""))
        parts.append(f"기관 {_won_eok(fl['sum_2w']['inst'])}")
    by = {i["key"]: i for i in ind}
    if "kospi" in by and by["kospi"]["chg_2w"] is not None:
        parts.append(f"KOSPI 2주 {by['kospi']['chg_2w']:+.1%}")
    if "usdkrw" in by and by["usdkrw"]["chg_2w"] is not None:
        parts.append(f"원/달러 {by['usdkrw']['last']:,.0f}원({by['usdkrw']['chg_2w']:+.1%})")
    if "sox" in by and by["sox"]["chg_2w"] is not None:
        parts.append(f"美 반도체지수 {by['sox']['chg_2w']:+.1%}")
    if "us10y" in by and by["us10y"]["chg_2w"] is not None:
        parts.append(f"美 10년물 {by['us10y']['last']:.2f}%({by['us10y']['chg_2w']:+.2f}%p)")
    if "vix" in by:
        v = by["vix"]["last"]
        parts.append(f"VIX {v:.1f}" + (" (불안 높음)" if v >= 25 else " (안정)" if v < 15 else ""))
    return " · ".join(parts)


def build(asof: pd.Timestamp):
    ind = indicators(asof)
    fl = flows(asof)
    nw = news(asof)
    out = {"asof": str(asof.date()), "window_days": WINDOW_DAYS,
           "updated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
           "summary": summary(ind, fl), "indicators": ind, "flows": fl, "news": nw}
    C.DOCS_DATA.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[market] 지표 {len(ind)}개, 수급 {'있음' if fl else '없음'}, 뉴스 {len(nw)}건 · {out['summary']}")
    return out


if __name__ == "__main__":
    build(pd.Timestamp(datetime.now(KST).date()))
