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
    ("연합뉴스", "https://www.yna.co.kr/rss/market.xml"),
    ("한국경제", "https://www.hankyung.com/feed/finance"),
    ("한국경제", "https://www.hankyung.com/feed/economy"),
    ("매일경제", "https://www.mk.co.kr/rss/50200011/"),
    ("매일경제", "https://www.mk.co.kr/rss/30100041/"),
]
KW_DAYS = 7
# 키워드 집계에서 뺄 일반어(시장 전반·동사·수식어 등)
STOP = set("""
코스피 코스닥 증시 주가 주식 시장 투자 투자자 외국인 기관 개인 상승 하락 급등 급락 강세 약세 출발 마감 장중 개장 종가
보합 반등 등락 거래 매수 매도 순매수 순매도 지수 종목 특징주 오늘 내일 올해 내년 지난해 이번 이번주 최근 전망 우려 기대 가능성
확대 감소 증가 돌파 회복 유지 지속 영향 관련 대비 기준 발표 공개 추진 예정 계획 결정 논의 검토 개최 출시 진행 사실상 역대
최고 최저 최대 신고가 기록 만에 이후 이상 이하 미만 가운데 속에 위해 통해 대한 따른 위한 그리고 하지만 단독 종합 속보 포토 영상
증권 증권사 리포트 목표가 목표주가 상향 하향 실적 영업이익 매출 분기 전년 동기 흑자 적자 시그널 MK시그널 오전 오후 시황
신고 기대감 관심 주목 수혜 수혜주 테마 테마주 랠리 훈풍 찬바람 호재 악재 상한가 하한가 껑충 뚝 쑥 털썩
억원 조원 만원 달러 원화 원달러 퍼센트 포인트 대표 회장 사장 정부 대통령 장관 위원회 국회 서울 한국 국내 해외 글로벌 美 中 日
""".split())
JOSA = ("에서는", "으로는", "에서", "으로", "까지", "부터", "에게", "보다", "처럼", "마저", "조차", "은", "는", "이", "가",
        "을", "를", "에", "의", "도", "로", "와", "과", "만", "株", "주")
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
            r = _get(url, retries=1, timeout=10)
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


def _naver_mainnews(asof: pd.Timestamp, days=KW_DAYS, pages=3):
    out, fails = [], 0
    for i in range(days + 1):
        if fails >= 2:                   # 접속이 막히면 바로 포기(RSS로 대체)
            print("[market] 네이버 주요뉴스 접속 실패 → RSS만 사용")
            break
        d = (asof - pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        for pg in range(1, pages + 1):
            try:
                html = _get("https://finance.naver.com/news/mainnews.naver", params={"date": d, "page": pg},
                            retries=1, timeout=8).content.decode("euc-kr", errors="ignore")
            except Exception:
                fails += 1
                break
            found = re.findall(r'articleSubject">\s*<a href="([^"]+)"[^>]*>([^<]+)</a>', html)
            for link, title in found:
                out.append({"title": re.sub(r"\s+", " ", title).strip(), "date": d, "source": "네이버 금융",
                            "link": ("https://finance.naver.com" + link) if link.startswith("/") else link})
            if len(found) < 15:
                break
    return out


def _tokens(title: str):
    t = re.sub(r"\[[^\]]*\]|【[^】]*】|<[^>]*>|\([^)]*\)", " ", title)
    toks = []
    for w in re.findall(r"[가-힣A-Za-z][가-힣A-Za-z0-9]{1,}", t):
        for j in JOSA:
            if w.endswith(j) and len(w) - len(j) >= 2:
                w = w[: -len(j)]
                break
        if len(w) >= 2 and w not in STOP and not re.fullmatch(r"[0-9]+", w):
            toks.append(w)
    return toks


def weekly_keywords(asof: pd.Timestamp, M: dict | None):
    """최근 7일 기사 제목 키워드 빈도 + 최다 키워드 관련 종목 1개"""
    start = asof - pd.Timedelta(days=KW_DAYS)
    arts = []
    if SYNTHETIC:
        arts = [{"title": t, "date": str(asof.date()), "source": "가상", "link": "https://example.com"}
                for t in ["(가상) 조선 수주 호황 가상종목004 신고가", "(가상) 조선업 슈퍼사이클 가상종목004·가상종목049",
                          "(가상) 반도체 수출 증가", "(가상) 조선 빅3 수주 잔고 사상 최대", "(가상) 방산 수출 계약"]]
    else:
        arts = _naver_mainnews(asof)
        for src, url in RSS:                                   # RSS도 합침(7일 범위)
            try:
                root = ET.fromstring(_get(url, retries=1, timeout=10).content)
                for it in root.iter("item"):
                    try:
                        d = parsedate_to_datetime(it.findtext("pubDate") or "").astimezone(KST).replace(tzinfo=None)
                    except Exception:
                        continue
                    if d >= start.to_pydatetime():
                        arts.append({"title": (it.findtext("title") or "").strip(), "date": d.strftime("%Y-%m-%d"),
                                     "source": src, "link": (it.findtext("link") or "").strip()})
            except Exception:
                pass
    seen, uniq = set(), []
    for a in arts:
        k = re.sub(r"\W", "", a["title"])[:30]
        if a["title"] and k not in seen:
            seen.add(k)
            uniq.append(a)
    if not uniq:
        return None
    names = dict(M["names"]) if M else {}
    name2code = {v: k for k, v in names.items() if len(v) >= 2}
    kw_cnt, co_cnt = {}, {}
    for a in uniq:
        toks = set(_tokens(a["title"]))
        cos = {n for n in name2code if n in a["title"]}
        a["companies"] = sorted(cos)
        for n in cos:
            co_cnt[n] = co_cnt.get(n, 0) + 1
        for w in toks:
            if w in name2code:                                 # 회사명은 키워드 대신 '많이 언급된 종목'으로
                continue
            kw_cnt[w] = kw_cnt.get(w, 0) + 1
    top_kw = sorted(kw_cnt.items(), key=lambda x: -x[1])[:12]
    top_co = sorted(co_cnt.items(), key=lambda x: -x[1])[:8]

    pick = None
    if M is not None and top_kw:
        f = M["f"]
        for kw, n in top_kw[:3]:                               # 상위 키워드부터 관련 종목 탐색
            rel = [a for a in uniq if kw in a["title"]]
            cand = {}
            for a in rel:
                for c in a["companies"]:
                    cand[c] = cand.get(c, 0) + 1
            rows = []
            for nm, k in cand.items():
                code = name2code[nm]
                if code not in f.index or code in C.EXCLUDE:
                    continue
                r = f.loc[code]
                ok = (r["close"] > r["ma20"]) and (r["ret_21"] > 0) and (r["ret_21"] <= 0.40)
                rows.append((ok, k, float(r["close"] / r["high_252"]), float(r["ret_21"]), code, nm, r))
            rows = [x for x in rows if x[0]]
            if not rows:
                continue
            rows.sort(key=lambda x: (-x[1], -x[2]))               # 관련 기사 언급 많은 순 → 52주 고가 근접
            ok, k, nh, r21, code, nm, r = rows[0]
            pick = {"keyword": kw, "keyword_count": n, "code": code, "name": nm, "mentions": k,
                    "sector": M["sectors"].get(code), "close": float(r["close"]), "ret_21": r21,
                    "near_high": nh, "above_ma20": True,
                    "flow_f": None if pd.isna(r.get("flow_f")) else float(r["flow_f"]),
                    "articles": [{"title": a["title"], "link": a["link"], "date": a["date"], "source": a["source"]}
                                 for a in rel if nm in a["title"]][:3]}
            break
    return {"days": KW_DAYS, "articles": len(uniq), "top_keywords": top_kw, "top_companies": top_co, "pick": pick}


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


def build(asof: pd.Timestamp, M: dict | None = None):
    ind = indicators(asof)
    fl = flows(asof)
    nw = news(asof)
    try:
        kw = weekly_keywords(asof, M)
    except Exception as e:  # noqa
        print(f"[market] 키워드 실패: {e!r}")
        kw = None
    out = {"asof": str(asof.date()), "window_days": WINDOW_DAYS,
           "updated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
           "summary": summary(ind, fl), "indicators": ind, "flows": fl, "news": nw, "keywords": kw}
    if kw:
        print(f"[market] 키워드 상위: {kw['top_keywords'][:5]} · 종목 언급: {kw['top_companies'][:5]} · "
              f"추천: {kw['pick'] and (kw['pick']['keyword'], kw['pick']['name'])}")
    C.DOCS_DATA.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[market] 지표 {len(ind)}개, 수급 {'있음' if fl else '없음'}, 뉴스 {len(nw)}건 · {out['summary']}")
    return out


if __name__ == "__main__":
    build(pd.Timestamp(datetime.now(KST).date()))
