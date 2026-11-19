"""시세·종목·펀더멘털 데이터 수집.

1순위: 네이버 금융(로그인 불필요), 2순위: pykrx / FinanceDataReader.
환경변수 SYNTHETIC=1 이면 네트워크 없이 가상 데이터로 동작합니다(테스트용).
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

SYNTHETIC = os.environ.get("SYNTHETIC") == "1"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
_session = requests.Session()
_session.headers.update(UA)

ETF_PREFIX = ("KODEX", "TIGER", "KBSTAR", "RISE", "ACE", "SOL", "HANARO", "KOSEF", "ARIRANG",
              "PLUS", "KIWOOM", "TIMEFOLIO", "1Q", "BNK", "WON", "히어로즈", "마이다스", "에셋플러스",
              "TRUSTON", "UNICORN", "파워", "마이티", "VITA", "DAISHIN343", "KTOP", "ITF", "HK")


def _get(url, params=None, retries=3, timeout=15):
    for i in range(retries):
        try:
            r = _session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def _num(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = re.sub(r"[^0-9.\-]", "", str(x))
    return float(s) if s not in ("", "-", ".") else np.nan


def _clean_universe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["mcap"]).copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    bad = (
        ~df["code"].str.endswith("0")                       # 우선주
        | df["name"].str.contains("스팩|ETN|리츠|인프라", regex=True)
        | df["name"].str.upper().str.startswith(ETF_PREFIX)
    )
    df = df[~bad].drop_duplicates("code")
    return df.sort_values("mcap", ascending=False).reset_index(drop=True)


# ───────────────────────── 유니버스 ─────────────────────────
def _pages(top_n, per):
    return int(top_n * 1.6 / per) + 2      # 우선주·ETF·스팩 제외분 여유


def _universe_naver_api(top_n, markets):
    rows = []
    for market, pages in ((m, _pages(top_n, 100)) for m in markets):
        for p in range(1, pages + 1):
            r = _get(f"https://m.stock.naver.com/api/stocks/marketValue/{market}",
                     params={"page": p, "pageSize": 100})
            for s in r.json().get("stocks", []):
                if s.get("stockEndType", "stock") != "stock":
                    continue
                rows.append({"code": s["itemCode"], "name": s["stockName"], "market": market,
                             "mcap": _num(s.get("marketValue"))})
    return rows


def _universe_naver_html(top_n, markets):
    rows = []
    for sosok, market, pages in ((0 if m == "KOSPI" else 1, m, _pages(top_n, 50)) for m in markets):
        for p in range(1, pages + 1):
            r = _get("https://finance.naver.com/sise/sise_market_sum.naver",
                     params={"sosok": sosok, "page": p})
            html = r.content.decode("euc-kr", errors="ignore")
            codes = re.findall(r'href="/item/main\.naver\?code=(\d{6})" class="tltle">([^<]+)<', html)
            tables = pd.read_html(pd.io.common.StringIO(html))
            t = next(t for t in tables if "시가총액" in t.columns).dropna(subset=["종목명"])
            mcap = dict(zip(t["종목명"], t["시가총액"]))
            for code, name in codes:
                rows.append({"code": code, "name": name, "market": market, "mcap": _num(mcap.get(name))})
    return rows


def _universe_pykrx(top_n, markets):
    from pykrx import stock
    d = stock.get_nearest_business_day_in_a_week()
    rows = []
    for market in markets:
        cap = stock.get_market_cap(d, market=market)
        for code, r in cap.iterrows():
            rows.append({"code": code, "name": stock.get_market_ticker_name(code),
                         "market": market, "mcap": float(r["시가총액"])})
    return rows


def get_universe(top_n: int, markets=None) -> pd.DataFrame:
    """시가총액 상위 종목. columns: code, name, market, mcap"""
    from .config import UNIVERSE_MARKETS
    markets = tuple(markets or UNIVERSE_MARKETS)
    if SYNTHETIC:
        return _synthetic_universe(top_n)
    errors = []
    for fn in (_universe_naver_api, _universe_naver_html, _universe_pykrx):
        try:
            df = _clean_universe(pd.DataFrame(fn(top_n, markets)))
            if len(df) >= min(top_n, 100):
                print(f"[universe] {fn.__name__}: {'+'.join(markets)} 상위 {min(top_n, len(df))} / 수집 {len(df)} 종목")
                return df.head(top_n)
            errors.append(f"{fn.__name__}: {len(df)}개뿐")
        except Exception as e:  # noqa
            errors.append(f"{fn.__name__}: {e!r}")
    raise RuntimeError("유니버스 수집 실패: " + " | ".join(errors))


# ───────────────────────── 일봉 시세 ─────────────────────────
def _ohlcv_naver(symbol, count):
    r = _get("https://fchart.stock.naver.com/sise.nhn",
             params={"symbol": symbol, "timeframe": "day", "count": count, "requestType": 0})
    items = re.findall(r'data="([^"]+)"', r.text)
    rec = [x.split("|") for x in items]
    df = pd.DataFrame(rec, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce")
    return df


def _ohlcv_fdr(symbol, count):
    import FinanceDataReader as fdr
    sym = "KS11" if symbol == "KOSPI" else ("KQ11" if symbol == "KOSDAQ" else symbol)
    start = (pd.Timestamp.today() - pd.Timedelta(days=int(count * 1.5) + 10)).strftime("%Y-%m-%d")
    df = fdr.DataReader(sym, start)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index.name = "date"
    return df.tail(count)


def get_ohlcv(symbol: str, count: int) -> pd.DataFrame:
    """일봉(open, high, low, close, volume). symbol: 6자리 코드 또는 'KOSPI'"""
    if SYNTHETIC:
        if str(symbol).startswith("4"):
            return _synthetic_etf_ohlcv(symbol, count)
        return _synthetic_ohlcv(symbol, count)
    last = None
    for fn in (_ohlcv_naver, _ohlcv_fdr):
        try:
            df = fn(symbol, count)
            df = df[(df["close"] > 0)]
            if len(df):
                return df
        except Exception as e:  # noqa
            last = e
    print(f"[ohlcv] {symbol} 실패: {last!r}")
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def get_price_panel(codes, count, workers=8):
    """여러 종목 일봉 → {'open': DataFrame(date×code), 'close':..., 'volume':...}"""
    codes = list(dict.fromkeys(codes))

    def job(c):
        time.sleep(0.05)
        return c, get_ohlcv(c, count)

    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(job, codes))
    out = {}
    for f in ("open", "close", "volume"):
        out[f] = pd.DataFrame({c: d[f] for c, d in res.items() if len(d)}).sort_index()
    # 시가 0/결측(거래정지 등)은 종가로 대체
    out["open"] = out["open"].where(out["open"] > 0, out["close"])
    print(f"[prices] {len(out['close'].columns)}/{len(codes)} 종목, {len(out['close'])} 거래일")
    return out


# ───────────────────────── 펀더멘털 ─────────────────────────
def get_fundamentals(date: pd.Timestamp) -> pd.DataFrame | None:
    """해당일 전 종목 PER/PBR (pykrx). 실패하면 None → 가치 팩터 생략."""
    if SYNTHETIC:
        return _synthetic_fundamentals(date)
    try:
        from pykrx import stock
        parts = []
        for mk in ("KOSPI", "KOSDAQ"):          # "ALL"은 빈 결과가 와서 시장별로 조회
            for k in range(3):
                try:
                    x = stock.get_market_fundamental(date.strftime("%Y%m%d"), market=mk)
                except Exception:
                    x = None
                if x is not None and not x.empty and "PER" in x:
                    parts.append(x)
                    break
                time.sleep(2 * (k + 1))
            time.sleep(0.3)
        if not parts:
            return None
        df = pd.concat(parts)
        df.index = df.index.astype(str).str.zfill(6)
        return df[["PER", "PBR"]].replace(0, np.nan)
    except Exception as e:  # noqa
        print(f"[fundamental] {date.date()} 실패: {e!r}")
        return None


# ───────────────────────── 가상 데이터 (테스트용) ─────────────────────────
_SYN_N = 150
_SYN_DAYS = pd.bdate_range("2014-01-01", pd.Timestamp.today().normalize())


def _synthetic_universe(top_n):
    rng = np.random.default_rng(0)
    codes = [f"{100000 + i * 10:06d}" for i in range(_SYN_N)]
    return pd.DataFrame({"code": codes, "name": [f"가상종목{i:03d}" for i in range(_SYN_N)],
                         "market": rng.choice(["KOSPI", "KOSDAQ"], _SYN_N),
                         "mcap": np.sort(rng.lognormal(10, 1, _SYN_N))[::-1]}).head(top_n)


def _synthetic_ohlcv(symbol, count):
    seed = 7 if symbol == "KOSPI" else int(symbol)
    rng = np.random.default_rng(seed)
    n = len(_SYN_DAYS)
    mkt = np.random.default_rng(7).normal(0.0002, 0.011, n)
    if symbol == "KOSPI":
        r = mkt
        base = 2000
    else:
        beta, vol, drift = rng.uniform(0.6, 1.4), rng.uniform(0.01, 0.03), rng.normal(0.0002, 0.0004)
        r = drift + beta * mkt + rng.normal(0, vol, n) - 0.5 * vol ** 2
        base = rng.uniform(5000, 200000)
    close = base * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    vol_ = rng.integers(200000, 3000000, n)
    df = pd.DataFrame({"open": open_.round(), "high": np.maximum(open_, close).round(),
                       "low": np.minimum(open_, close).round(), "close": close.round(),
                       "volume": vol_}, index=_SYN_DAYS)
    df.index.name = "date"
    if os.environ.get("SYN_END"):
        df = df.loc[:os.environ["SYN_END"]]
    return df.tail(count)


def _synthetic_fundamentals(date):
    codes = _synthetic_universe(_SYN_N)["code"]
    rng = np.random.default_rng(int(date.strftime("%Y%m")))
    return pd.DataFrame({"PER": rng.uniform(3, 40, len(codes)), "PBR": rng.uniform(0.3, 4, len(codes))},
                        index=codes)


# ───────────────────────── 업종 분류 ─────────────────────────
def _sectors_naver():
    r = _get("https://finance.naver.com/sise/sise_group.naver", params={"type": "upjong"})
    html = r.content.decode("euc-kr", errors="ignore")
    groups = dict(re.findall(r'sise_group_detail\.naver\?type=upjong&(?:amp;)?no=(\d+)"[^>]*>([^<]+)<', html))
    print(f"[sector] 네이버 업종 {len(groups)}개")
    out = {}
    for no, name in groups.items():
        h = _get("https://finance.naver.com/sise/sise_group_detail.naver",
                 params={"type": "upjong", "no": no}).content.decode("euc-kr", errors="ignore")
        for code in re.findall(r'/item/main\.naver\?code=(\d{6})', h):
            out.setdefault(code, name.strip())
        time.sleep(0.1)
    return out


def _sectors_fdr():
    import FinanceDataReader as fdr
    df = fdr.StockListing("KRX-DESC")
    col = "Industry" if "Industry" in df else "Sector"
    return {str(c).zfill(6): s for c, s in zip(df["Code"], df[col]) if isinstance(s, str)}


def _sectors_krx():
    """KRX 업종분류 현황 (pykrx, KRX 로그인 필요). 업종 수는 적지만(약 20~30) 안정적."""
    if not (os.environ.get("KRX_ID") and os.environ.get("KRX_PW")):
        return {}
    from pykrx import stock
    d = stock.get_nearest_business_day_in_a_week()
    out = {}
    for mk in ("KOSPI", "KOSDAQ"):
        df = stock.get_market_sector_classifications(d, mk)
        if df is None or df.empty:
            continue
        df = df.reset_index() if "종목코드" not in df.columns else df
        code_col = "종목코드" if "종목코드" in df.columns else df.columns[0]
        for c, sct in zip(df[code_col], df["업종명"]):
            out[str(c).zfill(6)] = str(sct)
    return out


def get_sectors() -> dict:
    """종목코드 → 업종명 (현재 기준)"""
    if SYNTHETIC:
        codes = _synthetic_universe(_SYN_N)["code"]
        names = ["반도체", "2차전지", "조선", "방산", "바이오", "은행", "자동차", "화학", "인터넷", "건설",
                 "철강", "유통", "게임", "전력기기", "화장품"]
        return {c: names[i % len(names)] for i, c in enumerate(codes)}
    for fn in (_sectors_fdr, _sectors_naver, _sectors_krx):
        try:
            m = fn()
            print(f"[sector] {fn.__name__}: {len(m)} 종목, {len(set(m.values()))} 업종")
            if len(m) > 500:
                return m
        except Exception as e:  # noqa
            print(f"[sector] {fn.__name__} 실패: {e!r}")
    return {}


# ───────────────────────── 투자자별 순매수 ─────────────────────────
_FLOW_ERR = None


def check_krx_login():
    """KRX 로그인 상태 점검 (로그용). KRX_ID / KRX_PW 환경변수 필요."""
    if SYNTHETIC:
        return True
    has = bool(os.environ.get("KRX_ID")) and bool(os.environ.get("KRX_PW"))
    try:
        from pykrx import stock
        d = (pd.Timestamp.today() - pd.Timedelta(days=10)).strftime("%Y%m%d")
        e = pd.Timestamp.today().strftime("%Y%m%d")
        df = stock.get_market_trading_value_by_date(d, e, "005930")
        ok = df is not None and not df.empty
        f = stock.get_market_fundamental(stock.get_nearest_business_day_in_a_week(), market="KOSPI")
        ok2 = f is not None and not f.empty
        print(f"[krx] 계정 설정 {'있음' if has else '없음'} · 수급 조회 {'성공' if ok else '실패'} · "
              f"PER/PBR 조회 {'성공' if ok2 else '실패'}")
        return ok
    except Exception as e:  # noqa
        print(f"[krx] 계정 설정 {'있음' if has else '없음'} · 조회 실패: {e!r}")
        return False
def get_flow_history(code: str, start: str, end: str) -> pd.DataFrame | None:
    """일별 외국인·기관 순매수 대금(원). columns: foreign, inst  (pykrx/KRX)"""
    if SYNTHETIC:
        return _synthetic_flow(code)
    try:
        from pykrx import stock
        parts = []
        s0, e0 = pd.Timestamp(start), pd.Timestamp(end)
        while s0 <= e0:                      # KRX 조회 기간 제한 대비 2년 단위로 나눠 조회
            s1 = min(s0 + pd.DateOffset(years=2) - pd.Timedelta(days=1), e0)
            for k in range(4):               # KRX가 빈 응답을 주면 잠시 쉬고 재시도
                try:
                    df = stock.get_market_trading_value_by_date(s0.strftime("%Y%m%d"), s1.strftime("%Y%m%d"), code)
                except Exception:
                    df = None
                if df is not None and not df.empty and "외국인합계" in df:
                    parts.append(pd.DataFrame({"foreign": df["외국인합계"], "inst": df["기관합계"]}))
                    break
                time.sleep(2 * (k + 1))
            time.sleep(0.3)
            s0 = s1 + pd.Timedelta(days=1)
        if not parts:
            return None
        out = pd.concat(parts)
        return out[~out.index.duplicated()]
    except Exception as e:  # noqa
        global _FLOW_ERR
        _FLOW_ERR = repr(e)
        return None


def get_flow_recent_naver(code: str, pages: int = 2) -> pd.DataFrame | None:
    """네이버 '외국인·기관 매매' 탭 최근 ~40거래일. 순매매 '주식수'×종가로 대금 환산."""
    rows = []
    try:
        for p in range(1, pages + 1):
            html = _get("https://finance.naver.com/item/frgn.naver",
                        params={"code": code, "page": p}).content.decode("euc-kr", errors="ignore")
            for t in pd.read_html(pd.io.common.StringIO(html)):
                cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
                if any("외국인" in c for c in cols) and any("기관" in c for c in cols):
                    t.columns = cols
                    dcol = next(c for c in cols if "날짜" in c)
                    ccol = next(c for c in cols if "종가" in c)
                    icol = next(c for c in cols if "기관" in c)
                    fcol = next(c for c in cols if "외국인" in c and "순매매" in c)
                    t = t.dropna(subset=[dcol])
                    for _, r in t.iterrows():
                        try:
                            d = pd.to_datetime(str(r[dcol]), format="%Y.%m.%d")
                        except Exception:
                            continue
                        px = _num(r[ccol])
                        rows.append({"date": d, "foreign": _num(r[fcol]) * px, "inst": _num(r[icol]) * px})
                    break
        if not rows:
            return None
        return pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()
    except Exception:
        return None


def get_flow_panel(codes, start: str, end: str, recent_only=False, workers=None):
    """{'foreign': DataFrame(date×code), 'inst': ...} — 실패 종목은 빠짐"""
    if not recent_only and os.environ.get("FLOW_SOURCE") == "naver":
        return None                          # 빠른 실행용: KRX 대신 네이버 최근 수급만 사용
    if not recent_only and not SYNTHETIC and not (os.environ.get("KRX_ID") and os.environ.get("KRX_PW")):
        print("[flow] KRX 계정 없음 → KRX 수급 조회 생략")
        return None
    fails = {"n": 0, "ok": 0}

    def job(c):
        if not recent_only and fails["ok"] == 0 and fails["n"] >= 5:
            return c, None                      # 처음 5종목이 모두 실패하면 KRX 조회 중단(로그인 문제 등)
        df = None if recent_only else get_flow_history(c, start, end)
        if not recent_only:
            fails["n" if df is None else "ok"] += 1
        if df is None and recent_only:
            df = get_flow_history(c, start, end) if SYNTHETIC else get_flow_recent_naver(c)
        return c, df

    # KRX(pykrx)는 동시 요청에 빈 응답을 주므로 순차 조회, 네이버는 병렬
    workers = workers or (6 if recent_only else 1)
    with ThreadPoolExecutor(workers) as ex:
        res = {}
        for i, (c, d) in enumerate(ex.map(job, list(dict.fromkeys(codes)))):
            if d is not None and len(d):
                res[c] = d
            if (i + 1) % 50 == 0:
                print(f"[flow] {i + 1}/{len(codes)} 처리, 성공 {len(res)}")
    if not res:
        print(f"[flow] 수급 데이터 없음 (마지막 오류: {_FLOW_ERR})")
        return None
    print(f"[flow] {len(res)}/{len(codes)} 종목")
    return {k: pd.DataFrame({c: d[k] for c, d in res.items()}).sort_index() for k in ("foreign", "inst")}


def _synthetic_flow(code):
    o = _synthetic_ohlcv(code, 100000)
    rng = np.random.default_rng(int(code) + 1)
    tv = o["close"] * o["volume"]
    ret = o["close"].pct_change().fillna(0)
    f = tv * (0.3 * np.sign(ret.rolling(5).mean().fillna(0)) * 0.1 + rng.normal(0, 0.05, len(o)))
    return pd.DataFrame({"foreign": f, "inst": tv * rng.normal(0, 0.05, len(o))}, index=o.index)


# ───────────────────────── ETF ─────────────────────────
ETF_EXCLUDE = r"레버리지|인버스|2X|3X|선물|커버드콜|위클리|채권|국채|금리|달러|원유|골드|금현물|은현물|TDF|혼합|머니마켓|CD금리|KOFR|SOFR"


def get_theme_etfs(top_n: int = 80) -> pd.DataFrame:
    """국내 업종·테마 ETF (레버리지·인버스·채권 등 제외), 시가총액 상위. columns: code, name, mcap"""
    if SYNTHETIC:
        return _synthetic_etfs()
    r = _get("https://finance.naver.com/api/sise/etfItemList.nhn", params={"etfType": 0})
    try:
        js = r.json()
    except Exception:
        js = json.loads(r.content.decode("euc-kr", errors="ignore"))
    items = js["result"]["etfItemList"]
    df = pd.DataFrame([{"code": x["itemcode"], "name": x["itemname"], "tab": x.get("etfTabCode"),
                        "mcap": _num(x.get("marketSum"))} for x in items])
    df = df[(df["tab"] == 2) & ~df["name"].str.contains(ETF_EXCLUDE, regex=True)]
    df = df.sort_values("mcap", ascending=False).head(top_n).reset_index(drop=True)
    print(f"[etf] 업종·테마 ETF {len(df)}개")
    return df[["code", "name", "mcap"]]


def _synthetic_etfs():
    names = ["반도체", "2차전지", "조선", "방산", "바이오", "은행", "자동차", "화학", "인터넷", "건설",
             "철강", "유통", "게임", "전력기기", "화장품", "원전", "AI", "로봇"]
    rows = []
    for i, n in enumerate(names):
        for j, brand in enumerate(["KODEX", "TIGER"]):
            rows.append({"code": f"{400000 + i * 10 + j:06d}", "name": f"{brand} {n}", "mcap": 1000 - i})
    return pd.DataFrame(rows)


def _synthetic_etf_ohlcv(symbol, count):
    """테마별로 추세가 지속되는(레짐이 있는) 가상 ETF 시세"""
    i = (int(symbol) - 400000) // 10
    rng = np.random.default_rng(1000 + i)
    n = len(_SYN_DAYS)
    mkt = np.random.default_rng(7).normal(0.0002, 0.011, n)
    regime = np.repeat(rng.normal(0, 0.0015, n // 60 + 1), 60)[:n]
    noise_seed = int(symbol)
    r = 0.9 * mkt + regime + np.random.default_rng(noise_seed).normal(0, 0.006, n) + rng.normal(0, 0.008, n)
    start = 400 + 300 * (i % 5)                      # 상장 시점을 다르게
    close = 10000 * np.exp(np.cumsum(r))
    close[:start] = np.nan
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": np.where(np.isnan(close), np.nan, 300000)}, index=_SYN_DAYS).dropna()
    df.index.name = "date"
    if os.environ.get("SYN_END"):
        df = df.loc[:os.environ["SYN_END"]]
    return df.tail(count)
