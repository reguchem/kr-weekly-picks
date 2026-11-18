"""KRX 조회 진단: 기간·시점별로 어떤 요청이 실패하는지 확인"""
import time, traceback
import pandas as pd
from pykrx import stock
from pykrx.website.comm import auth

def t(label, fn):
    t0 = time.time()
    try:
        df = fn()
        n = 0 if df is None else len(df)
        cols = [] if df is None else list(df.columns)[:6]
        print(f"OK   {label}: {n}행 {cols} ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"FAIL {label}: {e!r} ({time.time()-t0:.1f}s)")

t("수급 10일 005930", lambda: stock.get_market_trading_value_by_date("20261105", "20261117", "005930"))
t("수급 3개월", lambda: stock.get_market_trading_value_by_date("20260801", "20261117", "005930"))
t("수급 1년", lambda: stock.get_market_trading_value_by_date("20251117", "20261117", "005930"))
t("수급 2년", lambda: stock.get_market_trading_value_by_date("20241118", "20261117", "005930"))
t("수급 2015년 1년", lambda: stock.get_market_trading_value_by_date("20150101", "20151231", "005930"))
t("수급 2020년 3개월", lambda: stock.get_market_trading_value_by_date("20200101", "20200331", "005930"))
t("수급 다른종목 3개월 000660", lambda: stock.get_market_trading_value_by_date("20260801", "20261117", "000660"))
t("PER 최근 KOSPI", lambda: stock.get_market_fundamental("20261117", market="KOSPI"))
t("PER 2024 KOSPI", lambda: stock.get_market_fundamental("20240105", market="KOSPI"))
t("PER 2015 KOSPI", lambda: stock.get_market_fundamental("20150109", market="KOSPI"))
t("PER 종목 기간 005930", lambda: stock.get_market_fundamental("20150101", "20261117", "005930"))
t("순매수 기간합계 KOSPI 외국인", lambda: stock.get_market_net_purchases_of_equities("20261020", "20261117", "KOSPI", "외국인"))
t("순매수 기간합계 2015", lambda: stock.get_market_net_purchases_of_equities("20150105", "20150130", "KOSPI", "외국인"))
print("연속 20회 3개월 수급:")
ok = 0
for i, c in enumerate(["005930","000660","005380","035420","051910","006400","068270","105560","055550","012330",
                       "035720","028260","066570","003550","034730","015760","032830","009150","017670","030200"]):
    try:
        df = stock.get_market_trading_value_by_date("20260801", "20261117", c)
        ok += int(df is not None and len(df) > 0)
    except Exception as e:
        print("  fail", c, repr(e)[:80])
print("  성공", ok, "/ 20")
