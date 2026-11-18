"""KRX 조회 진단 3: 실제 수집 함수 그대로 재현"""
import time
import pandas as pd
from pykrx import stock
from src import data

uni = data.get_universe(300)
codes = uni["code"].tolist()
print("유니버스 앞 10:", codes[:10])
s0, e0 = pd.Timestamp("2014-09-01"), pd.Timestamp("2026-11-17")
for c in codes[:3]:
    cur = s0
    while cur <= e0:
        s1 = min(cur + pd.DateOffset(years=2) - pd.Timedelta(days=1), e0)
        try:
            df = stock.get_market_trading_value_by_date(cur.strftime("%Y%m%d"), s1.strftime("%Y%m%d"), c)
            print(c, cur.date(), s1.date(), "행", 0 if df is None else len(df))
        except Exception as e:
            print(c, cur.date(), s1.date(), "예외", repr(e)[:100])
        cur = s1 + pd.Timedelta(days=1)
print("--- get_flow_history 메인 스레드")
for c in codes[3:6]:
    d = data.get_flow_history(c, "2014-09-01", "2026-11-17")
    print(c, None if d is None else len(d), data._FLOW_ERR)
print("--- get_flow_panel(스레드 1개)")
p = data.get_flow_panel(codes[6:16], "2014-09-01", "2026-11-17")
print(None if p is None else p["foreign"].shape)
