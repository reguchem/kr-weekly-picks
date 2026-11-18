"""KRX 조회 진단 2: 연속 요청 한도와 회복 조건 확인"""
import time
import pandas as pd
from pykrx import stock
from pykrx.website.comm import auth

codes = list(stock.get_market_ticker_list("20261117", market="KOSPI"))[:120]
def q(c, a="20240101", b="20251231"):
    try:
        df = stock.get_market_trading_value_by_date(a, b, c)
        return df is not None and len(df) > 0
    except Exception:
        return False

first_fail, t0 = None, time.time()
for i, c in enumerate(codes[:80]):
    ok = q(c)
    if not ok and first_fail is None:
        first_fail = i
        print(f"첫 실패: {i}번째 요청 ({time.time()-t0:.0f}s 경과)")
    if first_fail is not None and i - first_fail >= 3:
        break
print("80회 내 실패 없음" if first_fail is None else "")
if first_fail is not None:
    for w in (30, 90, 180):
        time.sleep(w)
        print(f"{w}s 대기 후:", q(codes[0]))
    auth._auth_session = None  # 새 세션으로 재로그인
    print("재로그인 후:", q(codes[1]))
    print("짧은 기간(1년) 요청:", q(codes[2], "20250101", "20251231"))
