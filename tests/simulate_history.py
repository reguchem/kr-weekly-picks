"""가상 데이터로 과거 N주 동안 매주 화요일 실행을 재현 (테스트용, 네트워크·API 불필요).
SYNTHETIC=1 python -m tests.simulate_history 20
"""
import os, sys, json
import pandas as pd
from src import weekly, config as C
from src.strategy import signal_dates

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
C.DOCS_DATA.mkdir(parents=True, exist_ok=True)
recs = {"weeks": []}
for p in (weekly.PERF_PATH, weekly.REC_PATH, weekly.WARN_PATH):
    p.unlink(missing_ok=True)
days = pd.bdate_range(end=pd.Timestamp.today(), periods=n * 5 + 5)
sigs = signal_dates(days)[-n:]
for d in sigs:
    os.environ["SYN_END"] = str(d.date())
    M = weekly.build_market(recs, d, refresh_sectors=True)
    M["slot"] = "tue" if d.weekday() == 0 else "thu"
    M["trade_date"] = str(pd.bdate_range(d + pd.Timedelta(days=1), periods=1)[0].date())
    rec = weekly.make_recommendation(recs, M)
    if rec:
        recs["weeks"].append(rec)
    weekly.update_performance(recs)   # 실전처럼 매주 성과(보유 수량) 갱신
os.environ.pop("SYN_END")
M = weekly.build_market(recs, days[-1], refresh_sectors=False)
warn = weekly.check_warnings(recs, M)
warn["history"] = []
weekly.WARN_PATH.write_text(json.dumps(warn, ensure_ascii=False, indent=1))
for i in warn["items"]:
    print("[점검]", i["name"], i["level"], i["signals"])
weekly.update_performance(recs)
recs["updated_at"] = "test"
weekly.REC_PATH.write_text(json.dumps(recs, ensure_ascii=False, indent=1))
