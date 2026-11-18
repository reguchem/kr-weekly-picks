"""과거 화요일에 같은 전략이 무엇을 추천했을지 재현하고, 그대로 샀다면 지금 수익률이 얼마인지 계산.

실행: python -m src.replay 2026-11-03 2026-11-10   → docs/data/replay.json
- 시장 필터를 적용한 결과(실제 규칙)와, 필터를 무시했을 때 골랐을 종목을 모두 보여줌
- 유니버스·ETF 목록·업종 분류는 현재 기준(약간의 사후 정보 포함)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

from . import config as C
from . import weekly
from .data import get_ohlcv
from .etf import ETF_BUY, ETF_SELL

tmp = Path(tempfile.mkdtemp())
weekly.PERF_PATH = tmp / "perf.json"       # 실제 운용 기록을 건드리지 않음
weekly.BRIEF_PATH = tmp / "brief.json"


def evaluate(rec, latest):
    rows, invest, now_val = [], 0.0, 0.0
    for p in rec["picks"]:
        c = p["code"]
        sh, entry = p.get("shares") or 0, p.get("close")
        last = latest.get(c)
        if not sh or not entry or last is None:
            continue
        etf = p.get("type") == "etf"
        buy_amt = sh * entry * (1 + (ETF_BUY if etf else C.BUY_COST))
        sell_amt = sh * last * (1 - (ETF_SELL if etf else C.SELL_COST))
        invest += buy_amt
        now_val += sell_amt
        rows.append({"type": p.get("type"), "code": c, "name": p["name"], "shares": sh, "entry": entry,
                     "last": last, "ret": last / entry - 1, "pnl": sell_amt - buy_amt})
    cash = C.SEED - invest
    total = cash + now_val
    return rows, total


def run(dates):
    out = {"seed": C.SEED, "cases": []}
    k = get_ohlcv("KOSPI", 40)
    for ds in dates:
        d = pd.Timestamp(ds)
        recs = {"weeks": []}
        M = weekly.build_market(recs, d, refresh_sectors=True)
        t = M["t"]
        case = {"date": str(t.date()), "market_ok": M["market_ok"], "kospi": M["kospi_now"], "kospi_ma": M["kospi_ma"]}
        variants = [("rule", M["market_ok"])]
        if not M["market_ok"]:
            variants.append(("ignore_filter", True))
        for name, ok in variants:
            M2 = dict(M, market_ok=ok)
            rec = weekly.make_recommendation({"weeks": []}, M2)
            codes = [p["code"] for p in rec["picks"]]
            latest = {}
            for c in codes:
                o = get_ohlcv(c, 10)
                if len(o):
                    latest[c] = float(o["close"].iloc[-1])
                    case["last_date"] = str(o.index[-1].date())
            rows, total = evaluate(rec, latest)
            sc_n = len(M2["f"])
            print(f"[{name}] 업종 분류 {len(M['sectors'])}종목, 점수 대상 {sc_n}, 상위 업종 "
                  f"{list(M['sector_table'].index[:5]) if len(M['sector_table']) else []}, "
                  f"ETF 후보 {len(M['E'].scores(t))}")
            case[name] = {"picks": rows, "total": total, "ret": total / C.SEED - 1}
        kc = k["close"]
        k0 = float(kc.loc[:t].iloc[-1])
        case["kospi_ret"] = float(kc.iloc[-1]) / k0 - 1
        case["kospi_last_date"] = str(kc.index[-1].date())
        out["cases"].append(case)
        print(f"\n=== {case['date']} (화) 추천 → {case.get('last_date', '')} 기준 ===")
        print(f"시장 필터: KOSPI {case['kospi']:,.0f} / {C.MARKET_MA}일선 {case['kospi_ma'] or 0:,.0f} → "
              f"{'투자' if case['market_ok'] else '현금 보유'}")
        for name in ("rule", "ignore_filter"):
            if name not in case:
                continue
            v = case[name]
            print(f"[{'규칙대로' if name == 'rule' else '필터 무시했다면'}] 100만원 → {v['total']:,.0f}원 ({v['ret']:+.2%})")
            for r in v["picks"]:
                print(f"   {r['type']:5s} {r['name']}({r['code']}) {r['shares']}주 {r['entry']:,.0f}→{r['last']:,.0f}원 "
                      f"{r['ret']:+.2%} 손익 {r['pnl']:+,.0f}원")
        print(f"KOSPI 같은 기간 {case['kospi_ret']:+.2%}")
    C.DOCS_DATA.mkdir(parents=True, exist_ok=True)
    (C.DOCS_DATA / "replay.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    run(sys.argv[1:] or [str((pd.Timestamp.today() - pd.Timedelta(days=15)).date())])
