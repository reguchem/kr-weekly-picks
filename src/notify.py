"""텔레그램 알림 (선택). GitHub 저장소 Secrets에 TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID를 넣으면 동작."""
import os

import requests

PAGE_URL = os.environ.get("PAGE_URL", "")


def send_telegram(rec: dict):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    buys = [p for p in rec["picks"] if p["action"] == "BUY"]
    holds = [p for p in rec["picks"] if p["action"] == "HOLD"]
    lines = [f"[추천] {rec['signal_date']} 종가 기준 → {rec.get('trade_date') or '다음 거래일'} 09:00 시가 매매 (종목 3 + 테마 ETF 2)"]
    if rec.get("market_ok") is False:
        lines.append(f"이번 주: 현금 보유 — KOSPI가 {rec.get('market_ma')}일 평균선 아래")
    if rec["sells"]:
        lines.append("편출: " + ", ".join(f"{s['name']}({s['code']})" for s in rec["sells"]))
    if buys:
        lines.append("신규 편입: " + ", ".join(
            f"{p['name']}({p['code']}, {'ETF' if p.get('type') == 'etf' else (p.get('sector') or '')}) "
            f"종가 {p['close']:,.0f}원" + (f" · {p['shares']}주" if p.get('shares') else "") for p in buys))
    if holds:
        lines.append("유지: " + ", ".join(p["name"] for p in holds))
    if not rec["sells"] and not buys:
        lines.append("이번 주는 변경 없음 — 기존 종목 유지")
    else:
        lines.append("참고: 표시 가격은 기준일 종가, 수량은 그 가격 기준 추정 — 실제 매수는 다음 거래일 시가")
    if PAGE_URL:
        lines.append(PAGE_URL)
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": "\n".join(lines)}, timeout=10)
        print("[알림] 텔레그램 전송")
    except Exception as e:  # noqa
        print(f"[알림] 실패: {e!r}")


def send_warnings(date: str, items: list[dict]):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat or not items:
        return
    lines = [f"[보유 종목 점검] {date}"]
    for i in items:
        lines.append(f"{i['level']}: {i['name']} (매수 후 {i['ret']:+.1%})")
        lines += [f"  - {s}" for s in i["signals"]]
    lines.append("참고 정보입니다. 매도 여부는 직접 판단하세요. (정기 교체: 화 08:00·목 16:00 추천)")
    if PAGE_URL:
        lines.append(PAGE_URL)
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": "\n".join(lines)}, timeout=10)
    except Exception as e:  # noqa
        print(f"[알림] 실패: {e!r}")
