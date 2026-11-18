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
    lines = [f"[주간 3종목] {rec['signal_date']} ({rec['strategy_label']})"]
    if rec["sells"]:
        lines.append("편출: " + ", ".join(f"{s['name']}({s['code']})" for s in rec["sells"]))
    if buys:
        lines.append("신규 편입: " + ", ".join(
            f"{p['name']}({p['code']}, {p.get('sector') or ''}) 종가 {p['close']:,.0f}원" for p in buys))
    if holds:
        lines.append("유지: " + ", ".join(p["name"] for p in holds))
    if not rec["sells"] and not buys:
        lines.append("이번 주는 변경 없음 — 기존 종목 유지")
    else:
        lines.append("참고: 추천가는 오늘 종가 — 같은 가격 매매는 16:00까지 시간외 종가")
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
    lines.append("참고 정보입니다. 매도 여부는 직접 판단하세요. (전략상 정기 교체는 화요일)")
    if PAGE_URL:
        lines.append(PAGE_URL)
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": "\n".join(lines)}, timeout=10)
    except Exception as e:  # noqa
        print(f"[알림] 실패: {e!r}")
