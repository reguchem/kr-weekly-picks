"""Claude API로 시장 브리핑(주 1회)과 보유 종목 뉴스 점검(매일).

GitHub 저장소 Secrets에 ANTHROPIC_API_KEY가 있을 때만 동작하고, 없거나 실패하면 None을 돌려줘
정량 로직만으로 계속 진행합니다. 뉴스 판단은 점수에 ±0.15만 반영되는 보정값입니다.
"""
from __future__ import annotations

import json
import os
import re

import requests

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-5-5"
API_VERSION = os.environ.get("ANTHROPIC_VERSION") or "2023-06-01"


def _call(prompt: str, max_searches: int, max_tokens: int = 4000) -> dict | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("[llm] ANTHROPIC_API_KEY 없음 → 건너뜀")
        return None
    body = {
        "model": MODEL, "max_tokens": max_tokens,
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": max_searches,
                   "user_location": {"type": "approximate", "country": "KR", "timezone": "Asia/Seoul"}}],
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        r = requests.post(API_URL, json=body, timeout=300, headers={
            "x-api-key": key, "anthropic-version": API_VERSION, "content-type": "application/json"})
        r.raise_for_status()
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S) or re.search(r"(\{.*\})", text, re.S)
        out = json.loads(m.group(1)) if m else None
        u = data.get("usage", {})
        print(f"[llm] 완료 (입력 {u.get('input_tokens')}, 출력 {u.get('output_tokens')} 토큰)")
        return out
    except Exception as e:  # noqa
        print(f"[llm] 실패: {e!r}")
        return None


def market_brief(date: str, sectors: list[dict], holdings: list[dict]) -> dict | None:
    """sectors: [{"sector","rank","ret_21","foreign_flow"}...] 정량 업종 순위"""
    names = [s["sector"] for s in sectors]
    table = "\n".join(f"- {s['rank']}위 {s['sector']}: 1개월 중앙값 {s['ret_21']:+.1%}, "
                      f"외국인 20일 순매수 강도 {s['foreign_flow'] if s['foreign_flow'] is not None else 'n/a'}"
                      for s in sectors)
    hold = ", ".join(f"{h['name']}({h['sector']})" for h in holdings) or "없음"
    prompt = f"""당신은 한국 주식시장 리서치 애널리스트입니다. 오늘은 {date} 장마감 직후입니다.
웹 검색으로 **최근 1~2주** 정보를 확인해 아래를 분석하세요. 추측은 하지 말고, 확인한 사실만 근거로 쓰세요.

1. 외국인 투자자 매수·매도 경향 (코스피/코스닥 순매수 규모, 집중 매수·매도 업종)
2. 국내 경제 동향 (기준금리·환율·수출입 지표·정부 정책·주요 실적)
3. 세계 정세 (미국 금리·물가, 미국 증시와 반도체 사이클, 중국, 지정학 리스크, 원자재)
4. 위 내용이 아래 업종들에 주는 향후 1~4주 영향

정량 분석 결과 현재 강한 업종 순위:
{table}

현재 보유 종목: {hold}

반드시 마지막에 아래 형식의 JSON 하나만 ```json 코드블록으로 출력하세요. sector_outlook의 키는 위 업종명 중에서만 고르고,
근거가 뚜렷한 업종만 넣으세요(+1 긍정, -1 부정, 근거 약하면 생략).
```json
{{
  "summary": "시장 전반 요약 3~4문장",
  "foreign": "외국인 수급 요약 1~2문장",
  "domestic": "국내 경제 요약 1~2문장",
  "global": "세계 정세 요약 1~2문장",
  "sector_outlook": {{"업종명": 1}},
  "sector_reason": {{"업종명": "근거 한 문장"}},
  "risks": ["향후 1~4주 주요 위험 요인", "..."]
}}
```"""
    out = _call(prompt, max_searches=8, max_tokens=5000)
    if not out:
        return None
    so = {k: max(-1, min(1, int(v))) for k, v in (out.get("sector_outlook") or {}).items()
          if k in names and str(v).lstrip("+-").isdigit()}
    out["sector_outlook"] = so
    return out


def holdings_news(date: str, holdings: list[dict]) -> dict | None:
    """보유 종목별 최근 뉴스 리스크 {code: {"risk": 0|1|2, "reason": str}}"""
    if not holdings:
        return {}
    lst = "\n".join(f"- {h['name']} ({h['code']}, {h.get('sector') or '업종 미상'})" for h in holdings)
    prompt = f"""오늘은 {date}입니다. 아래 한국 상장 종목들의 **최근 3일 이내 뉴스·공시**를 웹 검색으로 확인하고,
향후 주가 하락 위험이 큰 악재(실적 쇼크, 유상증자·CB 발행, 소송·규제, 대주주 매도, 업황 급변, 테마 소멸 등)가 있는지 평가하세요.
확인된 사실만 근거로 쓰고, 뉴스가 없으면 risk 0으로 두세요.

{lst}

마지막에 아래 형식의 JSON만 ```json 코드블록으로 출력하세요. risk: 0=특이사항 없음, 1=주의, 2=심각한 악재.
```json
{{"종목코드": {{"risk": 0, "reason": "한 문장"}}}}
```"""
    out = _call(prompt, max_searches=2 + len(holdings), max_tokens=2000)
    if out is None:
        return None
    return {str(k).zfill(6): {"risk": int(v.get("risk", 0)), "reason": str(v.get("reason", ""))}
            for k, v in out.items() if isinstance(v, dict)}
