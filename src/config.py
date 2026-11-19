"""전역 설정. 숫자를 바꾸면 백테스트·주간 추천 모두에 반영됩니다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DATA = ROOT / "docs" / "data"      # GitHub Pages가 읽는 JSON
CACHE = ROOT / "data" / "cache"         # 백테스트용 캐시 (선택)

# ── 유니버스 ─────────────────────────────────────────────
import os as _os
UNIVERSE_MARKETS = tuple(_os.environ.get("UNIVERSE_MARKETS", "KOSPI").split(","))   # 후보 시장
UNIVERSE_SIZE = int(_os.environ.get("UNIVERSE_SIZE", "500"))                       # 시가총액 상위 N 종목
MIN_TRADING_VALUE = 1e9      # 20일 평균 거래대금 하한 (원) — 10억
MIN_HISTORY_DAYS = 260       # 최소 상장 기간(거래일)

# ── 추천 제외 종목 (합병·상장폐지 예정 등) ──────────────
EXCLUDE = {
    "020560": "아시아나항공 — 2026-12-17 대한항공에 흡수합병 예정(합병 전 거래정지)",
}

# ── 포트폴리오 규칙 ─────────────────────────────────────
SEED = 1_000_000             # 시드머니(원). 수익률·수량 계산을 실제 금액·정수 주식으로 함
HOLD_WEEKS_EST = 8           # 교체 판단 시 가정하는 평균 보유기간(추천 횟수, 주 2회 × 4주): 예상수익 차이 × 8 > 왕복비용이면 교체
MAX_PRICE_FRACTION = 0.5     # 주가 ≤ (평가금액/종목수) × 0.5 인 종목만 신규 후보 (1주 단위 자투리 현금 최소화)
N_PICKS = 3                  # 개별 종목 수
N_ETFS = 2                   # 테마 ETF 수
SLOTS = N_PICKS + N_ETFS     # 자금은 5등분 (종목 3 + ETF 2)
MARKET_MA = 120              # KOSPI가 이 이동평균선 아래면 '하락장 모드' (0이면 필터 끔)
# 하락장 모드: KOSPI 대비 강한 것만, 자리 일부만 채우고, 손절을 짧게
BEAR_STOCKS = 2              # 하락장 최대 개별 종목 수
BEAR_ETFS = 1                # 하락장 최대 ETF 수
BEAR_STOP = 0.07             # 하락장에서 편입한 종목은 편입가 대비 -7%면 손절
BEAR_RS_STOCK = 0.10         # 3개월 수익률이 KOSPI보다 10%p 이상 높아야 함
BEAR_RS_ETF = 0.05           # ETF는 5%p 이상
BUFFER_RANK = 15             # 보유 종목은 순위가 이 안에 있으면 계속 보유 (교체 최소화)

# ── 거래 비용 (편도, 비율) ───────────────────────────────
FEE = 0.00015                # 증권사 수수료 0.015% (한국투자증권 실제 수수료에 맞게 수정)
SELL_TAX = 0.0020            # 증권거래세(농특세 포함) — 세율 변경 시 수정
SLIPPAGE = 0.0010            # 시가 매매 슬리피지 추정 0.1%
BUY_COST = FEE + SLIPPAGE
SELL_COST = FEE + SELL_TAX + SLIPPAGE

# ── 백테스트 ────────────────────────────────────────────
BACKTEST_START = "2015-01-01"
IN_SAMPLE_END = "2021-12-31"  # 전략 선택은 이 날짜까지의 성과로만, 이후는 검증 구간
DEFAULT_STRATEGY = "theme_flow_slow"   # 테마·수급(느린 교체) 전략
STRATEGY_MODE = "fixed"           # "fixed": 위 전략 고정 / "auto": 백테스트 인샘플 1위 자동 선택
TRADE_WEEKDAY = 0             # (호환용) 첫 신호 요일
# 추천 일정: ① 화요일 08:00 (월요일 종가 기준) ② 목요일 17:00 (목요일 종가 기준)
# 매매는 각각 다음 거래일 시가(화 09:00, 금 09:00)
WEEKDAYS_KR = ["월", "화", "수", "목", "금"]

# ── 체결 방식 ───────────────────────────────────────────
# 추천(전날/당일 종가 기준) → 다음 거래일 시가에 체결
EXECUTE_AT = "next_open"

# ── 수익률 표시 구간 (거래일) ───────────────────────────
HORIZONS = {"1w": 5, "1m": 21, "3m": 63}
