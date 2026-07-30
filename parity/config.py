"""프로젝트 전역 상수와 종목 정의.

모든 시계열은 UTC bar-start 기준으로 정규화해 저장한다.
세션 시각(09:00, 22:30 등)은 코드 어디에도 하드코딩하지 않는다 —
세션은 봉의 존재로 관측한다(DST 자동 대응).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Instrument:
    key: str          # "kr" | "us" | "fx"
    ticker: str       # yfinance 티커
    name_ko: str
    currency: str
    exchange_tz: str  # 참고용 — 데이터는 항상 UTC로 정규화
    kind: str         # "equity" | "fx"
    filename: str     # data/ 아래 CSV 파일명


INSTRUMENTS: dict[str, Instrument] = {
    "kr": Instrument(
        key="kr",
        ticker="000660.KS",
        name_ko="SK하이닉스",
        currency="KRW",
        exchange_tz="Asia/Seoul",
        kind="equity",
        filename="kr_000660.csv",
    ),
    "us": Instrument(
        key="us",
        ticker="SKHY",
        name_ko="SK하이닉스 ADR",
        currency="USD",
        exchange_tz="America/New_York",
        kind="equity",
        filename="us_skhy.csv",
    ),
    "fx": Instrument(
        key="fx",
        ticker="KRW=X",
        name_ko="원/달러 환율",
        currency="KRW per USD",
        exchange_tz="Europe/London",  # Yahoo가 FX에 쓰는 타임존 (UTC 정규화로 무관해짐)
        kind="fx",
        filename="fx_usdkrw.csv",
    ),
}

# ADR 10주 = 본주 1주 (2026-07-10 나스닥 상장 SKHY, 확인된 비율)
ADR_RATIO = 10

INTERVAL = "30m"
BAR_SECONDS = 1800
FETCH_PERIOD = "60d"    # yfinance 30분봉 조회 한도
TOPUP_PERIOD = "5d"     # 앱 런타임 top-up 조회 범위

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# 저장 스키마 (모든 CSV 동일): ts_utc는 bar-start, ISO-8601 Z 표기
BAR_COLUMNS = ["ts_utc", "open", "high", "low", "close", "volume"]
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
FLOAT_FORMAT = "%.4f"

# FX는 주중 거의 연속 거래 — 이보다 오래 멈추면 피드 이상으로 간주
FX_STALE_MINUTES = 120

# 한국식 등락/식별 색 (부모 프로젝트 컨벤션)
COLOR_KR = "#ef4035"
COLOR_US = "#3478e5"
COLOR_NEUTRAL = "#6b7280"
