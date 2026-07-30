"""yfinance 30분봉 수집과 표준 프레임 정규화.

fetch_bars()가 반환하는 프레임은 store·grid·차트가 공유하는 표준 형태다:
BAR_COLUMNS 순서, ts_utc는 UTC bar-start(:00/:30 격자), 가격은 원시
체결가(auto_adjust=False), volume은 int64. 봉은 ts_utc + 30분 <= now(UTC)
일 때만 확정(finalized)으로 본다.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from parity.config import (
    BAR_COLUMNS,
    BAR_SECONDS,
    FETCH_PERIOD,
    INTERVAL,
    Instrument,
)

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


class FetchError(RuntimeError):
    """yfinance 조회 실패 또는 수신 데이터의 규약 위반."""


def fetch_bars(
    inst: Instrument,
    *,
    period: str = FETCH_PERIOD,
    drop_partial: bool = True,
    max_attempts: int = 4,
    now_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """티커의 30분봉을 받아 표준 프레임으로 정규화해 반환한다.

    auto_adjust=False는 협상 불가 — yfinance 기본값(True)을 쓰면 배당
    소급조정으로 이미 저장된 패리티 이력이 조용히 다시 쓰인다.
    실패(예외 또는 빈 프레임) 시 지수 백오프로 재시도하고, 모두 실패하면
    FetchError를 던진다.
    """
    # 지연 import: _normalize 단위 테스트가 yfinance의 네트워크 기계를
    # 요구하지 않도록 함수 안에서 import한다.
    import yfinance as yf

    last_error: Exception | None = None
    raw: pd.DataFrame | None = None
    for attempt in range(max_attempts):
        try:
            candidate = yf.Ticker(inst.ticker).history(
                period=period,
                interval=INTERVAL,
                prepost=False,
                auto_adjust=False,
                actions=False,
                repair=False,
            )
        except Exception as exc:  # 네트워크/파서 예외 전부 재시도 대상
            last_error = exc
            candidate = None
        if candidate is not None and not candidate.empty:
            raw = candidate
            break
        if attempt < max_attempts - 1:
            time.sleep(1.5 * 2**attempt)

    if raw is None:
        raise FetchError(
            f"{inst.ticker}: {max_attempts}회 시도에도 {INTERVAL} 봉을 받지 못했습니다"
        ) from last_error

    return _normalize(
        raw,
        inst,
        drop_partial=drop_partial,
        now_utc=now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC"),
    )


def _normalize(
    raw: pd.DataFrame,
    inst: Instrument,
    *,
    drop_partial: bool,
    now_utc: pd.Timestamp,
) -> pd.DataFrame:
    """yfinance .history() 원본을 표준 프레임으로 변환한다.

    - 인덱스(거래소 로컬 tz)를 UTC로 변환해 ts_utc 컬럼으로 만든다.
    - 격자 검증: :00/:30 정각이 아니면 DST 처리 오류·데이터 오염의
      신호이므로 절대 조용히 반올림하지 않고 FetchError를 던진다.
    - close가 NaN인 행 제거, volume은 fillna(0) 후 int64.
    - drop_partial이면 아직 안 닫힌 봉(ts_utc + 30분 > now_utc)을 버린다.
    """
    index = raw.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise FetchError(
            f"{inst.ticker}: tz-aware DatetimeIndex가 아닙니다 ({type(index).__name__})"
        )
    ts = index.tz_convert("UTC")

    off_lattice = (
        ~np.isin(np.asarray(ts.minute), (0, 30))
        | (np.asarray(ts.second) != 0)
        | (np.asarray(ts.microsecond) != 0)
    )
    if off_lattice.any():
        samples = ts[off_lattice][:3].strftime("%Y-%m-%dT%H:%M:%S%z").tolist()
        raise FetchError(
            f"{inst.ticker}: 30분 격자를 벗어난 타임스탬프 발견: {samples}"
        )

    data = raw.rename(columns=str.lower)
    missing = [col for col in _REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise FetchError(f"{inst.ticker}: 필수 컬럼 누락: {missing}")

    # 필요한 5개 컬럼만 취한다 (Dividends, Stock Splits 등 나머지는 버림).
    out = pd.DataFrame(
        {
            "ts_utc": ts,
            "open": data["open"].to_numpy(dtype="float64"),
            "high": data["high"].to_numpy(dtype="float64"),
            "low": data["low"].to_numpy(dtype="float64"),
            "close": data["close"].to_numpy(dtype="float64"),
            "volume": pd.to_numeric(data["volume"], errors="coerce").to_numpy(
                dtype="float64"
            ),
        }
    )

    # NaN close 제거 + 0/음수 가격 글리치 봉 제거 (Yahoo가 간혹 0가격 봉을
    # 내보낸다 — repair=False이므로 여기서 걸러야 ratio=inf 오염을 막는다)
    out = out[out["close"].notna() & (out["close"] > 0)]
    out["volume"] = out["volume"].fillna(0).astype("int64")

    if drop_partial:
        out = out[out["ts_utc"] + pd.Timedelta(seconds=BAR_SECONDS) <= now_utc]

    out = (
        out.sort_values("ts_utc", kind="stable")
        .drop_duplicates("ts_utc", keep="last")
        .reset_index(drop=True)
    )
    out["ts_utc"] = out["ts_utc"].astype("datetime64[ns, UTC]")
    return out[BAR_COLUMNS]
