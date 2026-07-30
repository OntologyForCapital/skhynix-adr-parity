"""parity.fetch 단위 테스트 — 네트워크 없음, 합성 프레임만 사용."""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from parity import fetch as fetch_mod
from parity.config import BAR_COLUMNS, FETCH_PERIOD, INSTRUMENTS, INTERVAL
from parity.fetch import FetchError, _normalize, fetch_bars

KR = INSTRUMENTS["kr"]
US = INSTRUMENTS["us"]
FX = INSTRUMENTS["fx"]


def _raw(
    index: pd.DatetimeIndex,
    closes: list[float | None],
    volumes: list[float] | None = None,
    extra_columns: bool = False,
) -> pd.DataFrame:
    """yfinance .history() 모양의 원본 프레임을 만든다 (None → NaN close)."""
    arr = np.asarray(closes, dtype="float64")  # None은 NaN으로 변환됨
    frame = pd.DataFrame(
        {
            "Open": arr - 1.0,
            "High": arr + 1.0,
            "Low": arr - 2.0,
            "Close": arr,
            "Volume": volumes if volumes is not None else [100.0] * len(arr),
        },
        index=index,
    )
    if extra_columns:
        frame["Dividends"] = 0.0
        frame["Stock Splits"] = 0.0
    return frame


def _install_fake_yfinance(monkeypatch: pytest.MonkeyPatch, frames: list[pd.DataFrame]):
    """sys.modules에 가짜 yfinance를 주입한다 (실제 import 기계 미사용).

    frames[i]는 i번째 history() 호출의 반환값 (마지막 원소가 반복됨).
    반환값: (ticker, kwargs) 기록 리스트.
    """
    calls: list[tuple[str, dict]] = []

    class _FakeTicker:
        def __init__(self, ticker: str) -> None:
            self._ticker = ticker

        def history(self, **kwargs) -> pd.DataFrame:
            calls.append((self._ticker, dict(kwargs)))
            return frames[min(len(calls) - 1, len(frames) - 1)]

    module = types.ModuleType("yfinance")
    module.Ticker = _FakeTicker
    monkeypatch.setitem(sys.modules, "yfinance", module)
    return calls


# F1: 런던(BST 포함) 인덱스가 올바른 UTC 격자로 정규화된다
def test_f1_london_bst_normalizes_to_utc_lattice() -> None:
    idx = pd.DatetimeIndex(
        ["2026-01-14 08:00", "2026-07-15 07:00", "2026-07-15 07:30"]
    ).tz_localize("Europe/London")  # 1월=GMT(UTC+0), 7월=BST(UTC+1)
    raw = _raw(idx, [1400.0, 1401.0, 1402.0])

    out = _normalize(
        raw, FX, drop_partial=True, now_utc=pd.Timestamp("2026-07-16 00:00", tz="UTC")
    )

    assert list(out.columns) == BAR_COLUMNS
    expected = [
        pd.Timestamp("2026-01-14 08:00", tz="UTC"),
        pd.Timestamp("2026-07-15 06:00", tz="UTC"),
        pd.Timestamp("2026-07-15 06:30", tz="UTC"),
    ]
    assert out["ts_utc"].tolist() == expected
    assert str(out["ts_utc"].dtype) == "datetime64[ns, UTC]"
    assert out["close"].dtype == "float64"
    assert out["volume"].dtype == "int64"


# F2: 격자를 벗어난 타임스탬프는 FetchError (조용한 반올림 금지)
def test_f2_off_lattice_timestamp_raises() -> None:
    idx = pd.DatetimeIndex(["2026-07-15 06:15"]).tz_localize("UTC")
    raw = _raw(idx, [10.0])
    with pytest.raises(FetchError):
        _normalize(
            raw,
            KR,
            drop_partial=False,
            now_utc=pd.Timestamp("2026-07-16 00:00", tz="UTC"),
        )


# F3: 미확정 봉은 drop_partial=True에서만 제거된다
def test_f3_partial_bar_dropped_only_when_requested() -> None:
    idx = pd.DatetimeIndex(["2026-07-15 05:30", "2026-07-15 06:00"]).tz_localize("UTC")
    raw = _raw(idx, [10.0, 11.0])
    now = pd.Timestamp("2026-07-15 06:10", tz="UTC")

    kept = _normalize(raw, KR, drop_partial=True, now_utc=now)
    assert kept["ts_utc"].tolist() == [pd.Timestamp("2026-07-15 05:30", tz="UTC")]

    both = _normalize(raw, KR, drop_partial=False, now_utc=now)
    assert both["ts_utc"].tolist() == [
        pd.Timestamp("2026-07-15 05:30", tz="UTC"),
        pd.Timestamp("2026-07-15 06:00", tz="UTC"),
    ]


# F4: fetch_bars가 안전한 kwargs를 정확히 전달한다 (auto_adjust=False 필수)
def test_f4_fetch_bars_passes_safe_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.DatetimeIndex(["2026-07-15 09:00", "2026-07-15 09:30"]).tz_localize(
        "Asia/Seoul"
    )  # KST 09:00/09:30 = UTC 00:00/00:30
    calls = _install_fake_yfinance(monkeypatch, [_raw(idx, [250000.0, 251000.0])])

    out = fetch_bars(KR, now_utc=pd.Timestamp("2026-07-15 02:00", tz="UTC"))

    ticker, kwargs = calls[0]
    assert ticker == KR.ticker
    assert kwargs["auto_adjust"] is False
    assert kwargs["prepost"] is False
    assert kwargs["actions"] is False
    assert kwargs["repair"] is False
    assert kwargs["interval"] == INTERVAL == "30m"
    assert kwargs["period"] == FETCH_PERIOD
    assert len(out) == 2


# F5: 매번 빈 프레임이면 재시도 후 FetchError (sleep은 monkeypatch로 대체)
def test_f5_empty_frames_raise_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_yfinance(monkeypatch, [pd.DataFrame()])
    sleeps: list[float] = []
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(FetchError) as err:
        fetch_bars(US, max_attempts=3)

    assert US.ticker in str(err.value)  # 메시지에 티커 포함
    assert len(calls) == 3
    assert sleeps == [1.5, 3.0]  # 1.5 * 2**attempt, 마지막 시도 후에는 안 잔다


# F6: 여분 컬럼 제거, NaN close 행 제거, NaN volume → 0
def test_f6_extras_dropped_nan_close_dropped_nan_volume_zero() -> None:
    idx = pd.DatetimeIndex(
        ["2026-07-15 00:00", "2026-07-15 00:30", "2026-07-15 01:00"]
    ).tz_localize("UTC")
    raw = _raw(
        idx,
        [10.0, None, 12.0],
        volumes=[100.0, 50.0, float("nan")],
        extra_columns=True,
    )

    out = _normalize(
        raw, US, drop_partial=True, now_utc=pd.Timestamp("2026-07-15 02:00", tz="UTC")
    )

    assert list(out.columns) == BAR_COLUMNS  # Dividends/Stock Splits 제거됨
    assert len(out) == 2  # NaN close 행(00:30) 제거됨
    assert out["ts_utc"].tolist() == [
        pd.Timestamp("2026-07-15 00:00", tz="UTC"),
        pd.Timestamp("2026-07-15 01:00", tz="UTC"),
    ]
    assert out["volume"].dtype == "int64"
    assert out["volume"].tolist() == [100, 0]


# F7: 0/음수 가격 글리치 봉은 제거된다 (ratio=inf 오염 방지)
def test_f7_non_positive_close_dropped() -> None:
    idx = pd.DatetimeIndex(
        [
            "2026-07-15 00:00",
            "2026-07-15 00:30",
            "2026-07-15 01:00",
            "2026-07-15 01:30",
        ]
    ).tz_localize("UTC")
    raw = _raw(idx, [10.2, 0.0, -1.0, 10.4])

    out = _normalize(
        raw, US, drop_partial=False, now_utc=pd.Timestamp("2026-07-16", tz="UTC")
    )

    assert len(out) == 2
    assert (out["close"] > 0).all()
