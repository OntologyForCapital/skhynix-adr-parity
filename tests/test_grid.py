"""build_ratio_grid 단위 테스트 — 합성 프레임만 사용 (네트워크 호출 없음)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from parity.config import BAR_COLUMNS, FX_STALE_MINUTES
from parity.grid import build_ratio_grid

# 출력 계약 (grid.GRID_COLUMNS를 재사용하지 않고 독립적으로 고정)
EXPECTED_COLUMNS = [
    "ts_utc",
    "kr_close",
    "us_close",
    "fx_close",
    "kr_ts_utc",
    "us_ts_utc",
    "fx_ts_utc",
    "kr_live",
    "us_live",
    "fx_live",
    "kr_age_min",
    "us_age_min",
    "fx_age_min",
    "session",
    "kr_open_slot",
    "us_open_slot",
    "fx_stale",
]


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def make_bars(ts_list, closes=None) -> pd.DataFrame:
    """canonical bar frame 생성 헬퍼 (open/high/low = close, volume 0)."""
    idx = pd.to_datetime(list(ts_list), utc=True)
    n = len(idx)
    if closes is None:
        closes = [100.0] * n
    close = np.asarray(closes, dtype="float64")
    frame = pd.DataFrame(
        {
            "ts_utc": pd.Series(idx, dtype="datetime64[ns, UTC]"),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.zeros(n, dtype="int64"),
        }
    )
    return frame[BAR_COLUMNS]


def test_g1_asof_join_ages_and_dtypes():
    kr = make_bars(["2026-07-30 06:00"], [100.0])
    us = make_bars(["2026-07-30 13:30"], [9.0])
    fx = make_bars(["2026-07-30 06:00", "2026-07-30 13:30"], [1400.0, 1410.0])
    grid = build_ratio_grid(kr, us, fx)

    assert list(grid.columns) == EXPECTED_COLUMNS
    assert isinstance(grid.index, pd.RangeIndex)
    assert len(grid) == 2
    assert list(grid["ts_utc"]) == [ts("2026-07-30 06:00"), ts("2026-07-30 13:30")]

    # dtype 계약
    assert str(grid["ts_utc"].dtype) == "datetime64[ns, UTC]"
    assert str(grid["kr_ts_utc"].dtype) == "datetime64[ns, UTC]"
    assert str(grid["us_ts_utc"].dtype) == "datetime64[ns, UTC]"
    assert grid["kr_close"].dtype == np.float64
    assert grid["kr_age_min"].dtype == np.float64
    assert grid["kr_live"].dtype == np.bool_
    assert grid["kr_open_slot"].dtype == np.bool_
    assert grid["fx_stale"].dtype == np.bool_

    # 06:00 슬롯 — KR 세션, US 봉은 아직 미래라 결측
    r0 = grid.iloc[0]
    assert bool(r0["kr_live"]) is True
    assert r0["kr_close"] == 100.0
    assert r0["kr_age_min"] == 0.0
    assert pd.isna(r0["us_close"])
    assert pd.isna(r0["us_ts_utc"])
    assert bool(r0["us_live"]) is False
    assert pd.isna(r0["us_age_min"])
    assert r0["fx_close"] == 1400.0
    assert bool(r0["fx_live"]) is True
    assert r0["session"] == "KR"
    assert bool(r0["kr_open_slot"]) is True
    assert bool(r0["fx_stale"]) is False

    # 13:30 슬롯 — US 세션, KR 종가는 450분 전 값을 as-of로 사용
    r1 = grid.iloc[1]
    assert bool(r1["us_live"]) is True
    assert bool(r1["kr_live"]) is False
    assert r1["kr_age_min"] == 450.0
    assert r1["kr_close"] == 100.0
    assert r1["kr_ts_utc"] == ts("2026-07-30 06:00")
    assert r1["us_close"] == 9.0
    assert r1["fx_close"] == 1410.0
    assert r1["session"] == "US"
    assert bool(r1["us_open_slot"]) is True
    assert bool(r1["kr_open_slot"]) is False


def test_g2_equity_union_excludes_fx_only_slots():
    # 토요일(2026-08-01)에는 FX 봉만 존재 → 그 슬롯은 격자에 없어야 한다
    kr = make_bars(["2026-07-31 06:00"], [100.0])
    us = make_bars([])
    fx = make_bars(
        ["2026-07-31 06:00", "2026-08-01 10:00", "2026-08-01 10:30"],
        [1400.0, 1401.0, 1402.0],
    )
    grid = build_ratio_grid(kr, us, fx)
    assert list(grid["ts_utc"]) == [ts("2026-07-31 06:00")]

    # us 프레임이 비어도 (상장 전 백테스트) 예외 없이 전부 결측/False
    r0 = grid.iloc[0]
    assert pd.isna(r0["us_close"])
    assert pd.isna(r0["us_ts_utc"])
    assert bool(r0["us_live"]) is False
    assert pd.isna(r0["us_age_min"])
    assert bool(r0["us_open_slot"]) is False

    # 주식 둘 다 비면 0행 격자 (컬럼은 유지, 예외 없음)
    empty = build_ratio_grid(make_bars([]), make_bars([]), fx)
    assert len(empty) == 0
    assert list(empty.columns) == EXPECTED_COLUMNS


def test_g3_open_slot_flags():
    # KR 00:00~06:00 UTC 13개 연속 슬롯, 이후 US 13:30부터
    kr_ts = pd.date_range("2026-07-30 00:00", periods=13, freq="30min", tz="UTC")
    us_ts = pd.date_range("2026-07-30 13:30", periods=3, freq="30min", tz="UTC")
    kr = make_bars(kr_ts, [100.0] * 13)
    us = make_bars(us_ts, [9.0] * 3)
    fx = make_bars([])
    grid = build_ratio_grid(kr, us, fx)

    assert len(grid) == 16
    assert grid.loc[grid["kr_open_slot"], "ts_utc"].tolist() == [ts("2026-07-30 00:00")]
    assert grid.loc[grid["us_open_slot"], "ts_utc"].tolist() == [ts("2026-07-30 13:30")]

    # FX 데이터가 아예 없으면 모든 행이 fx_stale
    assert grid["fx_age_min"].isna().all()
    assert grid["fx_stale"].all()


def test_g4_us_dst_transition():
    # 2026-11-01 미국 DST 종료: 세션 시작이 13:30 UTC(EDT) → 14:30 UTC(EST)로 이동.
    # 세션 시각을 하드코딩하지 않아도 봉의 존재만으로 open_slot이 잡혀야 한다.
    us = make_bars(
        [
            "2026-10-30 13:30",
            "2026-10-30 14:00",
            "2026-11-02 14:30",
            "2026-11-02 15:00",
        ],
        [9.0, 9.1, 9.2, 9.3],
    )
    kr = make_bars(["2026-11-02 00:00", "2026-11-02 00:30"], [100.0, 101.0])
    fx = make_bars([])
    grid = build_ratio_grid(kr, us, fx)

    open_slots = grid.loc[grid["us_open_slot"], "ts_utc"].tolist()
    assert open_slots == [ts("2026-10-30 13:30"), ts("2026-11-02 14:30")]

    # 서울 벽시계로는 22:30 (EDT 개장) → 23:30 (EST 개장)
    seoul = [t.tz_convert("Asia/Seoul") for t in open_slots]
    assert (seoul[0].hour, seoul[0].minute) == (22, 30)
    assert (seoul[1].hour, seoul[1].minute) == (23, 30)

    # DST 종료 후에는 13:30 UTC 슬롯이 존재하지 않는다
    assert ts("2026-11-02 13:30") not in set(grid["ts_utc"])


def test_g5_holiday_staleness_and_fx_stale():
    # KR가 하루 종일 휴장 → 다음 날 US 행의 kr_age_min > 1440, 행은 유지
    kr = make_bars(["2026-07-27 02:00"], [100.0])
    us = make_bars(["2026-07-28 13:30"], [9.0])
    fx = make_bars(["2026-07-27 02:00", "2026-07-28 11:00"], [1400.0, 1405.0])
    grid = build_ratio_grid(kr, us, fx)

    assert len(grid) == 2  # staleness로 행을 버리지 않는다

    r1 = grid.iloc[1]
    assert r1["kr_age_min"] == 2130.0  # 35.5시간
    assert r1["kr_age_min"] > 1440
    assert r1["kr_close"] == 100.0  # stale이어도 유효한 프리미엄 재료

    # FX 공백 150분 → FX_STALE_MINUTES(120) 초과 → stale
    assert r1["fx_age_min"] == 150.0
    assert r1["fx_age_min"] > FX_STALE_MINUTES
    assert bool(r1["fx_stale"]) is True

    # 신선한 FX (age 0) → stale 아님
    r0 = grid.iloc[0]
    assert r0["fx_age_min"] == 0.0
    assert bool(r0["fx_stale"]) is False
