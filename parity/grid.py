"""KR/US 주식 봉을 괴리율 계산용 격자로 결합.

격자 슬롯은 KR·US 주식 봉 시작 시각(UTC)의 합집합으로만 만든다.
FX 전용 슬롯은 의도적으로 제외한다 — 양쪽 주가가 모두 멈춘 채 환율만
움직이는 구간은 프리미엄 정보가 아니라 노이즈이고, 주말 슬롯도 이
규칙으로 자동으로 사라진다.

각 leg는 merge_asof(direction="backward")로 "슬롯 이전(포함)의 마지막
봉"을 붙인다. 오래된 값이라도 행을 버리지 않는다 — KR 휴장일의 US
세션 행도 유효한 프리미엄 읽기다. 정직함은 행 삭제가 아니라
*_live / *_age_min / fx_stale 플래그로 표현한다.

세션 시각 하드코딩 없음(세션은 봉의 존재로 관측 — DST 자동 대응),
격자 로직 내부에 타임존 변환 없음 (UTC in, UTC out).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from parity.config import FX_STALE_MINUTES

_TS_DTYPE = "datetime64[ns, UTC]"

# build_ratio_grid 출력 컬럼 (순서 고정)
GRID_COLUMNS = [
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


def _attach_leg(grid: pd.DataFrame, bars: pd.DataFrame, key: str) -> pd.DataFrame:
    """슬롯마다 해당 leg의 '슬롯 이전(포함) 마지막 봉'을 붙인다.

    추가되는 컬럼:
      {key}_close   : 사용한 봉의 종가 (없으면 NaN)
      {key}_ts_utc  : 실제 사용한 봉의 시작 시각 (없으면 NaT)
      {key}_live    : 슬롯 시각에 정확히 시작하는 봉이 있는지
      {key}_age_min : slot - 사용한 봉 시작, 분 단위 (live면 0.0, 봉 없으면 NaN)
    """
    close_col = f"{key}_close"
    ts_col = f"{key}_ts_utc"

    if bars.empty:
        out = grid.copy()
        out[close_col] = np.full(len(out), np.nan, dtype="float64")
        out[ts_col] = pd.Series(pd.NaT, index=out.index, dtype=_TS_DTYPE)
    else:
        right = bars[["ts_utc", "close"]].rename(columns={"close": close_col})
        right = right.assign(**{ts_col: right["ts_utc"]})
        right[close_col] = right[close_col].astype("float64")
        out = pd.merge_asof(
            grid,
            right[["ts_utc", ts_col, close_col]],
            on="ts_utc",
            direction="backward",
            allow_exact_matches=True,
        )

    out[f"{key}_live"] = (out[ts_col] == out["ts_utc"]).astype(bool)
    out[f"{key}_age_min"] = (out["ts_utc"] - out[ts_col]).dt.total_seconds() / 60.0
    return out


def build_ratio_grid(kr: pd.DataFrame, us: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """세 canonical bar frame을 괴리율 격자로 결합한다.

    Args:
        kr, us, fx: BAR_COLUMNS 스키마의 봉 프레임 (비어 있어도 됨).
            ts_utc 오름차순·중복 없음 가정 (canonical 계약).

    Returns:
        GRID_COLUMNS 순서의 DataFrame, ts_utc 오름차순, RangeIndex.
        주식 둘 다 비면 0행 프레임 (컬럼은 유지, 예외 없음).
    """
    equity_ts = [df["ts_utc"] for df in (kr, us) if not df.empty]
    if equity_ts:
        slot_ts = (
            pd.concat(equity_ts, ignore_index=True)
            .drop_duplicates()
            .sort_values(ignore_index=True)
        )
    else:
        slot_ts = pd.Series([], dtype=_TS_DTYPE)

    grid = pd.DataFrame({"ts_utc": slot_ts})
    grid = _attach_leg(grid, kr, "kr")
    grid = _attach_leg(grid, us, "us")
    grid = _attach_leg(grid, fx, "fx")

    # 격자 슬롯은 주식 봉의 합집합이므로 모든 행에 live 주식이 >= 1개 존재
    kr_live = grid["kr_live"]
    us_live = grid["us_live"]
    grid["session"] = np.select(
        [kr_live & us_live, kr_live],
        ["BOTH", "KR"],
        default="US",
    )

    # 세션 시작 감지: 이 슬롯에 live인데 직전 격자 슬롯에는 live가 아니었음
    # (첫 행은 직전 슬롯이 없으므로 open_slot == live)
    for key in ("kr", "us"):
        live = grid[f"{key}_live"]
        prev_live = live.shift(1, fill_value=False)
        grid[f"{key}_open_slot"] = (live & ~prev_live).astype(bool)

    # FX age가 NaN이면 FX 데이터 자체가 없다는 뜻 — 그것도 고장이므로 stale
    fx_age = grid["fx_age_min"]
    grid["fx_stale"] = (fx_age.isna() | (fx_age > FX_STALE_MINUTES)).astype(bool)

    return grid[GRID_COLUMNS]
