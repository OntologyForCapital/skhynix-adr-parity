"""compute_ratio 단위 테스트 — 순수 계산·NaN 전파 검증."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from parity.ratio import compute_ratio


def make_grid(kr_closes, us_closes, fx_closes) -> pd.DataFrame:
    """ratio 계산에 필요한 최소 격자 프레임."""
    return pd.DataFrame(
        {
            "kr_close": kr_closes,
            "us_close": us_closes,
            "fx_close": fx_closes,
        },
        dtype="float64",
    )


def test_r1_basic_ratio_and_premium():
    grid = make_grid([1_322_000.0], [100.0], [1000.0])
    out = compute_ratio(grid)

    # 1,322,000 / (100 * 10 * 1,000) = 1.3220
    assert out.loc[0, "ratio"] == pytest.approx(1.3220)
    assert out.loc[0, "premium_pct"] == pytest.approx((1 / 1.322 - 1) * 100)

    assert out["ratio"].dtype == np.float64
    assert out["premium_pct"].dtype == np.float64

    # 사본 반환 — 원본 격자는 변경되지 않는다
    assert "ratio" not in grid.columns
    assert "premium_pct" not in grid.columns


def test_r2_realistic_adr_premium():
    grid = make_grid([1_322_000.0], [115.50], [1385.0])
    out = compute_ratio(grid)

    # 분모 = 115.50 * 10 * 1385.0 = 1,599,675
    assert out.loc[0, "ratio"] == pytest.approx(1_322_000 / 1_599_675)

    exact_premium = (1_599_675 / 1_322_000 - 1) * 100
    assert out.loc[0, "premium_pct"] == pytest.approx(exact_premium)
    # 소수 둘째 자리 기준 약 +21.0%
    assert round(float(out.loc[0, "premium_pct"]), 2) == pytest.approx(21.0)


def test_r3_direction_and_dimension():
    # 130,000 KRW = 10 USD * 10 (ADR/본주) * 1,300 KRW/USD → 정확히 패리티
    # (곱하기 10이지 나누기 10이 아님: /10이었다면 ratio는 100이 된다)
    grid = make_grid([130_000.0, 130_000.0], [10.0, 10.0], [1300.0, 2600.0])
    out = compute_ratio(grid)

    assert out.loc[0, "ratio"] == pytest.approx(1.0)
    # 환율이 두 배가 되면 ratio는 절반
    assert out.loc[1, "ratio"] == pytest.approx(0.5)


def test_r4_nan_propagates_without_raising():
    grid = make_grid(
        [1_322_000.0, 1_322_000.0, np.nan],
        [np.nan, 100.0, 100.0],
        [1385.0, np.nan, 1385.0],
    )
    out = compute_ratio(grid)  # 예외 없이 통과해야 한다

    assert out["ratio"].isna().all()
    assert out["premium_pct"].isna().all()
    assert len(out) == 3  # dropna 하지 않는다


# R5: 0가격 퇴화 입력 → inf가 아니라 NaN (dropna로 걸러지도록)
def test_r5_zero_price_yields_nan_not_inf() -> None:
    out = compute_ratio(make_grid([130_000.0], [0.0], [1_300.0]))
    assert not np.isinf(out["ratio"]).any()
    assert out["ratio"].isna().all()
    assert out["premium_pct"].isna().all()

    # kr=0 → ratio 0 → premium_pct inf → NaN 처리
    out2 = compute_ratio(make_grid([0.0], [10.0], [1_300.0]))
    assert not np.isinf(out2["premium_pct"]).any()
