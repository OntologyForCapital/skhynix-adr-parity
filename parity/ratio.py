"""괴리율(패리티) 계산 — 순수 함수, I/O 없음."""
from __future__ import annotations

import numpy as np
import pandas as pd

from parity.config import ADR_RATIO


def compute_ratio(grid: pd.DataFrame) -> pd.DataFrame:
    """격자에 ratio / premium_pct 컬럼을 더한 사본을 반환한다.

    ratio = kr_close / (us_close * ADR_RATIO * fx_close)
        KRW / (USD * (ADR/본주) * (KRW/USD)) → 무차원.
        1.0 미만이면 환산 ADR 가격이 본주보다 비싸다는 뜻 (ADR 프리미엄).
    premium_pct = (1/ratio - 1) * 100
        ADR 프리미엄(%). ratio 0.83 → 약 +21%.

    어느 leg든 NaN이면 ratio/premium_pct도 NaN으로 전파된다.
    여기서 dropna 하지 않는다 — 표시 여부는 차트 레이어가 결정.
    """
    out = grid.copy()
    denom = (
        out["us_close"].astype("float64")
        * float(ADR_RATIO)
        * out["fx_close"].astype("float64")
    )
    out["ratio"] = out["kr_close"].astype("float64") / denom
    # 0가격 등 퇴화 입력의 ±inf는 NaN으로 — dropna(subset=["ratio"])가
    # 실제로 걸러내도록 (inf는 dropna를 통과해 차트 축을 붕괴시킨다).
    # premium은 정리된 ratio에서 계산해야 -100% 같은 유령 값이 안 생긴다.
    out["ratio"] = out["ratio"].replace([np.inf, -np.inf], np.nan)
    out["premium_pct"] = (1.0 / out["ratio"] - 1.0) * 100.0
    out["premium_pct"] = out["premium_pct"].replace([np.inf, -np.inf], np.nan)
    return out
