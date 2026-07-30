"""CSV 저장소 입출력과 병합.

repo에 커밋되는 data/*.csv가 유일한 영속 저장소다(Streamlit Cloud
파일시스템은 휘발성). 같은 데이터는 항상 바이트 단위로 동일한 파일을
만들어야 하며(결정적 직렬화), 이 바이트 동일성이 GitHub Actions의
커밋 가드 신호로 쓰인다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from parity.config import BAR_COLUMNS, DATA_DIR, FLOAT_FORMAT, TS_FORMAT, Instrument

_VALUE_COLUMNS = ["open", "high", "low", "close", "volume"]
_FLOAT_TOLERANCE = 1e-9
# 저장 정밀도(FLOAT_FORMAT "%.4f")의 소수 자릿수 — revised 비교 기준
_STORE_DECIMALS = int(FLOAT_FORMAT.rstrip("f").rsplit(".", 1)[1])


class StoreError(RuntimeError):
    """저장소 CSV의 손상 또는 규약 위반."""


@dataclass(frozen=True)
class MergeStats:
    """병합 결과 요약: 신규 봉 수와 값이 수정된 봉 수."""

    added: int
    revised: int


def empty_frame() -> pd.DataFrame:
    """표준 스키마의 빈 프레임(정확한 dtype 포함)을 반환한다."""
    return pd.DataFrame(
        {
            "ts_utc": pd.Series([], dtype="datetime64[ns, UTC]"),
            "open": pd.Series([], dtype="float64"),
            "high": pd.Series([], dtype="float64"),
            "low": pd.Series([], dtype="float64"),
            "close": pd.Series([], dtype="float64"),
            "volume": pd.Series([], dtype="int64"),
        }
    )[BAR_COLUMNS]


def load_store(inst: Instrument, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """종목 CSV를 읽어 표준 프레임으로 반환한다. 파일이 없으면 빈 프레임.

    컬럼 불일치, ts_utc 순증가·중복 없음 위반, close 결측 등 규약 위반은
    StoreError로 크게 실패시킨다 (조용한 복구 금지).
    """
    path = Path(data_dir) / inst.filename
    if not path.exists():
        return empty_frame()

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise StoreError(f"{path.name}: CSV를 읽을 수 없습니다") from exc

    if list(frame.columns) != BAR_COLUMNS:
        raise StoreError(
            f"{path.name}: 컬럼 불일치 {list(frame.columns)} != {BAR_COLUMNS}"
        )

    try:
        ts = pd.to_datetime(frame["ts_utc"], utc=True)
    except Exception as exc:
        raise StoreError(f"{path.name}: ts_utc 파싱 실패") from exc
    if ts.isna().any():
        raise StoreError(f"{path.name}: ts_utc에 결측/파싱 불가 값이 있습니다")
    if not (ts.is_unique and ts.is_monotonic_increasing):
        raise StoreError(f"{path.name}: ts_utc가 순증가·중복 없음 규약을 위반했습니다")
    if pd.to_numeric(frame["close"], errors="coerce").isna().any():
        raise StoreError(f"{path.name}: close에 결측값이 있습니다")

    try:
        out = pd.DataFrame(
            {
                "ts_utc": ts.astype("datetime64[ns, UTC]"),
                "open": frame["open"].astype("float64"),
                "high": frame["high"].astype("float64"),
                "low": frame["low"].astype("float64"),
                "close": frame["close"].astype("float64"),
                "volume": frame["volume"].astype("int64"),
            }
        )
    except Exception as exc:
        raise StoreError(f"{path.name}: dtype 변환 실패") from exc
    return out[BAR_COLUMNS].reset_index(drop=True)


def merge_bars(
    stored: pd.DataFrame, fresh: pd.DataFrame
) -> tuple[pd.DataFrame, MergeStats]:
    """stored 위에 fresh를 겹쳐 병합한다 (같은 ts는 fresh가 이긴다).

    added   = fresh에만 있던 ts 수
    revised = 양쪽에 있으면서 OHLCV 중 하나라도 달라진 ts 수.
              CSV는 FLOAT_FORMAT(%.4f)으로 라운딩돼 저장되므로, 저장을 거친 값과
              원본 정밀도의 재조회값을 그대로 비교하면 전 행이 정정으로 잡힌다.
              따라서 저장 정밀도로 라운딩한 뒤 비교한다 (실제 파일에 반영될
              변화만 정정으로 센다).
    """
    combined = pd.concat([stored, fresh], ignore_index=True)
    merged = (
        combined.sort_values("ts_utc", kind="stable")
        .drop_duplicates("ts_utc", keep="last")
        .reset_index(drop=True)
    )[BAR_COLUMNS]

    stored_last = stored.drop_duplicates("ts_utc", keep="last").set_index("ts_utc")
    fresh_last = fresh.drop_duplicates("ts_utc", keep="last").set_index("ts_utc")
    common = fresh_last.index.intersection(stored_last.index)
    added = int(fresh_last.index.difference(stored_last.index).size)

    revised = 0
    if len(common) > 0:
        before = stored_last.loc[common, _VALUE_COLUMNS].to_numpy(dtype="float64")
        after = fresh_last.loc[common, _VALUE_COLUMNS].to_numpy(dtype="float64")
        changed = ~np.isclose(
            np.round(before, _STORE_DECIMALS),
            np.round(after, _STORE_DECIMALS),
            rtol=0.0,
            atol=_FLOAT_TOLERANCE,
            equal_nan=True,
        )
        revised = int(changed.any(axis=1).sum())

    return merged, MergeStats(added=added, revised=revised)


def write_store(
    inst: Instrument, frame: pd.DataFrame, data_dir: Path = DATA_DIR
) -> bool:
    """프레임을 결정적으로 직렬화해 기록한다.

    같은 데이터는 항상 동일한 바이트를 만들므로, 기존 파일과 바이트가
    같으면 쓰지 않고 False를 반환한다 (Actions 커밋 가드 신호).
    """
    path = Path(data_dir) / inst.filename

    out = frame.loc[:, BAR_COLUMNS].copy()
    out["ts_utc"] = out["ts_utc"].dt.tz_convert("UTC").dt.strftime(TS_FORMAT)
    payload = out.to_csv(
        index=False, float_format=FLOAT_FORMAT, lineterminator="\n"
    ).encode("utf-8")

    if path.exists() and path.read_bytes() == payload:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    # 원자적 교체: 쓰기 도중 실패해도 유일한 영구 저장소인 CSV가
    # 절단된 채 남지 않는다 (임시 파일 후 os.replace)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return True
