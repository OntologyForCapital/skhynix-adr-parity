"""parity.store 단위 테스트 — 네트워크 없음, tmp_path 사용."""
from __future__ import annotations

import pandas as pd
import pytest

from parity.config import BAR_COLUMNS, INSTRUMENTS
from parity.store import (
    MergeStats,
    StoreError,
    empty_frame,
    load_store,
    merge_bars,
    write_store,
)

KR = INSTRUMENTS["kr"]

_HEADER = "ts_utc,open,high,low,close,volume\n"
_ROW_A = "2026-07-15T00:00:00Z,10.0000,10.5000,9.5000,10.2000,100\n"
_ROW_B = "2026-07-15T00:30:00Z,10.2000,10.8000,10.0000,10.6000,200\n"


def _frame(rows: list[tuple]) -> pd.DataFrame:
    """(ts, open, high, low, close, volume) 튜플 목록 → 표준 프레임."""
    frame = pd.DataFrame(rows, columns=BAR_COLUMNS)
    frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    for col in ("open", "high", "low", "close"):
        frame[col] = frame[col].astype("float64")
    frame["volume"] = frame["volume"].astype("int64")
    return frame


# S1: 같은 ts는 fresh가 이기고 revised로 집계된다
def test_s1_fresh_wins_and_counts_revised() -> None:
    t = "2026-07-15T00:00:00Z"
    stored = _frame([(t, 9.0, 11.0, 8.0, 10.0, 100)])
    fresh = _frame([(t, 9.0, 11.0, 8.0, 10.5, 100)])

    merged, stats = merge_bars(stored, fresh)

    assert len(merged) == 1
    assert merged.loc[0, "close"] == 10.5
    assert stats == MergeStats(added=0, revised=1)


# S1 보강: 허용오차(1e-9) 이내 차이는 revised가 아니다
def test_s1_within_tolerance_not_revised() -> None:
    t = "2026-07-15T00:00:00Z"
    stored = _frame([(t, 9.0, 11.0, 8.0, 10.0, 100)])
    fresh = stored.copy()
    fresh.loc[0, "close"] = 10.0 + 1e-12

    _, stats = merge_bars(stored, fresh)

    assert stats == MergeStats(added=0, revised=0)


# S2: 뒤섞인 입력을 병합해도 정렬·유일 불변식이 유지되고, 병합은 멱등이다
def test_s2_sorted_unique_invariant_and_idempotency() -> None:
    stored = _frame(
        [
            ("2026-07-15T00:00:00Z", 1.0, 1.0, 1.0, 1.0, 10),
            ("2026-07-15T01:00:00Z", 3.0, 3.0, 3.0, 3.0, 30),
        ]
    )
    fresh = _frame(
        [
            ("2026-07-15T01:30:00Z", 4.0, 4.0, 4.0, 4.0, 40),  # 일부러 역순 입력
            ("2026-07-15T00:30:00Z", 2.0, 2.0, 2.0, 2.0, 20),
        ]
    )

    merged, stats = merge_bars(stored, fresh)

    assert merged["ts_utc"].is_monotonic_increasing
    assert merged["ts_utc"].is_unique
    assert len(merged) == 4
    assert stats == MergeStats(added=2, revised=0)

    again, stats2 = merge_bars(merged, fresh)
    pd.testing.assert_frame_equal(again, merged)
    assert stats2 == MergeStats(added=0, revised=0)


# S3: 첫 기록 True, 동일 재기록 False(바이트 동일성), 라운드트립 정확
def test_s3_write_byte_identity_and_roundtrip(tmp_path) -> None:
    frame = _frame(
        [
            ("2026-07-15T00:00:00Z", 10.1234, 10.5, 9.5, 10.25, 100),
            ("2026-07-15T00:30:00Z", 10.25, 10.75, 10.0, 10.5, 200),
        ]
    )

    assert write_store(KR, frame, data_dir=tmp_path) is True
    assert write_store(KR, frame, data_dir=tmp_path) is False  # 동일 바이트 → 미기록

    loaded = load_store(KR, data_dir=tmp_path)
    pd.testing.assert_frame_equal(loaded, frame)  # 4dp 값이라 정확히 왕복

    changed = frame.copy()
    changed.loc[0, "close"] = 10.26
    assert write_store(KR, changed, data_dir=tmp_path) is True


# S4: 손상된 CSV는 StoreError로 크게 실패한다
def test_s4_load_store_rejects_corrupt_csv(tmp_path) -> None:
    path = tmp_path / KR.filename

    path.write_text(_HEADER + _ROW_B + _ROW_A, encoding="utf-8")  # 순서 뒤섞임
    with pytest.raises(StoreError):
        load_store(KR, data_dir=tmp_path)

    path.write_text(_HEADER + _ROW_A + _ROW_A, encoding="utf-8")  # ts 중복
    with pytest.raises(StoreError):
        load_store(KR, data_dir=tmp_path)

    path.write_text(  # volume 컬럼 누락
        "ts_utc,open,high,low,close\n2026-07-15T00:00:00Z,1.0,1.0,1.0,1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(StoreError):
        load_store(KR, data_dir=tmp_path)

    path.write_text(  # close 결측
        _HEADER + "2026-07-15T00:00:00Z,1.0000,1.0000,1.0000,,100\n",
        encoding="utf-8",
    )
    with pytest.raises(StoreError):
        load_store(KR, data_dir=tmp_path)


# S5: 빈 저장소 + fresh → 전부 added / 파일 없으면 빈 프레임
def test_s5_empty_store_plus_fresh(tmp_path) -> None:
    missing = load_store(KR, data_dir=tmp_path)  # 파일 없음
    pd.testing.assert_frame_equal(missing, empty_frame())

    fresh = _frame(
        [
            ("2026-07-15T00:00:00Z", 1.0, 1.0, 1.0, 1.0, 10),
            ("2026-07-15T00:30:00Z", 2.0, 2.0, 2.0, 2.0, 20),
        ]
    )
    merged, stats = merge_bars(empty_frame(), fresh)

    assert stats == MergeStats(added=len(fresh), revised=0)
    pd.testing.assert_frame_equal(merged, fresh)


# S6: 저장 정밀도(4dp) 아래의 차이는 revised가 아니다
# (CSV round-trip한 값 vs 원본 정밀도 재조회값 — 실제 파일이 안 바뀌면 정정 아님)
def test_s6_sub_precision_diff_not_revised(tmp_path) -> None:
    t = "2026-07-15T00:00:00Z"
    stored = _frame([(t, 144.2000, 144.2000, 144.2000, 144.2000, 100)])
    fresh = _frame(
        [(t, 144.20000091552734, 144.2000004, 144.1999996, 144.20000305, 100)]
    )
    merged, stats = merge_bars(stored, fresh)
    assert stats == MergeStats(added=0, revised=0)

    # 4dp에서 실제로 달라지면 revised다
    fresh2 = _frame([(t, 144.2000, 144.2000, 144.2000, 144.2001, 100)])
    _, stats2 = merge_bars(stored, fresh2)
    assert stats2 == MergeStats(added=0, revised=1)

    # write_store도 바이트 동일해야 한다 (커밋 가드와 일관)
    assert write_store(KR, stored, data_dir=tmp_path) is True
    assert write_store(KR, merged, data_dir=tmp_path) is False
