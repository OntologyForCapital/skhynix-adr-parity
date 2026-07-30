"""데이터 갱신 파이프라인의 유일한 진입점.

첫 실행이 곧 60일 백필이고 이후 실행은 증분 갱신이다 — 동일 코드, 완전 멱등.
매 실행마다 yfinance 60일 전체 창을 다시 받아 저장분과 병합하므로
장기간 실행이 끊겨도 다음 실행에서 스스로 복구한다.

GitHub Actions cron이 평일 2회 실행하며, 로컬에서도 인자 없이 그대로 실행한다.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# 어디서 실행해도 저장소 루트를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parity import config, fetch, store  # noqa: E402
from parity.config import Instrument  # noqa: E402


def _stat(stats: object, key: str) -> object:
    """dict 또는 dataclass 형태의 stats에서 값을 꺼낸다."""
    if isinstance(stats, dict):
        return stats.get(key, 0)
    return getattr(stats, key, 0)


def update_instrument(inst: Instrument) -> str:
    """한 종목을 갱신하고 요약 한 줄을 돌려준다."""
    fresh = fetch.fetch_bars(inst, period=config.FETCH_PERIOD, drop_partial=True)
    stored = store.load_store(inst)
    merged, stats = store.merge_bars(stored, fresh)
    store.write_store(inst, merged)

    label = Path(inst.filename).stem
    added = _stat(stats, "added")
    revised = _stat(stats, "revised")
    total = len(merged)
    last = merged["ts_utc"].iloc[-1].strftime(config.TS_FORMAT) if total else "n/a"
    return f"{label}: +{added} added, {revised} revised, total {total} rows, last {last}"


def main() -> int:
    """세 종목을 각각 갱신한다.

    한 종목이라도 성공하면 0을 반환해 Actions가 성공분을 커밋하게 하고,
    셋 모두 실패하면 1을 반환해 빨간 실행(실패 메일)을 남긴다.
    """
    succeeded = 0
    for key, inst in config.INSTRUMENTS.items():
        try:
            print(update_instrument(inst))
            succeeded += 1
        except Exception:
            print(f"{key}: 갱신 실패 ({inst.ticker})", file=sys.stderr)
            traceback.print_exc()
    return 0 if succeeded > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
