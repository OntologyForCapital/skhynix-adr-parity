"""Streamlit 페이지 본문.

streamlit을 임포트하는 모듈은 이 파일과 streamlit_app.py 뿐이다.
데이터 흐름: 저장분(load_store) + 런타임 top-up(fetch_bars, 5분 캐시) 병합
-> 기간 필터 -> charts 빌더.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from parity import charts, config, fetch, ratio, store
from parity.grid import build_ratio_grid

# 기간 프리셋 -> 필터 폭 (None = 전체)
_PERIODS: dict[str, "pd.Timedelta | None"] = {
    "1주": pd.Timedelta(days=7),
    "2주": pd.Timedelta(days=14),
    "1개월": pd.Timedelta(days=30),
    "3개월": pd.Timedelta(days=90),
    "전체": None,
}

_MODES = {"괴리율(배)": "ratio", "ADR 프리미엄(%)": "premium"}

_STALE_STORE_DAYS = 3


@st.cache_data(ttl=300, show_spinner="실시간 시세 조회 중…")
def _cached_topup(key: str) -> "tuple[pd.DataFrame, pd.Timestamp] | None":
    """5분 캐시 top-up 조회. 실패하면 None (저장분만으로 렌더).

    조회 시각을 함께 반환한다 — provisional(진행 중 봉) 판정은 렌더 시각이
    아니라 조회 시각 기준이어야 한다. 캐시에 남은 진행 중 봉이 봉 종료 후
    렌더에서 확정봉으로 둔갑하는 것을 막는다.
    대화형 경로는 빨리 실패해야 하므로 재시도를 배치(기본 4회)보다 짧게 잡는다.
    """
    inst = config.INSTRUMENTS[key]
    try:
        frame = fetch.fetch_bars(
            inst, period=config.TOPUP_PERIOD, drop_partial=False, max_attempts=2
        )
    except Exception:  # FetchError 포함 — 네트워크 실패는 화면을 죽이지 않는다
        return None
    return frame, pd.Timestamp.now(tz="UTC")


def _fmt_kst(ts: object) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    return ts.tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M")  # type: ignore[union-attr]


def _fmt_et(ts: object) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    return ts.tz_convert("America/New_York").strftime("%m-%d %H:%M")  # type: ignore[union-attr]


def _age_status(value: object) -> str:
    if pd.isna(value):
        return "—"
    minutes = int(value)  # type: ignore[arg-type]
    return "실시간" if minutes <= 0 else f"이월 {minutes}분"


def _last_two(frame: pd.DataFrame, column: str) -> "tuple[float | None, float | None]":
    """마지막 값과 그 직전 값 (없으면 None)."""
    if frame.empty:
        return None, None
    series = frame[column]
    last = float(series.iloc[-1])
    prev = float(series.iloc[-2]) if len(series) >= 2 else None
    return last, prev


def render_page() -> None:
    """대시보드 전체 본문."""
    now = pd.Timestamp.now(tz="UTC")

    # ------------------------------------------------------------------
    # 1. 데이터: 저장분 + 실시간 top-up 병합, provisional 표시
    # ------------------------------------------------------------------
    stored_frames: dict[str, pd.DataFrame] = {}
    live_frames: dict[str, "pd.DataFrame | None"] = {}
    frames: dict[str, pd.DataFrame] = {}
    failed_legs: list[str] = []
    for key, inst in config.INSTRUMENTS.items():
        stored = store.load_store(inst)
        topup = _cached_topup(key)
        if topup is None:
            failed_legs.append(inst.name_ko)
            live = None
            frame = stored.copy()
            frame["provisional"] = False  # 저장분은 완결봉만 담는다
        else:
            live, fetched_at = topup
            frame = store.merge_bars(stored, live)[0].copy()
            frame["provisional"] = (
                frame["ts_utc"] + pd.Timedelta(seconds=config.BAR_SECONDS)
                > fetched_at
            )
        stored_frames[key] = stored
        live_frames[key] = live
        frames[key] = frame

    st.title("SK하이닉스 본주 vs ADR(SKHY) 괴리율")

    if failed_legs:
        if len(failed_legs) == len(config.INSTRUMENTS):
            st.warning("실시간 갱신 실패 — 저장된 데이터만 표시 중입니다")
        else:
            st.warning(
                "일부 실시간 갱신 실패 (" + ", ".join(failed_legs) + ") — "
                "해당 종목은 저장된 데이터 기준입니다"
            )

    # ------------------------------------------------------------------
    # 2. 저장분(리포지토리) 신선도 배너
    # ------------------------------------------------------------------
    stored_last = [
        frame["ts_utc"].max() for frame in stored_frames.values() if not frame.empty
    ]
    # min(): 한 종목만 수집이 죽어도 감지해야 한다 (max()는 환율만 살아 있어도 침묵)
    if stored_last and now - min(stored_last) > pd.Timedelta(days=_STALE_STORE_DAYS):
        st.warning(
            f"저장 데이터 중 {_STALE_STORE_DAYS}일 이상 갱신되지 않은 종목이 "
            "있습니다 — GitHub Actions 탭에서 수집 워크플로 상태를 확인하세요."
        )

    # ------------------------------------------------------------------
    # 3. 사이드바
    # ------------------------------------------------------------------
    with st.sidebar:
        period_label = st.radio(
            "기간 프리셋", list(_PERIODS), index=list(_PERIODS).index("2주")
        )
        mode_label = st.radio("괴리율 표시", list(_MODES), index=0)
        if st.button("새로고침"):
            _cached_topup.clear()
            st.rerun()
        with st.expander("데이터 상태"):
            status_rows = []
            for key, inst in config.INSTRUMENTS.items():
                stored = stored_frames[key]
                live = live_frames[key]
                status_rows.append(
                    {
                        "종목": inst.name_ko,
                        "저장 마지막 봉(KST)": _fmt_kst(
                            stored["ts_utc"].max() if not stored.empty else None
                        ),
                        "실시간 마지막 봉(KST)": _fmt_kst(
                            live["ts_utc"].max()
                            if live is not None and not live.empty
                            else None
                        ),
                        "저장 행수": len(stored),
                        "실시간 행수": 0 if live is None else len(live),
                        "저장 시작일": (
                            "—"
                            if stored.empty
                            else stored["ts_utc"]
                            .min()
                            .tz_convert("Asia/Seoul")
                            .strftime("%Y-%m-%d")
                        ),
                    }
                )
            st.dataframe(
                pd.DataFrame(status_rows), hide_index=True, use_container_width=True
            )

    period_delta = _PERIODS[period_label]
    mode = _MODES[mode_label]

    # ------------------------------------------------------------------
    # 4. 메트릭 (전체 병합 프레임 기준 — 기간 프리셋과 무관)
    # ------------------------------------------------------------------
    ratio_full = pd.DataFrame()
    if all(not frames[key].empty for key in ("kr", "us", "fx")):
        ratio_full = ratio.compute_ratio(
            build_ratio_grid(frames["kr"], frames["us"], frames["fx"])
        ).dropna(subset=["ratio"])

    kr_last, kr_prev = _last_two(frames["kr"], "close")
    us_last, us_prev = _last_two(frames["us"], "close")
    fx_last, fx_prev = _last_two(frames["fx"], "close")
    prem_last, prem_prev = _last_two(ratio_full, "premium_pct")

    col_kr, col_us, col_fx, col_prem = st.columns(4)
    if kr_last is None:
        col_kr.metric("000660 현재가", "—")
    else:
        delta = None
        if kr_prev:
            change = kr_last - kr_prev
            delta = f"{change:+,.0f} ({change / kr_prev * 100:+.2f}%)"
        col_kr.metric("000660 현재가", f"₩{kr_last:,.0f}", delta)
    if us_last is None:
        col_us.metric("SKHY 현재가", "—")
    else:
        delta = None
        if us_prev:
            change = us_last - us_prev
            delta = f"{change:+.2f} ({change / us_prev * 100:+.2f}%)"
        col_us.metric("SKHY 현재가", f"${us_last:,.2f}", delta)
    if fx_last is None:
        col_fx.metric("USD/KRW", "—")
    else:
        delta = f"{fx_last - fx_prev:+.2f}" if fx_prev else None
        col_fx.metric("USD/KRW", f"{fx_last:,.2f}", delta)
    if prem_last is None:
        col_prem.metric("ADR 프리미엄", "—")
    else:
        delta = f"{prem_last - prem_prev:+.2f}%p" if prem_prev is not None else None
        col_prem.metric("ADR 프리미엄", f"{prem_last:+.2f}%", delta)

    # ------------------------------------------------------------------
    # 5. 차트 3단 (기간 필터는 add_trade_index 이전에 적용)
    # ------------------------------------------------------------------
    if period_delta is None:
        view = {key: frame for key, frame in frames.items()}
    else:
        cutoff = now - period_delta
        view = {
            key: frame[frame["ts_utc"] >= cutoff].reset_index(drop=True)
            for key, frame in frames.items()
        }

    # 5-1. 본주
    inst_kr = config.INSTRUMENTS["kr"]
    st.subheader(f"{inst_kr.name_ko} (000660)")
    if view["kr"].empty:
        st.info("선택한 기간에 본주 데이터가 없습니다.")
    else:
        st.altair_chart(
            charts.price_line_chart(view["kr"], inst_kr, color=config.COLOR_KR),
            use_container_width=True,
        )
        last_row = view["kr"].iloc[-1]
        suffix = " (진행 중 봉)" if bool(last_row["provisional"]) else ""
        st.caption(f"마지막 봉: {_fmt_kst(last_row['ts_utc'])} KST{suffix}")

    # 5-2. ADR
    inst_us = config.INSTRUMENTS["us"]
    st.subheader(f"{inst_us.name_ko} (SKHY)")
    if view["us"].empty:
        st.info("선택한 기간에 ADR 데이터가 없습니다.")
    else:
        st.altair_chart(
            charts.price_line_chart(view["us"], inst_us, color=config.COLOR_US),
            use_container_width=True,
        )
        last_row = view["us"].iloc[-1]
        suffix = " (진행 중 봉)" if bool(last_row["provisional"]) else ""
        st.caption(
            f"마지막 봉: {_fmt_kst(last_row['ts_utc'])} KST / "
            f"{_fmt_et(last_row['ts_utc'])} ET{suffix}"
        )

    # 5-3. 괴리율
    st.subheader("괴리율 / ADR 프리미엄")
    if ratio_full.empty:
        st.info("괴리율을 계산할 데이터가 부족합니다 (본주/ADR/환율 중 일부 없음).")
    else:
        # 기간 필터는 격자 계산 *후* 격자 행에 적용한다. 다리 프레임을 먼저
        # 자르면 창 시작 부분에서 이월값을 못 찾아 행이 사라지고, 개장 첫 봉
        # 마커가 창 경계에서 가짜로 생긴다.
        if period_delta is None:
            ratio_view = ratio_full.reset_index(drop=True)
        else:
            ratio_view = ratio_full[
                ratio_full["ts_utc"] >= now - period_delta
            ].reset_index(drop=True)
        if ratio_view.empty:
            st.info("선택한 기간에 괴리율 데이터가 없습니다.")
        else:
            st.altair_chart(
                charts.ratio_chart(ratio_view, mode=mode),
                use_container_width=True,
            )
            last_row = ratio_view.iloc[-1]
            st.caption(
                f"마지막 갱신 상태 — 본주 {_age_status(last_row['kr_age_min'])} · "
                f"ADR {_age_status(last_row['us_age_min'])} · "
                f"환율 {last_row['fx_close']:,.2f}"
            )
            if "fx_stale" in ratio_view.columns and bool(last_row["fx_stale"]):
                st.error(
                    f"환율 피드 지연 — 마지막 환율 봉이 {config.FX_STALE_MINUTES}분 "
                    "이상 갱신되지 않았습니다. 괴리율이 왜곡될 수 있습니다."
                )

    # ------------------------------------------------------------------
    # 6. 푸터
    # ------------------------------------------------------------------
    st.caption(
        "데이터 출처: Yahoo Finance 30분봉(지연/무료 시세) · "
        f"ADR {config.ADR_RATIO}주 = 본주 1주 · 원시(무수정) 가격 사용 · "
        "본주 30분봉은 15:00 KST까지만 제공되어 마감 동시호가(15:20~15:30)가 "
        "반영되지 않습니다 — 미국 장중 괴리율의 본주 다리는 15:00 가격 기준입니다"
    )
