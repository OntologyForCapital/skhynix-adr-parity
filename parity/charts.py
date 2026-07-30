"""순수 Altair 차트 빌더.

부모 프로젝트(kis_hourly_dashboard)의 검증된 기법을 그대로 옮긴다:
- 갭 없는 x축: 기간 필터 후 trade_index(0..n-1)를 부여하고 모든 x 인코딩은
  trade_index:Q 를 쓴다(timestamp:T 금지 — 휴장/비중첩 세션 구간이 접힌다).
- 눈금 라벨: 최대 8개 위치를 골라 trade_index -> KST 라벨 문자열을 매핑하는
  labelExpr 축을 만든다.
- 하단 오버뷰 스트립의 interval brush 가 메인 패널의 x 스케일 domain을 구동한다.

이 모듈은 streamlit 및 형제 모듈(fetch/store/grid/ratio)을 임포트하지 않는다 —
DataFrame만 소비한다.
"""
from __future__ import annotations

import json

import altair as alt
import pandas as pd

from parity.config import COLOR_KR, COLOR_NEUTRAL, COLOR_US, Instrument

alt.data_transformers.disable_max_rows()

_SESSION_LABELS = {"KR": "KR 장중", "US": "US 장중"}


def add_trade_index(frame: pd.DataFrame) -> pd.DataFrame:
    """ts_utc 오름차순 복사본에 trade_index와 표시용 라벨 컬럼을 부여한다.

    - trade_index: 0..n-1 (갭 없는 x축의 좌표)
    - label_kst: "MM-DD HH:MM" (Asia/Seoul)
    - label_et:  "MM-DD HH:MM" (America/New_York, US 차트 툴팁용)
    """
    result = frame.sort_values("ts_utc").reset_index(drop=True).copy()
    result["trade_index"] = range(len(result))
    result["label_kst"] = (
        result["ts_utc"].dt.tz_convert("Asia/Seoul").dt.strftime("%m-%d %H:%M")
    )
    result["label_et"] = (
        result["ts_utc"].dt.tz_convert("America/New_York").dt.strftime("%m-%d %H:%M")
    )
    return result


def trading_axis(frame: pd.DataFrame) -> alt.Axis:
    """trade_index -> KST 라벨을 매핑하는 labelExpr 축 (눈금 최대 8개).

    frame은 add_trade_index를 거친 상태여야 한다(label_kst 필요).
    """
    count = len(frame)
    if count == 0:
        return alt.Axis(title=None, labelAngle=0, tickSize=0, grid=False)
    tick_count = min(8, count)
    if tick_count <= 1:
        positions = [0]
    else:
        positions = sorted(
            {
                round(index * (count - 1) / (tick_count - 1))
                for index in range(tick_count)
            }
        )
    labels: dict[int, str] = {
        index: str(frame.iloc[index]["label_kst"]) for index in positions
    }
    label_expr = " : ".join(
        [
            f"datum.value == {index} ? {json.dumps(label, ensure_ascii=False)}"
            for index, label in labels.items()
        ]
        + ["''"]
    )
    return alt.Axis(
        title=None,
        values=positions,
        labelExpr=label_expr,
        labelAngle=0,
        labelPadding=7,
        tickSize=0,
        grid=False,
    )


def price_line_chart(
    frame: pd.DataFrame,
    inst: Instrument,
    *,
    color: str,
    height: int = 300,
) -> alt.VConcatChart:
    """단일 종목 가격 라인 차트 (라인 + 옅은 영역, 하단 brush 오버뷰).

    frame: 캐노니컬 봉 프레임. 선택적으로 boolean "provisional" 컬럼이 있으면
    진행 중 봉을 속 빈 원으로 표시하고 툴팁에 상태를 덧붙인다.
    """
    chart_frame = add_trade_index(frame)
    has_provisional = "provisional" in chart_frame.columns
    if has_provisional:
        chart_frame["bar_status"] = (
            chart_frame["provisional"].astype(bool).map(
                {True: "진행 중 봉", False: "확정 봉"}
            )
        )

    if inst.currency == "USD":
        close_tooltip = alt.Tooltip("close:Q", title="종가($)", format="$,.2f")
        axis_format = "$,.2f"
    else:
        close_tooltip = alt.Tooltip("close:Q", title="종가(₩)", format=",.0f")
        axis_format = ",.0f"

    tooltip: list[alt.Tooltip] = [alt.Tooltip("label_kst:N", title="시각(KST)")]
    if inst.key == "us":
        tooltip.append(alt.Tooltip("label_et:N", title="시각(ET)"))
    tooltip.append(close_tooltip)
    tooltip.append(alt.Tooltip("volume:Q", title="거래량", format=","))
    if has_provisional:
        tooltip.append(alt.Tooltip("bar_status:N", title="상태"))

    brush = alt.selection_interval(encodings=["x"], name=f"{inst.key}_range")
    x_scale = alt.Scale(domain=brush)
    x_axis = trading_axis(chart_frame)
    hover = alt.selection_point(
        fields=["trade_index"], nearest=True, on="pointerover", empty=False
    )
    base = alt.Chart(chart_frame)

    area = base.mark_area(color=color, opacity=0.10).encode(
        x=alt.X("trade_index:Q", scale=x_scale, axis=x_axis),
        y=alt.Y(
            "close:Q",
            title=None,
            # area 마크는 기본 stack(기준 0)이 걸려 zero=False를 무력화한다
            stack=None,
            scale=alt.Scale(zero=False),
            axis=alt.Axis(orient="right", format=axis_format, grid=True, tickCount=6),
        ),
    )
    line = base.mark_line(color=color, strokeWidth=1.6).encode(
        x=alt.X("trade_index:Q", scale=x_scale, axis=None),
        y=alt.Y("close:Q", scale=alt.Scale(zero=False), axis=None),
    )
    layers: list[alt.Chart] = [area, line]

    if has_provisional and bool(chart_frame["provisional"].astype(bool).any()):
        provisional_frame = chart_frame[chart_frame["provisional"].astype(bool)]
        layers.append(
            alt.Chart(provisional_frame)
            .mark_point(shape="circle", filled=False, color=color, size=70, strokeWidth=1.6)
            .encode(
                x=alt.X("trade_index:Q", scale=x_scale, axis=None),
                y=alt.Y("close:Q", scale=alt.Scale(zero=False), axis=None),
                tooltip=tooltip,
            )
        )

    selector = (
        base.mark_point(opacity=0, size=80)
        .encode(
            x=alt.X("trade_index:Q", scale=x_scale, axis=None),
            y=alt.Y("close:Q", scale=alt.Scale(zero=False), axis=None),
            tooltip=tooltip,
        )
        .add_params(hover)
    )
    crosshair = (
        base.mark_rule(color="#98a0aa", strokeWidth=1)
        .encode(x=alt.X("trade_index:Q", scale=x_scale, axis=None))
        .transform_filter(hover)
    )
    layers.extend([selector, crosshair])
    main = alt.layer(*layers).properties(height=height)

    overview = (
        base.mark_line(color=color, strokeWidth=1)
        .encode(
            x=alt.X("trade_index:Q", title=None, axis=None),
            y=alt.Y("close:Q", title=None, axis=None, scale=alt.Scale(zero=False)),
        )
        .add_params(brush)
        .properties(height=60)
    )
    return (
        alt.vconcat(main, overview, spacing=4)
        .resolve_scale(x="independent", y="independent")
        .configure_view(stroke=None)
        .configure_axis(
            domain=False,
            gridColor="#edf0f3",
            gridOpacity=1,
            labelColor="#68717d",
            tickColor="#cfd5dc",
        )
    )


def _age_label(value: object) -> str:
    """kr/us_age_min -> '실시간' 또는 '이월 N분' (결측은 '—')."""
    if pd.isna(value):
        return "—"
    minutes = int(value)  # type: ignore[arg-type]
    return "실시간" if minutes <= 0 else f"이월 {minutes}분"


def _session_bands(chart_frame: pd.DataFrame) -> pd.DataFrame:
    """session 컬럼의 연속 구간(run)을 (x, x2, session_label) 스팬으로 접는다."""
    if chart_frame.empty:
        return pd.DataFrame(columns=["x", "x2", "session_label"])
    session = chart_frame["session"]
    run_id = (session != session.shift()).cumsum()
    bands = (
        chart_frame.groupby(run_id)
        .agg(
            session=("session", "first"),
            start=("trade_index", "first"),
            end=("trade_index", "last"),
        )
        .reset_index(drop=True)
    )
    bands = bands[bands["session"].isin(list(_SESSION_LABELS))].copy()
    bands["x"] = bands["start"] - 0.5
    bands["x2"] = bands["end"] + 0.5
    bands["session_label"] = bands["session"].map(_SESSION_LABELS)
    return bands[["x", "x2", "session_label"]]


def ratio_chart(
    grid_frame: pd.DataFrame,
    *,
    mode: str = "ratio",
    height: int = 320,
) -> alt.VConcatChart:
    """괴리율/프리미엄 차트.

    grid_frame: compute_ratio 출력(호출자가 dropna(subset=["ratio"]) 완료).
    mode: "ratio" -> 괴리율(배), "premium" -> ADR 프리미엄(%).

    레이어(모두 trade_index 기준):
      1. 세션 배경 밴드 (KR/US 어느 시장이 열려 있는지 타임라인 표시)
      2. 메인 라인 (ratio 또는 premium_pct)
      3. 이론 패리티 점선 기준선 (ratio=1.0 / premium=0.0)
      4. 개장 슬롯 마커 (방금 열린 시장이 상대 다리의 누적 변동을 재가격하는 지점)
      5. 리치 툴팁 (표시 문자열은 데이터 컬럼으로 사전 계산 — altair 포맷 문자열은
         '이월 N분' 같은 조건부 텍스트를 만들 수 없다)
    """
    if mode not in ("ratio", "premium"):
        raise ValueError(f"알 수 없는 mode: {mode!r} (ratio | premium)")

    chart_frame = add_trade_index(grid_frame)

    # --- 표시 문자열 사전 계산 -------------------------------------------
    chart_frame["session_label"] = [
        _SESSION_LABELS.get(value, "—" if pd.isna(value) else str(value))
        for value in chart_frame["session"]
    ]
    chart_frame["kr_status"] = chart_frame["kr_age_min"].map(_age_label)
    chart_frame["us_status"] = chart_frame["us_age_min"].map(_age_label)
    chart_frame["kr_close_disp"] = [
        "—" if pd.isna(value) else f"₩{value:,.0f}" for value in chart_frame["kr_close"]
    ]
    chart_frame["us_close_disp"] = [
        "—" if pd.isna(value) else f"${value:,.2f}" for value in chart_frame["us_close"]
    ]
    chart_frame["fx_close_disp"] = [
        "—" if pd.isna(value) else f"{value:,.2f}" for value in chart_frame["fx_close"]
    ]

    if mode == "ratio":
        y_field = "ratio"
        y_format = ".4f"
        y_title = "괴리율 (본주 ÷ ADR×10×환율)"
        reference = 1.0
    else:
        y_field = "premium_pct"
        y_format = "+.1f"
        y_title = "ADR 프리미엄 (%)"
        reference = 0.0

    tooltip = [
        alt.Tooltip("label_kst:N", title="시각(KST)"),
        alt.Tooltip("session_label:N", title="세션"),
        alt.Tooltip("kr_close_disp:N", title="본주 종가"),
        alt.Tooltip("kr_status:N", title="본주 상태"),
        alt.Tooltip("us_close_disp:N", title="ADR 종가"),
        alt.Tooltip("us_status:N", title="ADR 상태"),
        alt.Tooltip("fx_close_disp:N", title="USD/KRW"),
        alt.Tooltip("ratio:Q", title="괴리율", format=".4f"),
        alt.Tooltip("premium_pct:Q", title="ADR 프리미엄(%)", format="+.2f"),
    ]

    brush = alt.selection_interval(encodings=["x"], name="ratio_range")
    x_scale = alt.Scale(domain=brush)
    x_axis = trading_axis(chart_frame)
    hover = alt.selection_point(
        fields=["trade_index"], nearest=True, on="pointerover", empty=False
    )

    # 1. 세션 배경 밴드 — 어느 시장이 살아 있는지 보여주는 타임라인 범례
    bands = _session_bands(chart_frame)
    band_layer = (
        alt.Chart(bands)
        .mark_rect(opacity=0.07)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=None),
            x2="x2:Q",
            color=alt.Color(
                "session_label:N",
                title=None,
                scale=alt.Scale(
                    domain=list(_SESSION_LABELS.values()),
                    range=[COLOR_KR, COLOR_US],
                ),
                legend=alt.Legend(
                    orient="top", direction="horizontal", symbolOpacity=0.5
                ),
            ),
        )
    )

    # 2. 메인 라인
    line = (
        alt.Chart(chart_frame)
        .mark_line(color=COLOR_NEUTRAL, strokeWidth=1.6)
        .encode(
            x=alt.X("trade_index:Q", scale=x_scale, axis=x_axis),
            y=alt.Y(
                f"{y_field}:Q",
                title=y_title,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(
                    orient="right", format=y_format, grid=True, tickCount=6
                ),
            ),
        )
    )

    # 3. 이론 패리티 기준선 (점선) + 라벨
    reference_frame = pd.DataFrame({"reference": [reference]})
    rule = (
        alt.Chart(reference_frame)
        .mark_rule(color=COLOR_NEUTRAL, strokeDash=[5, 4], strokeWidth=1)
        .encode(y=alt.Y("reference:Q", axis=None))
    )
    rule_label = (
        alt.Chart(reference_frame)
        .mark_text(
            text="이론 패리티",
            align="left",
            baseline="bottom",
            dx=4,
            dy=-3,
            fontSize=11,
            color=COLOR_NEUTRAL,
        )
        .encode(x=alt.value(4), y=alt.Y("reference:Q", axis=None))
    )

    # 4. 개장 슬롯 마커 (속 빈 원)
    if "kr_open_slot" in chart_frame.columns:
        kr_slot = chart_frame["kr_open_slot"].fillna(False).astype(bool)
    else:
        kr_slot = pd.Series(False, index=chart_frame.index)
    if "us_open_slot" in chart_frame.columns:
        us_slot = chart_frame["us_open_slot"].fillna(False).astype(bool)
    else:
        us_slot = pd.Series(False, index=chart_frame.index)
    slots = chart_frame[kr_slot | us_slot].copy()
    slots["slot_market"] = [
        "KR 개장" if flag else "US 개장" for flag in kr_slot.loc[slots.index]
    ]
    slot_layer = (
        alt.Chart(slots)
        .mark_point(shape="circle", filled=False, size=70, strokeWidth=1.6)
        .encode(
            x=alt.X("trade_index:Q", scale=x_scale, axis=None),
            y=alt.Y(f"{y_field}:Q", scale=alt.Scale(zero=False), axis=None),
            color=alt.Color(
                "slot_market:N",
                scale=alt.Scale(
                    domain=["KR 개장", "US 개장"], range=[COLOR_KR, COLOR_US]
                ),
                legend=None,
            ),
            tooltip=tooltip,
        )
    )

    # 5. 투명 셀렉터 + 크로스헤어 (리치 툴팁)
    selector = (
        alt.Chart(chart_frame)
        .mark_point(opacity=0, size=80)
        .encode(
            x=alt.X("trade_index:Q", scale=x_scale, axis=None),
            y=alt.Y(f"{y_field}:Q", scale=alt.Scale(zero=False), axis=None),
            tooltip=tooltip,
        )
        .add_params(hover)
    )
    crosshair = (
        alt.Chart(chart_frame)
        .mark_rule(color="#98a0aa", strokeWidth=1)
        .encode(x=alt.X("trade_index:Q", scale=x_scale, axis=None))
        .transform_filter(hover)
    )

    main = (
        alt.layer(
            band_layer, line, rule, rule_label, slot_layer, selector, crosshair
        )
        .resolve_scale(color="independent")
        .properties(height=height)
    )

    overview = (
        alt.Chart(chart_frame)
        .mark_line(color=COLOR_NEUTRAL, strokeWidth=1)
        .encode(
            x=alt.X("trade_index:Q", title=None, axis=None),
            # 미니맵은 반드시 본 차트와 같은 지표를 그린다 — premium_pct는
            # ratio의 감소 변환이라 ratio 모드에서 거울상이 된다
            y=alt.Y(
                f"{y_field}:Q", title=None, axis=None, scale=alt.Scale(zero=False)
            ),
        )
        .add_params(brush)
        .properties(height=60)
    )
    return (
        alt.vconcat(main, overview, spacing=4)
        .resolve_scale(x="independent", y="independent")
        .configure_view(stroke=None)
        .configure_axis(
            domain=False,
            gridColor="#edf0f3",
            gridOpacity=1,
            labelColor="#68717d",
            tickColor="#cfd5dc",
        )
    )
