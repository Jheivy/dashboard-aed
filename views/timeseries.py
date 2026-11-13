"""Series temporales: evolución, estacionalidad y huecos en el registro."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eda.config import pal, style, tok

from ._components import card, kpi_row, page_header
from ._ui import NONE, Ctx, chart, es_int, need, top_with_other

FREQS = {"Hora": "h", "Día": "D", "Semana": "W", "Mes": "MS", "Trimestre": "QS", "Año": "YS"}
AGGS = {"nº de filas": "size", "suma": "sum", "media": "mean", "mediana": "median",
        "mínimo": "min", "máximo": "max", "% de nulos": "nulls"}
DAYS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def _suggest_freq(span: pd.Timedelta) -> str:
    days = span.days
    return ("Hora" if days <= 3 else "Día" if days <= 120 else "Semana" if days <= 730
            else "Mes" if days <= 3650 else "Año")


@st.cache_data(show_spinner=False, max_entries=32)
def _compute(_df: pd.DataFrame, key: str, tcol: str, freq: str, func: str, metric: str | None,
             group: str | None) -> dict:
    """Resampled series (per group), period counts and seasonal profiles — cached per selection."""
    d = _df[_df[tcol].notna()]

    def agg(frame: pd.DataFrame) -> pd.Series:
        r = frame.set_index(tcol).resample(freq)
        if func == "size":
            return r.size()
        if func == "nulls":
            return r[metric].apply(lambda s: s.isna().mean() * 100 if len(s) else np.nan)
        return getattr(r[metric], func)()

    if group is None:
        series = {None: agg(d)}
    else:
        g = top_with_other(d[group], 5)
        series = {k: agg(d[g == k]) for k in g.value_counts().index}
    counts = d.set_index(tcol).resample(freq).size()
    has_time = bool((d[tcol].dt.hour != 0).any() or (d[tcol].dt.minute != 0).any())
    keys = {"Mes del año": d[tcol].dt.month, "Día de la semana": d[tcol].dt.dayofweek}
    if has_time:
        keys["Hora del día"] = d[tcol].dt.hour
    seasonal = {}
    for title, ks in keys.items():
        if func == "size":
            seasonal[title] = d.groupby(ks).size()
        elif func == "nulls":
            seasonal[title] = d.groupby(ks)[metric].apply(lambda v: v.isna().mean() * 100)
        else:
            seasonal[title] = d.groupby(ks)[metric].agg(func)
    return dict(series=series, counts=counts, seasonal=seasonal, first=d[tcol].min(), last=d[tcol].max())


def render(ctx: Ctx) -> None:
    page_header("Modelado", "Series temporales",
                "Evolución de una métrica en el tiempo, patrones estacionales (hora, día de la semana, mes) y "
                "periodos sin datos.")
    if not need(bool(ctx.datetime), "No se ha detectado ninguna columna de fecha. Si existe, márcala como «fecha» "
                                    "en el panel lateral → Tipos de columna (también sirve una columna de año)."):
        return
    df = ctx.df
    c = st.columns([2, 1, 2, 2, 2, 1])
    tcol = c[0].selectbox("Columna de fecha", ctx.datetime, key="ts_col")
    tmin, tmax = df[tcol].min(), df[tcol].max()
    if pd.isna(tmin):
        st.warning("Sin fechas válidas.")
        return
    flist = list(FREQS)
    freq_label = c[1].selectbox("Frecuencia", flist, index=flist.index(_suggest_freq(tmax - tmin)),
                                key=f"ts_freq_{tcol}")
    agg_label = c[2].selectbox("Métrica", list(AGGS), key="ts_agg")
    metric = None
    if agg_label != "nº de filas":
        pool = ctx.numeric if agg_label != "% de nulos" else df.columns.tolist()
        if not pool:
            st.caption("No hay variables numéricas.")
            return
        metric = c[3].selectbox("Variable", pool, key="ts_metric")
    group = c[4].selectbox("Separar por", [NONE] + ctx.categorical, key="ts_group")
    roll = c[5].number_input("Media móvil", 1, 60, 1, key="ts_roll", help="Periodos; 1 = sin suavizado.")
    freq, func = FREQS[freq_label], AGGS[agg_label]
    res = _compute(df, ctx.key, tcol, freq, func, metric, None if group == NONE else group)
    counts = res["counts"]
    empty = counts[counts == 0]
    kpi_row([dict(label="Desde", value=f"{res['first']:%d/%m/%Y}"), dict(label="Hasta", value=f"{res['last']:%d/%m/%Y}"),
             dict(label="Periodos", value=es_int(len(counts))),
             dict(label="Periodos sin datos", value=es_int(len(empty)),
                  help="Huecos en el registro: pueden ser reales o fallos de recogida.")])
    ylab = agg_label + (f" de {metric}" if metric else "")
    colors = pal()
    with card(f"{ylab} por {freq_label.lower()}"):
        fig = go.Figure()
        if group == NONE:
            s = res["series"][None]
            fig.add_scatter(x=s.index, y=s.values, mode="lines" + ("+markers" if len(s) < 120 else ""), name=ylab,
                            line=dict(color=colors[0], width=2 if roll == 1 else 1),
                            opacity=1 if roll == 1 else 0.45, hovertemplate="%{x|%d/%m/%Y}: %{y:,.4g}<extra></extra>")
            if roll > 1:
                fig.add_scatter(x=s.index, y=s.rolling(int(roll), min_periods=1).mean().values, mode="lines",
                                name=f"media móvil ({roll})", line=dict(color=colors[0], width=2.5))
        else:
            for i, (k, s) in enumerate(res["series"].items()):
                if roll > 1:
                    s = s.rolling(int(roll), min_periods=1).mean()
                fig.add_scatter(x=s.index, y=s.values, mode="lines", name=str(k),
                                line=dict(color=tok()["subtle"] if k == "Otros" else colors[i], width=2),
                                hovertemplate="%{x|%d/%m/%Y}: %{y:,.4g}<extra>" + str(k) + "</extra>")
        style(fig, 380, legend=group != NONE or roll > 1)
        fig.update_yaxes(title=ylab)
        fig.update_layout(hovermode="x unified")
        chart(fig, "ts_main")
        if df[tcol].isna().any():
            st.caption(f"{es_int(df[tcol].isna().sum())} filas sin fecha excluidas.")
    if len(empty):
        runs = (empty.index.to_series().diff() != pd.tseries.frequencies.to_offset(freq)).cumsum()
        gaps = empty.index.to_series().groupby(runs).agg(["min", "max", "count"])
        gaps.columns = ["desde", "hasta", "periodos"]
        with st.expander(f"Huecos sin datos ({len(gaps)} tramos)", icon=":material/event_busy:"):
            st.dataframe(gaps.sort_values("periodos", ascending=False), hide_index=True)

    labels = {"Mes del año": (MONTHS, 1), "Día de la semana": (DAYS, 0), "Hora del día": (None, 0)}
    st.markdown(f"##### Patrones estacionales · {ylab}")
    cols = st.columns(len(res["seasonal"]), gap="medium")
    for (title, s), col in zip(res["seasonal"].items(), cols):
        names, offset = labels[title]
        x = [names[int(i) - offset] for i in s.index] if names else s.index.astype(int).astype(str)
        with col, card(title):
            fig = go.Figure(go.Bar(x=x, y=s.values, marker_color=colors[0],
                                   hovertemplate="%{x}: %{y:,.4g}<extra></extra>"))
            style(fig, 220, legend=False)
            fig.update_xaxes(type="category")
            chart(fig, f"ts_season_{title}")
    st.caption("Si el periodo no cubre años completos, algunos meses tendrán menos datos.")
