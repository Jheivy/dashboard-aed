"""Resumen: KPIs, principales problemas, composición, perfil de columnas y vista de los datos."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eda.config import pal, style, tok
from eda.profiling import column_profile, dataset_summary, quality_alerts, quality_scores

from ._components import alert_list, card, kpi_row, page_header, severity_pill
from ._ui import Ctx, chart, es_int, es_pct, num_format_config

TYPE_ORDER = ["numérica", "categórica", "booleana", "fecha", "texto", "identificador", "constante", "ignorar"]


def render(ctx: Ctx) -> None:
    df = ctx.df
    page_header("Datos", "Resumen del dataset",
                "Tamaño, calidad general, problemas prioritarios y perfil de cada variable de un vistazo.")
    summ = dataset_summary(df, ctx.key)
    alerts = quality_alerts(df, ctx.key, ctx.types, tuple(ctx.prep_log))
    scores = quality_scores(summ, alerts, df.columns)

    kpi_row([
        dict(label="Filas", value=es_int(summ["rows"])),
        dict(label="Columnas", value=str(summ["cols"])),
        dict(label="Celdas vacías", value=es_pct(summ["null_pct"]),
             help="Porcentaje de celdas sin valor en todo el dataset."),
        dict(label="Filas completas", value=es_pct(summ["complete_rows_pct"]),
             help="Filas sin ningún valor faltante."),
        dict(label="Filas duplicadas", value=es_int(summ["dup_rows"])),
        dict(label="Puntuación de calidad", value=es_pct(sum(scores.values()) / len(scores) * 100, 0),
             help="Media de completitud, unicidad de filas y columnas sin problemas graves."),
    ])

    left, right = st.columns([3, 2], gap="medium")
    with left, card("Problemas prioritarios", "Lo primero que conviene revisar"):
        if alerts.empty:
            st.success("No se han detectado problemas de calidad relevantes.", icon=":material/verified:")
        else:
            counts = alerts["gravedad"].value_counts()
            st.markdown(" ".join(severity_pill(s, int(counts.get(s, 0))) for s in ["Alta", "Media", "Baja"]),
                        unsafe_allow_html=True)
            top = alerts[alerts["gravedad"].isin(["Alta", "Media"])].head(6)
            if top.empty:
                top = alerts.head(5)
            alert_list(top.itertuples())
            if len(alerts) > len(top):
                st.caption(f"…y {len(alerts) - len(top)} más en **Calidad de datos**.")
    with right, card("Composición", "Columnas por tipo detectado"):
        tc = pd.Series(ctx.types).value_counts().reindex(TYPE_ORDER).dropna()
        colors = pal()
        fig = go.Figure(go.Bar(x=tc.values, y=tc.index, orientation="h", text=tc.values.astype(int),
                               textposition="outside", marker_color=colors[0],
                               hovertemplate="%{y}: %{x} columnas<extra></extra>"))
        style(fig, 60 + 34 * len(tc), legend=False)
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_xaxes(visible=False)
        chart(fig, "ov_types")
        st.markdown(f"<span style='color:{tok()['muted']};font-size:.82rem'>Memoria en uso: "
                    f"<b>{summ['mem_mb']:,.1f} MB</b></span>".replace(".", ","), unsafe_allow_html=True)

    with card("Perfil de columnas", "Estadísticos y mini-distribución de cada variable"):
        prof = column_profile(df, ctx.key, ctx.types)
        f1, f2 = st.columns([3, 2])
        types_avail = [t for t in TYPE_ORDER if t in set(prof["tipo"])]
        tsel = f1.pills("Tipos", types_avail, selection_mode="multi", default=types_avail,
                        key="ov_type_filter", label_visibility="collapsed")
        q = f2.text_input("Buscar columna", key="ov_search", placeholder="Buscar columna…",
                          label_visibility="collapsed")
        shown = prof[prof["tipo"].isin(tsel or types_avail)]
        if q:
            shown = shown[shown["columna"].str.contains(q, case=False, regex=False)]
        cfg = num_format_config(shown)
        cfg.update({
            "% nulos": st.column_config.ProgressColumn("% nulos", format="%.1f%%", min_value=0, max_value=100),
            "% únicos": st.column_config.NumberColumn(format="%.1f%%"),
            "% moda": st.column_config.NumberColumn(format="%.1f%%"),
            "% ceros": st.column_config.NumberColumn(format="%.1f%%"),
            "distribución": st.column_config.BarChartColumn(
                "distribución", help="Histograma (numéricas y fechas) o frecuencia de las 15 categorías "
                                     "más comunes."),
        })
        st.dataframe(shown, hide_index=True, column_config=cfg, height=min(36 * (len(shown) + 1) + 4, 520))

    with card("Vista de los datos"):
        c1, c2, c3 = st.columns([2, 1, 3])
        mode = c1.segmented_control("Filas", ["Primeras", "Últimas", "Aleatorias"], default="Primeras",
                                    key="ov_mode", label_visibility="collapsed") or "Primeras"
        n = c2.number_input("Cuántas", 5, 1000, 25, 5, key="ov_n", label_visibility="collapsed")
        sel_cols = c3.multiselect("Columnas", df.columns.tolist(), key="ov_cols",
                                  placeholder="Todas las columnas", label_visibility="collapsed")
        data = df[sel_cols] if sel_cols else df
        data = (data.head(n) if mode == "Primeras" else data.tail(n) if mode == "Últimas"
                else data.sample(min(n, len(data))))
        st.dataframe(data, height=380)
        with st.expander("Nombres de columna (copiables)", icon=":material/content_copy:"):
            st.code(repr(df.columns.tolist()), language="python")
