"""Análisis univariante: una variable en detalle, o todas de un vistazo."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from eda.charts import bin_edges, box_traces, hist_bar
from eda.config import SCALES, pal, style, tok
from eda.stats_utils import (apply_scale, compare_groups, describe_numeric, effect_label, fmt_p,
                             normality_tests, skew_text)

from ._components import card, kpi_row, page_header
from ._ui import (NONE, Ctx, chart, es_int, es_num, es_pct, finite, insight, need, sub_nav,
                  top_with_other, verdict)


def render(ctx: Ctx) -> None:
    page_header("Exploración", "Análisis univariante",
                "Distribución de cada variable: forma, tendencia central, dispersión, normalidad y "
                "frecuencias. Opcionalmente, compárala entre grupos.")
    part = sub_nav(["Variable numérica", "Variable categórica", "Todas las numéricas",
                    "Todas las categóricas"], key="u_nav")
    {"Variable numérica": _numeric, "Variable categórica": _categorical,
     "Todas las numéricas": _all_numeric, "Todas las categóricas": _all_categorical}[part](ctx)


def _numeric(ctx: Ctx) -> None:
    if not need(bool(ctx.numeric), "No hay columnas numéricas."):
        return
    df = ctx.df
    c = st.columns([2, 2, 2, 1])
    var = c[0].selectbox("Variable", ctx.numeric, key="u_var")
    group = c[1].selectbox("Comparar por grupo", [NONE] + ctx.categorical, key="u_group")
    scale = c[2].segmented_control("Escala", SCALES, default="lineal", key="u_scale") or "lineal"
    bins = c[3].number_input("Intervalos", 10, 150, 50, 5, key="u_bins")

    raw = finite(df[var])
    x = apply_scale(raw, scale)
    if x.empty:
        st.warning("No quedan valores válidos con esta escala (log/raíz no admiten ceros o negativos).")
        return
    d = describe_numeric(df[var])
    kpi_row([dict(label="Media", value=es_num(d["media"])), dict(label="Mediana", value=es_num(d["mediana"])),
             dict(label="Desv. típica", value=es_num(d["desv. típica"])),
             dict(label="Asimetría", value=es_num(d["asimetría"], 3)),
             dict(label="Nulos", value=es_int(d["nulos"])), dict(label="Valores únicos", value=es_int(d["únicos"]))])

    xlabel = var + ("" if scale == "lineal" else f" ({scale})")
    colors = pal()
    groups = None
    with card("Distribución", (f"{es_int(len(raw) - len(x))} valores fuera del dominio de la escala {scale} "
                               "excluidos") if len(x) < len(raw) else None):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.26, 0.74], vertical_spacing=0.04)
        if group == NONE:
            for tr in box_traces(x, var, colors[0]):
                fig.add_trace(tr, 1, 1)
            fig.add_trace(hist_bar(x, int(bins), var, colors[0]), 2, 1)
            for val, lab, dash, pos in [(x.mean(), "media", "dash", "top left"),
                                        (x.median(), "mediana", "dot", "top right")]:
                fig.add_vline(x=val, line_dash=dash, line_width=1.5, line_color=tok()["muted"], row=2, col=1,
                              annotation_text=lab, annotation_font_size=11, annotation_position=pos)
            fig.update_yaxes(showticklabels=False, row=1, col=1)
        else:
            g = top_with_other(df[group], 6)
            cats = [k for k in g.value_counts().index if k != "Otros"]
            groups = [(str(k), apply_scale(finite(df.loc[g == k, var]), scale)) for k in cats]
            edges = bin_edges([v.to_numpy() for _, v in groups], int(bins))
            for i, (nm, v) in enumerate(groups):
                col = colors[i % len(colors)]
                for tr in box_traces(v, nm, col):
                    fig.add_trace(tr, 1, 1)
                fig.add_trace(hist_bar(v, edges, nm, col, density=True, opacity=0.6, showlegend=True), 2, 1)
            fig.update_layout(barmode="overlay")
        style(fig, 440 if group == NONE else 480)
        fig.update_xaxes(title=xlabel, row=2, col=1)
        fig.update_yaxes(title="densidad" if group != NONE else "filas", row=2, col=1)
        chart(fig, "u_main")
        insight(skew_text(d["asimetría"]))

    a, b = st.columns([3, 2], gap="medium")
    with a, card("Forma de la distribución"):
        extra = st.segmented_control("Gráfico", ["Q-Q normal", "ECDF", "Violín"], default="Q-Q normal",
                                     key="u_extra", label_visibility="collapsed") or "Q-Q normal"
        xs = x.sample(5000, random_state=0) if len(x) > 5000 else x
        f2 = go.Figure()
        if extra == "Q-Q normal":
            (osm, osr), (slope, inter, _) = stats.probplot(xs, dist="norm")
            f2.add_scattergl(x=osm, y=osr, mode="markers", name="observado",
                             marker=dict(size=5, color=colors[0], opacity=0.6))
            f2.add_scatter(x=osm, y=slope * osm + inter, mode="lines", name="normal teórica",
                           line=dict(color=tok()["muted"], dash="dash", width=2))
            f2.update_xaxes(title="cuantiles teóricos")
            f2.update_yaxes(title="cuantiles observados")
            cap = "Si los puntos siguen la línea, la variable es aproximadamente normal."
        elif extra == "ECDF":
            v = np.sort(xs.to_numpy())
            f2.add_scatter(x=v, y=np.arange(1, len(v) + 1) / len(v), mode="lines", line_shape="hv",
                           line=dict(color=colors[0], width=2), name="ECDF",
                           hovertemplate="≤ %{x:,.4g}: %{y:.1%}<extra></extra>")
            f2.update_yaxes(title="proporción acumulada", tickformat=".0%")
            f2.update_xaxes(title=xlabel)
            cap = "Para cada valor, qué proporción de observaciones queda por debajo."
        else:
            f2.add_violin(y=xs, box_visible=True, meanline_visible=True, line_color=colors[0],
                          fillcolor=colors[0], opacity=0.5, name=var, points=False)
            cap = "Forma completa de la distribución con su caja y media."
        style(f2, 320, legend=extra == "Q-Q normal")
        chart(f2, "u_extra_fig")
        st.caption(cap)
    with b, card("Estadísticos"):
        tbl = pd.DataFrame({"valor": d}).drop(index=["n"])
        st.dataframe(tbl, column_config={"valor": st.column_config.NumberColumn(format="%.4g")}, height=360)

    tests = normality_tests(x)
    if tests:
        with card("Normalidad", "H₀: la variable sigue una distribución normal"
                                + ("" if scale == "lineal" else f" (en escala {scale})")):
            tdf = pd.DataFrame(tests)
            tdf["conclusión"] = np.where(tdf["p"] < 0.05, "No normal", "Compatible con normal")
            tdf["p"] = tdf["p"].map(fmt_p)
            st.dataframe(tdf, hide_index=True, column_config={"estadístico": st.column_config.NumberColumn(format="%.3f")})
            if len(x) > 5000:
                st.caption("Con muchas observaciones los tests rechazan la normalidad ante desviaciones mínimas: "
                           "guíate más por el Q-Q y la asimetría.")

    if groups:
        res = compare_groups([v.to_numpy() for _, v in groups], [n for n, _ in groups])
        if res:
            with card("Comparación entre grupos"):
                eff = f"{res['effect_name']} = {res['effect']:.3f} ({effect_label(res['effect_name'], res['effect'])})"
                verdict(res["p"], f"{res['test']}: hay diferencias entre grupos",
                        f"{res['test']}: sin diferencias significativas", extra=eff)
                summary = pd.DataFrame([{"grupo": n, "n": len(v), "media": v.mean(), "mediana": v.median(),
                                         "desv. típica": v.std()} for n, v in groups])
                st.dataframe(summary, hide_index=True, column_config={
                    c: st.column_config.NumberColumn(format="%.4g") for c in ["media", "mediana", "desv. típica"]})
                st.caption(f"Contraste paramétrico de referencia: {res['test_param']} p = {fmt_p(res['p_param'])}. "
                           "Se prioriza el no paramétrico porque no asume normalidad.")


def _categorical(ctx: Ctx) -> None:
    df = ctx.df
    cands = ctx.categorical + [c for c in ctx.text if df[c].nunique() < 5000]
    if not need(bool(cands), "No hay columnas categóricas."):
        return
    c = st.columns([2, 2, 1, 1])
    var = c[0].selectbox("Variable", cands, key="u_cat")
    topn = c[1].slider("Mostrar las N más frecuentes", 3, 50, 15, key="u_cat_top")
    as_pct = c[2].toggle("En %", key="u_cat_pct")
    incl_na = c[3].toggle("Incluir nulos", True, key="u_cat_na")
    s = df[var].astype(object)
    if incl_na:
        s = s.where(s.notna(), "(nulo)")
    vc = s.value_counts()
    n = vc.sum()
    p = vc / n
    ent = float(-(p * np.log2(p)).sum())
    kpi_row([dict(label="Categorías", value=es_int(len(vc))),
             dict(label="Más frecuente", value=str(vc.index[0])[:18] if len(vc) else "—"),
             dict(label="% de la moda", value=es_pct(vc.iloc[0] / n * 100) if len(vc) else "—"),
             dict(label="Equilibrio", value=es_pct(ent / np.log2(len(vc)) * 100, 0) if len(vc) > 1 else "—",
                  help="Entropía normalizada: 100% = todas igual de frecuentes; ~0% = una domina."),
             dict(label="Categorías raras (< 1%)", value=str(int((p < 0.01).sum())))])
    top = vc.head(topn)
    rest = vc.iloc[topn:].sum()
    labels = [str(i) for i in top.index] + ([f"Otras ({len(vc) - topn})"] if rest else [])
    vals = np.append(top.values, rest) if rest else top.values
    shown = vals / n * 100 if as_pct else vals
    a, b = st.columns([3, 2], gap="medium")
    with a, card(f"Frecuencias · {var}"):
        colors = [pal()[0]] * len(top) + ([tok()["subtle"]] if rest else [])
        fig = go.Figure(go.Bar(x=shown, y=labels, orientation="h", marker_color=colors,
                               text=[es_pct(v) if as_pct else es_int(v) for v in shown], textposition="outside",
                               hovertemplate="%{y}: %{x:,.1f}<extra></extra>"))
        style(fig, max(260, 26 * len(labels) + 50), legend=False)
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_xaxes(title="%" if as_pct else "filas")
        chart(fig, "u_cat_bar")
    with b, card("Curva de Pareto", "¿Cuántas categorías concentran la mayoría de filas?"):
        cum = (vc.head(50).cumsum() / n * 100)
        fig = go.Figure(go.Scatter(x=np.arange(1, len(cum) + 1), y=cum.values, mode="lines+markers",
                                   line=dict(color=pal()[1], width=2), marker=dict(size=7),
                                   customdata=cum.index.astype(str),
                                   hovertemplate="%{customdata}<br>acumulado %{y:.1f}%<extra></extra>"))
        fig.add_hline(y=80, line_dash="dot", line_color=tok()["subtle"], annotation_text="80%")
        style(fig, 280, legend=False)
        fig.update_yaxes(range=[0, 105], title="% acumulado")
        fig.update_xaxes(title="nº de categorías")
        chart(fig, "u_cat_pareto")
        st.caption(f"**{int((vc.cumsum() / n < 0.8).sum() + 1)}** de {len(vc)} categorías acumulan el 80% de las filas.")
    with st.expander("Tabla de frecuencias", icon=":material/table:"):
        st.dataframe(pd.DataFrame({"categoría": vc.index.astype(str), "n": vc.values, "%": p.values * 100,
                                   "% acumulado": vc.cumsum().values / n * 100}), hide_index=True,
                     column_config={"%": st.column_config.NumberColumn(format="%.2f"),
                                    "% acumulado": st.column_config.NumberColumn(format="%.1f")})


def _all_numeric(ctx: Ctx) -> None:
    if not need(bool(ctx.numeric), "No hay columnas numéricas."):
        return
    df = ctx.df
    c1, c2, c3 = st.columns([4, 2, 1])
    cols = c1.multiselect("Variables", ctx.numeric, default=ctx.numeric[:24], key="u_all_cols", max_selections=48)
    scale = c2.segmented_control("Escala", SCALES, default="lineal", key="u_all_scale") or "lineal"
    ncols = c3.number_input("Por fila", 2, 6, 4, key="u_all_grid")
    if not cols:
        return
    nrows = int(np.ceil(len(cols) / ncols))
    colors = pal()
    with card("Distribuciones", "Histogramas pre-agregados (rápidos con millones de filas)"):
        fig = make_subplots(rows=nrows, cols=int(ncols), subplot_titles=cols,
                            vertical_spacing=min(0.12, 0.5 / max(nrows, 1)), horizontal_spacing=0.05)
        for i, c in enumerate(cols):
            fig.add_trace(hist_bar(apply_scale(finite(df[c]), scale), 40, c, colors[0]),
                          i // int(ncols) + 1, i % int(ncols) + 1)
        style(fig, 170 * nrows + 40, legend=False)
        fig.update_annotations(font_size=11)
        chart(fig, "u_all_grid_fig")
    rows = []
    for c in cols:
        d = describe_numeric(df[c])
        if d:
            rows.append({"variable": c, **{k: d[k] for k in ["n", "nulos", "media", "mediana", "desv. típica",
                                                             "mín", "máx", "asimetría", "curtosis (exceso)",
                                                             "% ceros"]}})
    tbl = pd.DataFrame(rows)
    tbl["forma"] = tbl["asimetría"].map(lambda v: "" if pd.isna(v) else "simétrica" if abs(v) < 0.5
                                        else "asim. moderada" if abs(v) < 1 else "muy asimétrica")
    with card("Resumen"):
        st.dataframe(tbl, hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.4g") for c in tbl.columns
            if c not in ("variable", "forma", "n", "nulos")})


def _all_categorical(ctx: Ctx) -> None:
    if not need(bool(ctx.categorical), "No hay columnas categóricas."):
        return
    df = ctx.df
    c1, c2 = st.columns([4, 1])
    cols = c1.multiselect("Variables", ctx.categorical, default=ctx.categorical[:12], key="u_allc_cols")
    topn = c2.number_input("Top N", 3, 20, 8, key="u_allc_top")
    if not cols:
        return
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    with card("Frecuencias"):
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols,
                            vertical_spacing=min(0.12, 0.45 / max(nrows, 1)), horizontal_spacing=0.14)
        for i, c in enumerate(cols):
            vc = df[c].astype(object).where(df[c].notna(), "(nulo)").astype(str).value_counts().head(int(topn))[::-1]
            fig.add_trace(go.Bar(x=vc.values, y=[v[:22] for v in vc.index], orientation="h",
                                 marker_color=pal()[0], showlegend=False,
                                 hovertemplate="%{y}: %{x:,}<extra>" + c + "</extra>"),
                          i // ncols + 1, i % ncols + 1)
        style(fig, (22 * int(topn) + 70) * nrows + 30, legend=False)
        fig.update_annotations(font_size=11)
        chart(fig, "u_allc_fig")
    card_df = pd.DataFrame({
        "variable": cols, "categorías": [df[c].nunique() for c in cols],
        "% nulos": [df[c].isna().mean() * 100 for c in cols],
        "moda": [str(df[c].mode().iloc[0]) if df[c].notna().any() else None for c in cols],
        "% moda": [df[c].value_counts(normalize=True).iloc[0] * 100 if df[c].notna().any() else np.nan
                   for c in cols]})
    with card("Resumen"):
        st.dataframe(card_df, hide_index=True, column_config={
            "% nulos": st.column_config.NumberColumn(format="%.1f"), "% moda": st.column_config.NumberColumn(format="%.1f")})
