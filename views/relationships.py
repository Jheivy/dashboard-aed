"""Relaciones entre variables: correlación, asociación mixta, par de variables, pairplot, estratificada."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from eda.charts import box_traces, sample_for_plot
from eda.config import SCALES, div_scale, pal, seq_scale, style, tok
from eda.stats_utils import (apply_scales, association_matrix, cluster_order, compare_groups, corr_matrix, corr_pvalues,
                             cramers_v, effect_label, fmt_p, pearson_ci)

from ._components import card, kpi_row, page_header
from ._ui import (NONE, Ctx, chart, es_int, insight, need, pick_index, sample_rows, sub_nav, top_with_other,
                  verdict)

METHODS = {"pearson": "Pearson", "spearman": "Spearman", "kendall": "Kendall"}


def render(ctx: Ctx) -> None:
    page_header("Exploración", "Relaciones entre variables",
                "Correlaciones, asociaciones entre tipos distintos, análisis detallado de un par de variables "
                "y detección de variables de confusión.")
    part = sub_nav(["Correlación", "Asociación mixta", "Dos variables", "Pairplot", "Correlación estratificada"],
                   key="r_nav")
    {"Correlación": _correlation, "Asociación mixta": _mixed, "Dos variables": _pair, "Pairplot": _pairplot,
     "Correlación estratificada": _stratified}[part](ctx)


def _usable_numeric(ctx: Ctx) -> list[str]:
    return [c for c in ctx.numeric if ctx.df[c].nunique(dropna=True) > 1]


def _heatmap(m: pd.DataFrame, diverging: bool, text: bool) -> go.Figure:
    masked = m.mask(np.triu(np.ones(m.shape, dtype=bool), k=1))
    fig = go.Figure(go.Heatmap(
        z=masked.values, x=m.columns, y=m.index, zmin=-1 if diverging else 0, zmax=1,
        colorscale=div_scale() if diverging else seq_scale(), text=np.round(masked.values, 2),
        texttemplate="%{text}" if text else None, hoverongaps=False,
        hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>", colorbar=dict(thickness=10)))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-40)
    return fig


def _correlation(ctx: Ctx) -> None:
    cols_all = _usable_numeric(ctx)
    if not need(len(cols_all) >= 2, "Hacen falta al menos 2 columnas numéricas no constantes."):
        return
    c1, c2, c3 = st.columns([4, 2, 1])
    cols = c1.multiselect("Variables", cols_all, default=cols_all[:30], key="r_corr_vars")
    method = c2.segmented_control("Método", list(METHODS), format_func=METHODS.get, default="spearman",
                                  key="r_corr_m", help="Pearson: relación lineal. Spearman/Kendall: monótona por "
                                                       "rangos, robusta a outliers y asimetría.") or "spearman"
    sig = c3.toggle("Solo significativas", key="r_corr_sig", help="Oculta las celdas con p ≥ 0,05.")
    if len(cols) < 2:
        return
    data, dkey = ctx.df[cols], ctx.key
    if method == "kendall" and len(data) > 20000:
        data, dkey = data.sample(20000, random_state=0), ctx.key + ":s20k"
        st.caption("Kendall calculado sobre una muestra de 20.000 filas.")
    corr = corr_matrix(data, dkey, tuple(cols), method)
    order = cluster_order(corr)
    corr = corr.iloc[order, order]
    show = corr
    if sig:
        smp = sample_rows(data, 20000)
        pv = corr_pvalues(smp, dkey + str(tuple(cols)) + method, method).loc[corr.index, corr.columns]
        show = corr.mask(pv >= 0.05)
    with card(f"Matriz de correlación · {METHODS[method]}", "Ordenada por clusters: las variables relacionadas "
                                                            "quedan juntas"):
        fig = _heatmap(show, True, len(cols) <= 18)
        style(fig, max(420, 26 * len(cols) + 130))
        chart(fig, "r_corr_heat")

    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    pairs = upper.reindex(upper.abs().sort_values(ascending=False).index).head(15).reset_index()
    pairs.columns = ["variable A", "variable B", "r"]
    pairs["fuerza"] = pairs["r"].map(lambda r: effect_label("r", r))
    with card("Pares más correlacionados"):
        st.dataframe(pairs, hide_index=True, column_config={
            "r": st.column_config.ProgressColumn(format="%.3f", min_value=-1, max_value=1)})
    pear = corr_matrix(ctx.df, ctx.key, tuple(cols), "pearson")
    spear = corr_matrix(ctx.df, ctx.key, tuple(cols), "spearman")
    gap = (spear.abs() - pear.abs()).where(np.triu(np.ones(pear.shape, dtype=bool), k=1)).stack()
    big = gap[gap.abs() > 0.2]
    if len(big):
        txt = ", ".join(f"{a}–{b}" for a, b in big.abs().sort_values(ascending=False).index[:5])
        insight(f"Pearson y Spearman difieren mucho en **{txt}**: suele indicar relación no lineal o influencia "
                "de outliers. Revísalo en «Dos variables».", "warn")


def _mixed(ctx: Ctx) -> None:
    num = _usable_numeric(ctx)
    cat = [c for c in ctx.categorical if 1 < ctx.df[c].nunique() <= 50 and c not in num]
    if not need(len(num) + len(cat) >= 2, "Hacen falta al menos 2 variables."):
        return
    c1, c2 = st.columns(2)
    sel_n = c1.multiselect("Numéricas", num, default=num[:12], key="r_mx_num")
    sel_c = c2.multiselect("Categóricas (≤ 50 categorías)", cat, default=cat[:8], key="r_mx_cat")
    if len(sel_n) + len(sel_c) < 2:
        return
    data = sample_rows(ctx.df, 50000)
    m = association_matrix(data, ctx.key + ":mx", sel_n, sel_c)
    order = cluster_order(m)
    m = m.iloc[order, order]
    with card("Matriz de asociación mixta", "0–1 para cualquier tipo: |Spearman| entre numéricas, η numérica–"
                                            "categórica y V de Cramér entre categóricas. Sin dirección"):
        fig = _heatmap(m, False, len(m) <= 18)
        style(fig, max(420, 26 * len(m) + 130))
        chart(fig, "r_mx_heat")
        if len(ctx.df) > 50000:
            st.caption("Calculado sobre una muestra de 50.000 filas.")


def _pair(ctx: Ctx) -> None:
    df = ctx.df
    allv = _usable_numeric(ctx) + [c for c in ctx.categorical if c not in ctx.numeric]
    if not need(len(allv) >= 2, "Hacen falta al menos 2 variables."):
        return
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("Variable X", allv, key="r_px")
    y = c2.selectbox("Variable Y", [v for v in allv if v != x], key="r_py")
    color = c3.selectbox("Color", [NONE] + [c for c in ctx.categorical if c not in (x, y)], key="r_pc")
    isnum = lambda v: v in ctx.numeric and ctx.types.get(v) == "numérica"  # noqa: E731
    colors = pal()

    if isnum(x) and isnum(y):
        s1, s2, s3 = st.columns(3)
        sx = s1.selectbox(f"Escala de {x}", SCALES, key="r_psx")
        sy = s2.selectbox(f"Escala de {y}", SCALES, key="r_psy")
        trend = s3.segmented_control("Tendencia", ["Lineal", "LOWESS", "Ninguna"], default="Lineal",
                                     key="r_ptrend") or "Lineal"
        d = apply_scales(df, {x: sx, y: sy})
        if len(d) < 3:
            st.warning("Muy pocos pares válidos.")
            return
        r, lo, hi, p, n = pearson_ci(d[x], d[y])
        rho, _ = stats.spearmanr(d[x], d[y])
        kpi_row([dict(label="Pearson r", value=f"{r:.3f}", help=f"IC 95%: [{lo:.3f}, {hi:.3f}]"),
                 dict(label="Spearman ρ", value=f"{rho:.3f}"), dict(label="R² lineal", value=f"{r ** 2:.3f}"),
                 dict(label="Pares", value=es_int(n))])
        plot = sample_for_plot(d, 8000).copy()
        with card(f"{y} frente a {x}", f"Muestra de {es_int(len(plot))} de {es_int(len(d))} puntos"
                  if len(plot) < len(d) else None):
            fig = make_subplots(rows=2, cols=2, column_widths=[0.85, 0.15], row_heights=[0.15, 0.85],
                                shared_xaxes="columns", shared_yaxes="rows", horizontal_spacing=0.01,
                                vertical_spacing=0.01)
            if color != NONE:
                g = top_with_other(df.loc[plot.index, color], 3)
                cats = g.value_counts().index.tolist()
            else:
                g, cats = None, [None]
            for i, cat in enumerate(cats):
                sub = plot if cat is None else plot[g == cat]
                col = tok()["subtle"] if cat == "Otros" else colors[i]  # 'Otros' is neutral, never a hue
                name = "datos" if cat is None else str(cat)
                fig.add_trace(go.Scattergl(x=sub[x], y=sub[y], mode="markers", name=name, legendgroup=name,
                                           showlegend=cat is not None,
                                           marker=dict(size=6, color=col, opacity=0.45),
                                           hovertemplate=f"{x}: %{{x:,.4g}}<br>{y}: %{{y:,.4g}}<extra>{name}</extra>"),
                              2, 1)
                fig.add_trace(go.Histogram(x=sub[x], marker_color=col, opacity=0.6, showlegend=False,
                                           legendgroup=name, nbinsx=40, hoverinfo="skip"), 1, 1)
                fig.add_trace(go.Histogram(y=sub[y], marker_color=col, opacity=0.6, showlegend=False,
                                           legendgroup=name, nbinsy=40, hoverinfo="skip"), 2, 2)
            xs = np.linspace(d[x].min(), d[x].max(), 60)
            if trend == "Lineal":
                m_, b_ = np.polyfit(d[x], d[y], 1)
                fig.add_trace(go.Scatter(x=xs, y=m_ * xs + b_, mode="lines", name="tendencia lineal",
                                         line=dict(color=tok()["text"], width=2, dash="dash")), 2, 1)
            elif trend == "LOWESS":
                from statsmodels.nonparametric.smoothers_lowess import lowess
                base = sample_rows(d, 5000)
                lw = lowess(base[y], base[x], frac=0.3, return_sorted=True)
                fig.add_trace(go.Scatter(x=lw[:, 0], y=lw[:, 1], mode="lines", name="LOWESS",
                                         line=dict(color=tok()["text"], width=2)), 2, 1)
            fig.update_layout(barmode="overlay")
            style(fig, 540)
            fig.update_xaxes(title=x + ("" if sx == "lineal" else f" ({sx})"), row=2, col=1)
            fig.update_yaxes(title=y + ("" if sy == "lineal" else f" ({sy})"), row=2, col=1)
            fig.update_xaxes(showticklabels=False, row=2, col=2)
            fig.update_yaxes(showticklabels=False, row=1, col=1)
            chart(fig, "r_pair_scatter")
            verdict(p, f"Correlación de Pearson significativa ({effect_label('r', r)})",
                    "Sin correlación lineal significativa", extra=f"IC 95% de r: [{lo:.3f}, {hi:.3f}]")
            st.caption("Correlación no implica causalidad. Si ρ y r difieren mucho, la relación es monótona pero "
                       "no lineal, o hay outliers influyentes.")
    elif isnum(x) or isnum(y):
        num, cat = (x, y) if isnum(x) else (y, x)
        maxg = st.slider("Máx. categorías", 2, 20, 8, key="r_pmaxg")
        g = top_with_other(df[cat], maxg)
        d = pd.DataFrame({cat: g, num: pd.to_numeric(df[num], errors="coerce")}).dropna()
        order = d.groupby(cat)[num].median().sort_values(ascending=False).index.tolist()
        groups = [(k, d.loc[d[cat] == k, num]) for k in order]
        res = compare_groups([v.to_numpy() for _, v in groups], [str(k) for k, _ in groups])
        with card(f"{num} por {cat}", "Ordenado por mediana; cajas calculadas sobre todas las filas"):
            fig = go.Figure()
            for i, (k, v) in enumerate(groups):
                for tr in box_traces(v, str(k), colors[0] if k != "Otros" else tok()["subtle"], horizontal=False,
                                     max_outliers=200):
                    fig.add_trace(tr)
            style(fig, 440, legend=False)
            fig.update_yaxes(title=num)
            chart(fig, "r_pair_numcat")
            if res:
                eff = f"{res['effect_name']} = {res['effect']:.3f} ({effect_label(res['effect_name'], res['effect'])})"
                verdict(res["p"], f"{res['test']}: {num} difiere entre categorías de {cat}",
                        f"{res['test']}: sin diferencias significativas", extra=eff)
        with st.expander("Estadísticos por categoría", icon=":material/table:"):
            summ = d.groupby(cat)[num].describe().reindex(order)
            st.dataframe(summ, column_config={c: st.column_config.NumberColumn(format="%.4g") for c in summ.columns})
    else:
        c1, c2 = st.columns([1, 2])
        maxg = c1.slider("Máx. categorías por variable", 2, 20, 8, key="r_pmaxg2")
        norm = c2.segmented_control("Mostrar", ["Recuentos", "% por fila", "% por columna", "% total"],
                                    default="% por fila", key="r_pnorm") or "% por fila"
        gx, gy = top_with_other(df[x], maxg), top_with_other(df[y], maxg)
        ct = pd.crosstab(gx, gy)
        show = {"Recuentos": ct, "% por fila": ct.div(ct.sum(1), axis=0) * 100,
                "% por columna": ct.div(ct.sum(0), axis=1) * 100, "% total": ct / ct.values.sum() * 100}[norm]
        chi = stats.chi2_contingency(ct)
        v, p, dof = cramers_v(gx, gy)
        a, b = st.columns(2, gap="medium")
        with a, card(f"{x} × {y}", norm):
            fig = go.Figure(go.Heatmap(z=show.values, x=show.columns.astype(str), y=show.index.astype(str),
                                       colorscale=seq_scale(), text=np.round(show.values, 1), texttemplate="%{text}",
                                       hovertemplate="%{y} · %{x}: %{z:,.1f}<extra></extra>",
                                       colorbar=dict(thickness=10)))
            style(fig, 400)
            chart(fig, "r_pair_ct")
        with b, card("Residuos estandarizados", "> +2: ocurre más de lo esperado si fueran independientes · "
                                                "< −2: menos"):
            resid = (ct - chi.expected_freq) / np.sqrt(chi.expected_freq)
            fig = go.Figure(go.Heatmap(z=resid.values, x=resid.columns.astype(str), y=resid.index.astype(str),
                                       zmin=-4, zmax=4, colorscale=div_scale(), text=np.round(resid.values, 1),
                                       texttemplate="%{text}", colorbar=dict(thickness=10),
                                       hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>"))
            style(fig, 400)
            chart(fig, "r_pair_resid")
        verdict(p, f"Chi-cuadrado: {x} y {y} están asociadas", "Chi-cuadrado: sin asociación significativa",
                extra=f"V de Cramér = {v:.3f} ({effect_label('V', v)}) · gl = {dof}")
        low = (chi.expected_freq < 5).mean() * 100
        if low > 20:
            insight(f"El {low:.0f}% de las celdas tiene frecuencia esperada < 5: el test χ² es poco fiable. "
                    "Agrupa categorías (baja «Máx. categorías»).", "warn")


def _pairplot(ctx: Ctx) -> None:
    num = _usable_numeric(ctx)
    if not need(len(num) >= 2, "Hacen falta al menos 2 columnas numéricas."):
        return
    c1, c2, c3 = st.columns([3, 1, 1])
    vars_ = c1.multiselect("Variables (máx. 6)", num, default=num[:4], max_selections=6, key="r_pp_vars")
    color = c2.selectbox("Color", [NONE] + ctx.categorical, key="r_pp_color")
    n = c3.number_input("Puntos", 200, 5000, 1500, 100, key="r_pp_n")
    if len(vars_) < 2:
        return
    scales = {v: sc.selectbox(v, SCALES, key=f"r_pp_s_{v}") for sc, v in zip(st.columns(len(vars_)), vars_)}
    smp = sample_rows(ctx.df, int(n))
    d = apply_scales(smp, scales)
    if d.empty:
        st.warning("Sin filas válidas con estas escalas.")
        return
    usable = [c for c in vars_ if d[c].nunique() > 1]
    if len(usable) >= 3:
        vars_ = [usable[i] for i in cluster_order(d[usable].corr())] + [c for c in vars_ if c not in usable]
    k = len(vars_)
    cser, cats = None, [None]
    if color != NONE:
        cser = top_with_other(smp.loc[d.index, color], 3)
        cats = cser.value_counts().index.tolist()
    colors = pal()
    fig = make_subplots(rows=k, cols=k, shared_xaxes=True, horizontal_spacing=0.015, vertical_spacing=0.015)
    for i, yi in enumerate(vars_):
        for j, xj in enumerate(vars_):
            if j > i:
                continue
            for ci, cat in enumerate(cats):
                idx = d.index if cat is None else d.index[(cser == cat).to_numpy()]
                name = "todas" if cat is None else str(cat)
                pcol = tok()["subtle"] if cat == "Otros" else colors[ci]
                if i == j:
                    fig.add_trace(go.Histogram(x=d.loc[idx, xj], marker_color=pcol, opacity=0.6, name=name,
                                               legendgroup=name, nbinsx=30,
                                               showlegend=(cat is not None and i == 0)), i + 1, j + 1)
                else:
                    fig.add_trace(go.Scattergl(x=d.loc[idx, xj], y=d.loc[idx, yi], mode="markers", name=name,
                                               legendgroup=name, showlegend=False,
                                               marker=dict(size=4, color=pcol, opacity=0.45)), i + 1, j + 1)
            if i != j and d[xj].std() > 0 and d[yi].std() > 0:
                r = np.corrcoef(d[xj], d[yi])[0, 1]
                fig.add_annotation(text=f"r = {r:.2f}".replace(".", ","), xref="x domain", yref="y domain",
                                   x=0.03, y=0.97, showarrow=False, font=dict(size=10), row=i + 1, col=j + 1)
    for i in range(k):
        for j in range(k):
            if j > i:
                fig.update_xaxes(visible=False, row=i + 1, col=j + 1)
                fig.update_yaxes(visible=False, row=i + 1, col=j + 1)
            else:
                fig.update_xaxes(tickfont=dict(size=9), row=i + 1, col=j + 1)
                fig.update_yaxes(tickfont=dict(size=9), row=i + 1, col=j + 1)
        fig.update_yaxes(title_text=vars_[i], title_font=dict(size=10), row=i + 1, col=1)
        fig.update_xaxes(title_text=vars_[i], title_font=dict(size=10), row=k, col=i + 1)
    fig.update_layout(barmode="overlay")
    style(fig, 165 * k + 90)
    with card("Pairplot", f"{es_int(len(d))} filas · variables ordenadas por clusters de correlación"):
        chart(fig, "r_pp_fig")


def _stratified(ctx: Ctx) -> None:
    num = _usable_numeric(ctx)
    if not need(len(num) >= 3, "Hacen falta al menos 3 columnas numéricas (X, Y y Z)."):
        return
    insight("¿La correlación entre **X** e **Y** es engañosa porque ambas dependen de una tercera variable **Z**? "
            "Se divide Z en tramos (cuantiles) y se recalcula la correlación dentro de cada tramo (paradoja de "
            "Simpson / variable de confusión).")
    c = st.columns(4)
    x = c[0].selectbox("X", num, key="r_st_x")
    y = c[1].selectbox("Y", [v for v in num if v != x], key="r_st_y")
    z = c[2].selectbox("Z (estratificar por)", [v for v in num if v not in (x, y)], key="r_st_z")
    nb = c[3].slider("Tramos de Z", 3, 10, 5, key="r_st_nb")
    split_opts = [NONE] + ctx.categorical
    split = st.selectbox("Separar además por grupo (opcional)", split_opts,
                         index=pick_index(split_opts, (str(ctx.source_col).lower(),)) if ctx.source_col else 0,
                         key="r_st_split")
    base = ctx.df[[x, y, z]].apply(pd.to_numeric, errors="coerce").dropna()
    groups = [("Total", base)]
    if split != NONE:
        g = top_with_other(ctx.df.loc[base.index, split], 3)
        groups = [(str(k), base[g == k]) for k in g.value_counts().index if k != "Otros"]
    colors = pal()
    fig = go.Figure()
    fig.add_hline(y=0, line_color=tok()["subtle"], line_width=1)
    rows = []
    for i, (label, sub) in enumerate(groups):
        if len(sub) < 20:
            continue
        r_all = pearson_ci(sub[x], sub[y])
        try:
            bucket = pd.qcut(sub[z], nb, duplicates="drop")
        except ValueError:
            continue
        pts = []
        for pos, (iv, gsub) in enumerate(sub.groupby(bucket, observed=True), start=1):
            if len(gsub) < 5 or gsub[x].std() == 0 or gsub[y].std() == 0:
                continue
            r, lo, hi, p, n = pearson_ci(gsub[x], gsub[y])
            pts.append(dict(grupo=label, tramo=pos, intervalo=str(iv), r=r, lo=lo, hi=hi, p=p, n=n))
        if not pts:
            continue
        pdf = pd.DataFrame(pts)
        rows.append(pdf)
        fig.add_trace(go.Scatter(x=pdf["tramo"], y=pdf["r"], mode="markers+lines", line=dict(color=colors[i], width=2),
                                 error_y=dict(type="data", symmetric=False, array=pdf["hi"] - pdf["r"],
                                              arrayminus=pdf["r"] - pdf["lo"], thickness=1.5, width=4),
                                 marker=dict(size=9, color=colors[i]),
                                 name=f"{label} (r global = {r_all[0]:.2f}, n = {r_all[4]:,})"))
    if not rows:
        st.warning("No hay datos suficientes para estratificar.")
        return
    with card(f"Correlación {x}–{y} dentro de cada tramo de {z}", "Barras: IC 95% de r"):
        style(fig, 420)
        fig.update_xaxes(title=f"tramo de {z} (1 = valores más bajos)", dtick=1)
        fig.update_yaxes(title="r de Pearson en el tramo")
        chart(fig, "r_st_fig")
    out = pd.concat(rows, ignore_index=True)
    out["p"] = out["p"].map(fmt_p)
    with st.expander("Tabla por tramo", icon=":material/table:"):
        st.dataframe(out, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.3f")
                                                          for c in ["r", "lo", "hi"]})
    st.caption(f"Si la r global es débil pero dentro de los tramos es consistentemente de un signo (o al revés), "
               f"{z} está confundiendo la relación entre {x} e {y}.")
