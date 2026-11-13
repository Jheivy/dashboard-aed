"""Outliers: resumen de todas las variables, detalle univariante con tratamiento y multivariante."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from eda.charts import bin_edges, box_traces, hist_bar, sample_for_plot
from eda.config import normal_color, outlier_color, pal, seq_scale, style, tok
from eda.outliers import (UNI_HELP, UNI_METHODS, multivariate_scores, outlier_overview, uni_bounds,
                          winsorize)

from ._components import card, kpi_row, page_header
from ._ui import Ctx, chart, download_csv, es_int, es_num, es_pct, finite, insight, need, sub_nav

MULTI_METHODS = ["Mahalanobis robusta (MCD)", "Isolation Forest", "Local Outlier Factor", "Mahalanobis clásica"]


def render(ctx: Ctx) -> None:
    page_header("Exploración", "Outliers",
                "Valores anómalos variable a variable y combinaciones raras entre variables. Un outlier no es "
                "necesariamente un error: decide si corregir, excluir, recortar o conservar.")
    if not need(bool(ctx.numeric), "No hay columnas numéricas para buscar outliers."):
        return
    part = sub_nav(["Resumen de variables", "Detalle de una variable", "Multivariante"], key="o_nav")
    {"Resumen de variables": _overview, "Detalle de una variable": _detail,
     "Multivariante": _multivariate}[part](ctx)


def _overview(ctx: Ctx) -> None:
    c = st.columns(3)
    k = c[0].number_input("k de IQR", 1.0, 5.0, 1.5, 0.5, key="o_ov_k")
    z = c[1].number_input("Umbral z", 2.0, 6.0, 3.0, 0.5, key="o_ov_z")
    zr = c[2].number_input("Umbral z robusto", 2.0, 8.0, 3.5, 0.5, key="o_ov_zr")
    cols = [col for col in ctx.numeric if ctx.types.get(col) == "numérica"]
    ov = outlier_overview(ctx.df, ctx.key, cols, k, z, zr)
    if not need(not ov.empty, "No hay variables numéricas con suficiente variación."):
        return
    top = ov.head(30)
    colors = pal()
    with card("% de outliers por variable y método",
              "Si el z-score marca muchos menos que IQR/MAD, la variable es asimétrica o los propios outliers "
              "inflan la desviación (enmascaramiento): fíate más de IQR y MAD"):
        fig = go.Figure()
        for i, m in enumerate(["IQR (Tukey)", "Z-score", "Z-score robusto (MAD)"]):
            fig.add_bar(y=top["variable"], x=top[f"% {m}"], name=m, orientation="h", marker_color=colors[i],
                        hovertemplate="%{y}: %{x:.2f}%<extra>" + m + "</extra>")
        fig.update_layout(barmode="group")
        style(fig, max(300, 34 * len(top) + 90))
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_xaxes(title="% de valores marcados")
        chart(fig, "o_ov_bar")
    with card("Tabla resumen"):
        cfg = {c: st.column_config.NumberColumn(format="%.2f") for c in ov.columns if c.startswith("%") or c == "asimetría"}
        cfg.update({c: st.column_config.NumberColumn(format="%.4g")
                    for c in ["límite inf. IQR", "límite sup. IQR", "mín", "máx"]})
        st.dataframe(ov, hide_index=True, column_config=cfg)
    with card("Boxplots estandarizados", "Cada variable en unidades robustas: (x − mediana) / IQR"):
        sel = st.multiselect("Variables", ov["variable"].tolist(), default=ov["variable"].head(12).tolist(),
                             key="o_ov_box", label_visibility="collapsed")
        if sel:
            fig = go.Figure()
            for c in sel:
                x = finite(ctx.df[c])
                iqr = x.quantile(0.75) - x.quantile(0.25)
                for tr in box_traces((x - x.median()) / (iqr if iqr else (x.std() or 1)), c, colors[0],
                                     horizontal=False, max_outliers=150):
                    fig.add_trace(tr)
            style(fig, 400, legend=False)
            fig.update_yaxes(title="(x − mediana) / IQR")
            chart(fig, "o_ov_boxes")


def _detail(ctx: Ctx) -> None:
    df = ctx.df
    c = st.columns([2, 2, 2, 1])
    var = c[0].selectbox("Variable", ctx.numeric, key="o_var")
    method = c[1].selectbox("Método", list(UNI_METHODS), key="o_method")
    label, lo_, hi_, default, step = UNI_METHODS[method]
    param = c[2].slider(label, lo_, hi_, default, step, key=f"o_param_{method}")
    log_view = c[3].toggle("Eje log", key="o_log")
    st.caption(UNI_HELP[method])

    x = finite(df[var])
    if x.empty:
        st.warning("Sin valores válidos.")
        return
    lo, hi = uni_bounds(x, method, param)
    mask = (x < lo) | (x > hi)
    kpi_row([dict(label="Outliers", value=es_int(mask.sum())), dict(label="% de valores", value=es_pct(mask.mean() * 100, 2)),
             dict(label="Por abajo / arriba", value=f"{es_int((x < lo).sum())} / {es_int((x > hi).sum())}"),
             dict(label="Límite inferior", value=es_num(lo)), dict(label="Límite superior", value=es_num(hi))])

    with card("Valores y límites", "Puntos rojos = outliers (muestra de hasta 6.000 puntos, outliers siempre "
                                   "incluidos) · histograma completo debajo"):
        pts = sample_for_plot(pd.DataFrame({"x": x, "m": mask}), 6000, keep=mask)
        jitter = np.random.default_rng(0).uniform(-0.4, 0.4, len(pts))
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.35, 0.65], vertical_spacing=0.04)
        for flag, name, color, size, op in [(False, "normal", normal_color(), 4, 0.35),
                                            (True, "outlier", outlier_color(), 7, 0.9)]:
            sel = (pts["m"] == flag).to_numpy()
            fig.add_trace(go.Scattergl(x=pts["x"][sel], y=jitter[sel], mode="markers", name=name,
                                       marker=dict(size=size, color=color, opacity=op),
                                       hovertemplate="%{x:,.4g}<extra>" + name + "</extra>"), 1, 1)
        fig.add_trace(hist_bar(x, 80, var, normal_color()), 2, 1)
        for b in (lo, hi):
            fig.add_vline(x=b, line_dash="dash", line_width=1.5, line_color=outlier_color())
        fig.update_yaxes(showticklabels=False, row=1, col=1)
        if log_view and (x > 0).all():
            fig.update_xaxes(type="log")
        style(fig, 400)
        fig.update_xaxes(title=var, row=2, col=1)
        chart(fig, "o_detail_fig")

    a, b = st.columns([3, 2], gap="medium")
    clean = x[~mask]
    wins = winsorize(x, lo, hi)
    imp = pd.DataFrame({
        "con outliers": [len(x), x.mean(), x.median(), x.std(), x.skew(), x.min(), x.max()],
        "eliminando": [len(clean), clean.mean(), clean.median(), clean.std(), clean.skew(), clean.min(), clean.max()],
        "recortando (winsorizar)": [len(wins), wins.mean(), wins.median(), wins.std(), wins.skew(), wins.min(), wins.max()],
    }, index=["n", "media", "mediana", "desv. típica", "asimetría", "mín", "máx"])
    with a, card("Impacto de tratarlos", "Cómo cambian los estadísticos según la decisión"):
        st.dataframe(imp, column_config={c: st.column_config.NumberColumn(format="%.4g") for c in imp.columns})
        if abs(clean.mean() - x.mean()) > 0.1 * abs(x.std() or 1):
            insight("Los outliers desplazan la media más de 0,1 desviaciones típicas: describe esta variable con "
                    "la mediana o estadísticos robustos.", "warn")
    with b, card("¿Se concentran en algún grupo?"):
        if ctx.categorical:
            g = st.selectbox("Grupo", ctx.categorical, key="o_grp", label_visibility="collapsed")
            gs = df.loc[x.index, g].astype(str)
            rate = (mask.groupby(gs).mean() * 100).sort_values(ascending=False).head(15)
            fig = go.Figure(go.Bar(x=rate.values, y=rate.index, orientation="h", marker_color=outlier_color(),
                                   hovertemplate="%{y}: %{x:.2f}% outliers<extra></extra>"))
            fig.add_vline(x=mask.mean() * 100, line_dash="dot", line_color=tok()["muted"], annotation_text="global")
            style(fig, max(220, 26 * len(rate) + 60), legend=False)
            fig.update_yaxes(autorange="reversed", title="")
            fig.update_xaxes(title="% outliers")
            chart(fig, "o_grp_fig")
        else:
            st.caption("No hay variables categóricas.")

    rows_idx = mask[mask].index
    with card(f"Filas marcadas ({es_int(len(rows_idx))})"):
        if len(rows_idx):
            st.dataframe(df.loc[rows_idx].sort_values(var, ascending=False).head(500), height=280)
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                download_csv(lambda: df.loc[rows_idx], f"outliers_{var}.csv", "Filas outlier", key="o_dl_rows")
            with c2:
                download_csv(lambda: df.assign(**{f"outlier_{var}": df.index.isin(rows_idx)}),
                             f"datos_con_marca_outlier_{var}.csv", "Dataset con marca", key="o_dl_flag")


def _multivariate(ctx: Ctx) -> None:
    df = ctx.df
    with st.expander("¿Qué hace cada método?", icon=":material/school:"):
        st.markdown(
            "- **Mahalanobis robusta (MCD)**: distancia al centro teniendo en cuenta las correlaciones, con centro "
            "y covarianza estimados sin dejarse influir por los outliers. Umbral por χ².\n"
            "- **Mahalanobis clásica**: igual con media/covarianza normales (se enmascara con muchos outliers).\n"
            "- **Isolation Forest**: aísla puntos con cortes aleatorios; los anómalos se aíslan antes.\n"
            "- **Local Outlier Factor**: compara la densidad de cada punto con la de sus vecinos (anomalías "
            "locales dentro de clusters).\n\nLas filas marcadas por **varios métodos** son las más fiables.")
    with card("Configuración", "Busca filas cuya combinación de valores es rara aunque cada valor por separado "
                               "sea normal. Solo filas completas en las variables elegidas"):
        c1, c2 = st.columns([3, 2])
        vars_ = c1.multiselect("Variables", ctx.numeric,
                               default=[c for c in ctx.numeric if ctx.types.get(c) == "numérica"][:8], key="o_mv_vars")
        methods = c2.multiselect("Métodos", MULTI_METHODS, default=MULTI_METHODS[:3], key="o_mv_methods")
        c3, c4 = st.columns(2)
        contamination = c3.slider("Proporción esperada de anomalías (IF y LOF)", 0.005, 0.2, 0.02, 0.005,
                                  key="o_mv_cont")
        chi_q = c4.slider("Cuantil χ² (Mahalanobis)", 0.9, 0.999, 0.975, 0.001, key="o_mv_chi", format="%.3f")
    if not vars_ or not methods:
        st.info("Elige variables y al menos un método.", icon=":material/info:")
        return
    scores, notes = multivariate_scores(df, ctx.key, vars_, contamination, chi_q, tuple(methods))
    if "error" in notes:
        st.warning(notes["error"])
        return
    for m in methods:
        if m in notes:
            st.warning(f"{m}: {notes[m]}")
    flag_cols = [c for c in scores.columns if c.startswith("flag_")]
    if not flag_cols:
        st.info("Ningún método aplicable (Mahalanobis requiere 2+ variables).", icon=":material/info:")
        return
    kpi_row([dict(label=fc[5:], value=es_int(scores[fc].sum()), delta=es_pct(scores[fc].mean() * 100, 2),
                  delta_color="off") for fc in flag_cols])
    min_votes = st.slider("Consenso: marcado por al menos N métodos", 1, len(flag_cols), min(2, len(flag_cols)),
                          key="o_mv_votes")
    consensus = scores["nº métodos"] >= min_votes
    st.caption(f"**{es_int(consensus.sum())}** filas ({es_pct(consensus.mean() * 100, 2)}) marcadas por ≥ {min_votes} "
               f"método(s) · {es_int(len(scores))} filas completas analizadas. {notes.get('muestra', '')}")

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = df.loc[scores.index, notes["variables"]].astype(float)
    a, b = st.columns(2, gap="medium")
    with a, card("Proyección PCA", "Rojo = outlier por consenso"):
        if X.shape[1] >= 2:
            pc = PCA(n_components=2).fit(StandardScaler().fit_transform(X))
            P = pc.transform(StandardScaler().fit_transform(X))
            d = pd.DataFrame({"PC1": P[:, 0], "PC2": P[:, 1], "out": consensus.to_numpy(),
                              "votos": scores["nº métodos"].to_numpy()})
            d = sample_for_plot(d, 10000, keep=d["out"])
            fig = go.Figure()
            for flag, name, color, size in [(False, "normal", normal_color(), 5), (True, "outlier", outlier_color(), 8)]:
                s = d[d["out"] == flag]
                fig.add_scattergl(x=s["PC1"], y=s["PC2"], mode="markers", name=name, customdata=s["votos"],
                                  marker=dict(size=size, color=color, opacity=0.5 if not flag else 0.9),
                                  hovertemplate="votos: %{customdata}<extra>" + name + "</extra>")
            ev = pc.explained_variance_ratio_ * 100
            style(fig, 380)
            fig.update_xaxes(title=f"PC1 ({ev[0]:.1f}%)")
            fig.update_yaxes(title=f"PC2 ({ev[1]:.1f}%)")
            chart(fig, "o_mv_pca")
        else:
            st.caption("Hace falta más de una variable.")
    with b, card("Distribución de la puntuación"):
        score_cols = [c for c in scores.columns if not c.startswith("flag_") and c != "nº métodos"]
        sc = st.selectbox("Método", score_cols, key="o_mv_score", label_visibility="collapsed")
        flags = scores[f"flag_{sc}"]
        edges = bin_edges(scores[sc].to_numpy(), 70)
        fig = go.Figure([hist_bar(scores.loc[~flags, sc], edges, "normal", normal_color(), showlegend=True),
                         hist_bar(scores.loc[flags, sc], edges, "outlier", outlier_color(), showlegend=True)])
        fig.update_layout(barmode="overlay")
        if sc.startswith("Mahalanobis"):
            fig.add_vline(x=notes["umbral_chi2"], line_dash="dash", line_color=tok()["muted"],
                          annotation_text=f"χ² {chi_q:.1%}")
        style(fig, 330)
        fig.update_xaxes(title="puntuación de anomalía")
        chart(fig, "o_mv_hist")

    if len(flag_cols) >= 2:
        with st.expander("Coincidencia entre métodos (índice de Jaccard)", icon=":material/join_inner:"):
            names = [c[5:] for c in flag_cols]
            agree = np.array([[((scores[a_] & scores[b_]).sum() / max((scores[a_] | scores[b_]).sum(), 1))
                               for b_ in flag_cols] for a_ in flag_cols])
            fig = go.Figure(go.Heatmap(z=agree, x=names, y=names, zmin=0, zmax=1, colorscale=seq_scale(),
                                       text=np.round(agree, 2), texttemplate="%{text}",
                                       colorbar=dict(thickness=10)))
            style(fig, 300)
            chart(fig, "o_mv_jacc")

    with card("Filas más anómalas", "Y la variable en la que más se alejan de la mediana"):
        idx = scores.index[consensus]
        if len(idx):
            rows = df.loc[idx].copy()
            rows.insert(0, "nº métodos", scores.loc[idx, "nº métodos"])
            iqr = (X.quantile(0.75) - X.quantile(0.25)).replace(0, np.nan)
            rows.insert(1, "variable más atípica", ((X.loc[idx] - X.median()) / iqr).abs().idxmax(axis=1).fillna("—"))
            st.dataframe(rows.sort_values("nº métodos", ascending=False).head(500), height=300)
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                download_csv(rows, "outliers_multivariantes.csv", "Filas anómalas", key="o_dl_mv")
            with c2:
                download_csv(lambda: df.assign(outlier_multivariante_votos=scores["nº métodos"].reindex(df.index)),
                             "datos_con_votos_outlier.csv", "Dataset con votos", key="o_dl_mv_full")
        else:
            st.caption("Ninguna fila alcanza el consenso elegido.")
