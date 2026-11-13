"""Multivariante: PCA (estructura de las variables) y clustering K-means (grupos de filas)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eda.charts import sample_for_plot
from eda.config import div_scale, pal, style, tok

from ._components import card, kpi_row, page_header
from ._ui import NONE, Ctx, chart, download_csv, es_int, es_pct, insight, need, sample_rows, sub_nav, top_with_other


def render(ctx: Ctx) -> None:
    page_header("Modelado", "Análisis multivariante",
                "Estructura conjunta de las variables numéricas: componentes principales (PCA) para reducir "
                "dimensiones y K-means para descubrir grupos de observaciones parecidas.")
    num = [c for c in ctx.numeric if ctx.df[c].nunique() > 1]
    if not need(len(num) >= 2, "Hacen falta al menos 2 variables numéricas no constantes."):
        return
    part = sub_nav(["PCA", "Clustering (K-means)"], key="mv_nav")
    (_pca if part == "PCA" else _kmeans)(ctx, num)


def _complete(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return X.loc[:, X.nunique() > 1]


@st.cache_data(show_spinner="Calculando PCA…", max_entries=8)
def _fit_pca(_df: pd.DataFrame, key: str, cols: tuple):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = _complete(_df, list(cols))
    n_comp = min(X.shape[1], 15, len(X))
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_comp).fit(Z)
    return X.index, list(X.columns), pca.transform(Z), pca.explained_variance_ratio_, pca.components_


@st.cache_data(show_spinner="Probando distintos números de clusters…", max_entries=8)
def _kmeans_scan(_X: pd.DataFrame, key: str, k_max: int) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(_X)
    idx = np.random.default_rng(0).choice(len(Z), min(len(Z), 5000), replace=False)
    rows = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(Z)
        sil = silhouette_score(Z[idx], km.labels_[idx]) if len(set(km.labels_[idx])) > 1 else np.nan
        rows.append(dict(k=k, inercia=km.inertia_, silueta=sil))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Ajustando K-means…", max_entries=8)
def _kmeans_fit(_X: pd.DataFrame, key: str, k: int):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(_X)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    return labels, PCA(n_components=2).fit_transform(Z)


def _pca(ctx: Ctx, num: list[str]) -> None:
    c1, c2 = st.columns([3, 1])
    cols = c1.multiselect("Variables", num, default=[c for c in num if ctx.types.get(c) == "numérica"][:20],
                          key="mv_pca_vars")
    color = c2.selectbox("Color", [NONE] + ctx.categorical, key="mv_pca_color")
    if len(cols) < 2:
        return
    idx, used, scores, evr, comps = _fit_pca(ctx.df, ctx.key, tuple(cols))
    if not need(len(idx) >= 5 and len(used) >= 2, "Muy pocas filas completas en estas variables."):
        return
    labels = [f"PC{i + 1}" for i in range(len(evr))]
    kpi_row([dict(label="Varianza en PC1 + PC2", value=es_pct(evr[:2].sum() * 100)),
             dict(label="Componentes para el 80%", value=str(int(np.searchsorted(np.cumsum(evr), 0.8) + 1))),
             dict(label="Autovalor > 1 (Kaiser)", value=str(int((evr * len(used) > 1).sum()))),
             dict(label="Filas completas", value=es_int(len(idx)))])
    colors = pal()
    a, b = st.columns(2, gap="medium")
    with a, card("Varianza explicada", "Barras: por componente · línea: acumulada (guía en 80%)"):
        fig = go.Figure()
        fig.add_bar(x=labels, y=evr, name="por componente", marker_color=colors[0],
                    hovertemplate="%{x}: %{y:.1%}<extra></extra>")
        fig.add_scatter(x=labels, y=np.cumsum(evr), name="acumulada", mode="lines+markers",
                        line=dict(color=colors[1], width=2), marker=dict(size=8),
                        hovertemplate="hasta %{x}: %{y:.1%}<extra></extra>")
        fig.add_hline(y=0.8, line_dash="dot", line_color=tok()["subtle"])
        style(fig, 360)
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        chart(fig, "mv_scree")
    with b, card("Proyección PC1–PC2", "Flechas: hacia dónde crece cada variable (biplot)"):
        d = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1]}, index=idx)
        cats = [None]
        if color != NONE:
            d["g"] = top_with_other(ctx.df.loc[idx, color], 3)
            cats = d["g"].value_counts().index.tolist()
        d = sample_for_plot(d, 10000)
        fig = go.Figure()
        for i, cat in enumerate(cats):
            sub = d if cat is None else d[d["g"] == cat]
            col = tok()["subtle"] if cat == "Otros" else colors[i]
            fig.add_scattergl(x=sub["PC1"], y=sub["PC2"], mode="markers", name="filas" if cat is None else str(cat),
                              marker=dict(size=5, color=col, opacity=0.45), showlegend=cat is not None)
        scale = np.abs(scores[:, :2]).max() * 0.85
        for i, v in enumerate(used):
            fig.add_annotation(x=comps[0, i] * scale, y=comps[1, i] * scale, ax=0, ay=0, xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=1.5,
                               arrowcolor=tok()["text"])
            fig.add_annotation(x=comps[0, i] * scale * 1.12, y=comps[1, i] * scale * 1.12, text=v, showarrow=False,
                               font=dict(size=10, color=tok()["text"]))
        style(fig, 360)
        fig.update_xaxes(title=f"PC1 ({evr[0] * 100:.1f}%)")
        fig.update_yaxes(title=f"PC2 ({evr[1] * 100:.1f}%)")
        chart(fig, "mv_scatter")
    nshow = min(len(evr), 8)
    load = pd.DataFrame(comps[:nshow].T, index=used, columns=labels[:nshow])
    with card("Cargas", "Peso de cada variable en cada componente: las que comparten signo y tamaño se mueven juntas"):
        fig = go.Figure(go.Heatmap(z=load.values, x=load.columns, y=load.index, zmin=-1, zmax=1,
                                   colorscale=div_scale(), text=np.round(load.values, 2), texttemplate="%{text}",
                                   colorbar=dict(thickness=10), hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>"))
        style(fig, max(280, 26 * len(load) + 90))
        fig.update_yaxes(autorange="reversed")
        chart(fig, "mv_loadings")


def _kmeans(ctx: Ctx, num: list[str]) -> None:
    insight("Exploratorio: los clusters dependen de las variables elegidas y de su escala (aquí se estandarizan). "
            "Úsalo para generar hipótesis, no como verdad.")
    c1, c2 = st.columns([3, 1])
    cols = c1.multiselect("Variables", num, default=[c for c in num if ctx.types.get(c) == "numérica"][:6],
                          key="mv_km_vars")
    k_max = c2.number_input("Probar hasta k =", 3, 12, 8, key="mv_km_kmax")
    if len(cols) < 2:
        return
    X = _complete(ctx.df, cols)
    if not need(len(X) >= 20, "Muy pocas filas completas."):
        return
    Xf = sample_rows(X, 30000)
    xkey = f"{ctx.key}:{tuple(cols)}:{len(Xf)}"
    scan = _kmeans_scan(Xf, xkey, int(k_max))
    best = int(scan.loc[scan["silueta"].idxmax(), "k"]) if scan["silueta"].notna().any() else 3
    colors = pal()
    a, b = st.columns(2, gap="medium")
    # Two small charts instead of one dual-axis chart: never two y-scales on one plot.
    with a, card("Silueta por k", "Más alto = clusters más separados"):
        fig = go.Figure(go.Scatter(x=scan["k"], y=scan["silueta"], mode="lines+markers",
                                   line=dict(color=colors[0], width=2), marker=dict(size=8),
                                   hovertemplate="k = %{x}: %{y:.3f}<extra></extra>"))
        fig.add_vline(x=best, line_dash="dot", line_color=tok()["subtle"], annotation_text=f"k = {best}")
        style(fig, 220, legend=False)
        fig.update_xaxes(dtick=1, title="k")
        chart(fig, "mv_km_sil")
    with b, card("Inercia por k (método del codo)", "Busca el punto donde deja de bajar con fuerza"):
        fig = go.Figure(go.Scatter(x=scan["k"], y=scan["inercia"], mode="lines+markers",
                                   line=dict(color=colors[0], width=2), marker=dict(size=8),
                                   hovertemplate="k = %{x}: %{y:,.0f}<extra></extra>"))
        style(fig, 220, legend=False)
        fig.update_xaxes(dtick=1, title="k")
        chart(fig, "mv_km_elbow")
    k = st.slider("Número de clusters", 2, int(k_max), best, key="mv_km_k",
                  help=f"Sugerido por silueta: k = {best} (> 0,5 estructura clara, < 0,25 débil).")
    labels, P = _kmeans_fit(Xf, xkey, k)
    lab = pd.Series(labels, index=Xf.index).map(lambda v: f"Cluster {v + 1}")
    names = sorted(lab.unique(), key=lambda s: int(s.split()[-1]))
    sizes = lab.value_counts().reindex(names)
    c, d_ = st.columns([3, 2], gap="medium")
    with c, card("Clusters en el plano PCA"):
        d = sample_for_plot(pd.DataFrame({"PC1": P[:, 0], "PC2": P[:, 1], "c": lab.to_numpy()}), 10000)
        fig = go.Figure()
        for i, nm in enumerate(names):
            sub = d[d["c"] == nm]
            fig.add_scattergl(x=sub["PC1"], y=sub["PC2"], mode="markers", name=nm,
                              marker=dict(size=5, color=colors[i % 8], opacity=0.5))
        style(fig, 380)
        chart(fig, "mv_km_pca")
    with d_, card("Tamaño de cada cluster"):
        fig = go.Figure(go.Bar(x=sizes.values, y=names, orientation="h", marker_color=[colors[i % 8] for i in range(len(names))],
                               text=[es_pct(v / sizes.sum() * 100) for v in sizes.values], textposition="outside",
                               hovertemplate="%{y}: %{x:,} filas<extra></extra>"))
        style(fig, 380, legend=False)
        fig.update_yaxes(autorange="reversed")
        chart(fig, "mv_km_sizes")
    prof = Xf.groupby(lab).mean().reindex(names)
    zprof = (prof - Xf.mean()) / Xf.std()
    with card("Perfil de cada cluster", "Media estandarizada: + por encima de la media global, − por debajo"):
        fig = go.Figure(go.Heatmap(z=zprof.values, x=zprof.columns, y=zprof.index, zmin=-2, zmax=2,
                                   colorscale=div_scale(), text=np.round(zprof.values, 2), texttemplate="%{text}",
                                   colorbar=dict(thickness=10), hovertemplate="%{y} · %{x}: %{z:.2f} σ<extra></extra>"))
        style(fig, max(240, 40 * len(prof) + 90))
        fig.update_yaxes(autorange="reversed")
        chart(fig, "mv_km_profile")
        tbl = prof.copy()
        tbl.insert(0, "tamaño", sizes)
        st.dataframe(tbl, column_config={c: st.column_config.NumberColumn(format="%.3g") for c in tbl.columns})
    download_csv(lambda: ctx.df.loc[Xf.index].assign(cluster=lab), "datos_con_cluster.csv",
                 "Descargar filas con su cluster", key="mv_km_dl")
