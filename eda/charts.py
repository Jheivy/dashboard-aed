"""Trazas de Plotly pre-agregadas en el servidor.

Un histograma o boxplot con datos crudos envía cada punto al navegador (50.000 filas ≈ 1–2 MB
de JSON por gráfico). Aquí se calculan los bins y los cuartiles en NumPy y solo viajan los
resultados (unos pocos KB), con tooltips en español.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

MAX_POINTS = 10_000


def finite_values(x) -> np.ndarray:
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return a[np.isfinite(a)]


def bin_edges(values: list[np.ndarray] | np.ndarray, bins: int = 50) -> np.ndarray:
    """Shared bin edges so several groups can be overlaid on the same bins."""
    arrs = values if isinstance(values, list) else [values]
    allv = np.concatenate([finite_values(a) for a in arrs]) if arrs else np.array([])
    if allv.size == 0:
        return np.array([0.0, 1.0])
    lo, hi = float(allv.min()), float(allv.max())
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return np.histogram_bin_edges(allv, bins=bins, range=(lo, hi))


def hist_bar(x, bins: int | np.ndarray = 50, name: str = "", color: str | None = None,
             density: bool = False, opacity: float = 1.0, showlegend: bool = False,
             legendgroup: str | None = None) -> go.Bar:
    v = finite_values(x)
    edges = bins if isinstance(bins, np.ndarray) else bin_edges(v, bins)
    counts, edges = np.histogram(v, bins=edges)
    widths = np.diff(edges)
    y = counts / (counts.sum() * widths) if (density and counts.sum()) else counts
    pct = counts / counts.sum() * 100 if counts.sum() else counts * 0.0
    return go.Bar(
        x=(edges[:-1] + edges[1:]) / 2, y=y, width=widths * 0.94, name=name,
        marker=dict(color=color, line=dict(width=0)), opacity=opacity, showlegend=showlegend,
        legendgroup=legendgroup or name,
        customdata=np.column_stack([edges[:-1], edges[1:], counts, pct]),
        hovertemplate=("[%{customdata[0]:,.4g} ; %{customdata[1]:,.4g})<br>"
                       "<b>%{customdata[2]:,}</b> filas · %{customdata[3]:.1f}%"
                       f"<extra>{name}</extra>"),
    )


def box_stats(x) -> dict | None:
    v = finite_values(x)
    if v.size == 0:
        return None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = v[(v >= lo_f) & (v <= hi_f)]
    out = v[(v < lo_f) | (v > hi_f)]
    return dict(q1=q1, median=med, q3=q3, mean=v.mean(), n=v.size,
                lowerfence=inside.min() if inside.size else q1,
                upperfence=inside.max() if inside.size else q3, outliers=out)


def box_traces(x, name: str, color: str, horizontal: bool = True, max_outliers: int = 400,
               showlegend: bool = False) -> list:
    """Box drawn from precomputed quartiles + a capped sample of outlier points."""
    s = box_stats(x)
    if s is None:
        return []
    pos = [name]
    kw = dict(q1=[s["q1"]], median=[s["median"]], q3=[s["q3"]], mean=[s["mean"]],
              lowerfence=[s["lowerfence"]], upperfence=[s["upperfence"]])
    box = go.Box(**kw, name=name, marker_color=color, line=dict(width=1.5), fillcolor=None,
                 orientation="h" if horizontal else "v", showlegend=showlegend, boxpoints=False,
                 **({"y": pos} if horizontal else {"x": pos}),
                 hoverinfo="skip")
    out = s["outliers"]
    traces = [box]
    if out.size:
        if out.size > max_outliers:
            out = np.random.default_rng(0).choice(out, max_outliers, replace=False)
        pts = dict(x=out, y=[name] * out.size) if horizontal else dict(x=[name] * out.size, y=out)
        traces.append(go.Scattergl(**pts, mode="markers", showlegend=False, name=name,
                                   marker=dict(size=5, color=color, opacity=0.55),
                                   hovertemplate="%{" + ("x" if horizontal else "y") +
                                                 ":,.4g}<extra>" + name + " · outlier</extra>"))
    # Invisible point carrying the summary as tooltip (Box with precomputed stats has none)
    summary = (f"n = {s['n']:,}<br>mediana = {s['median']:,.4g}<br>Q1 = {s['q1']:,.4g} · "
               f"Q3 = {s['q3']:,.4g}<br>media = {s['mean']:,.4g}")
    hp = dict(x=[s["median"]], y=[name]) if horizontal else dict(x=[name], y=[s["median"]])
    traces.append(go.Scatter(**hp, mode="markers", marker=dict(size=14, opacity=0), showlegend=False,
                             hovertemplate=summary + f"<extra>{name}</extra>"))
    return traces


def sample_for_plot(df: pd.DataFrame, n: int = MAX_POINTS, keep: pd.Series | None = None) -> pd.DataFrame:
    """Random sample for scatter plots, always keeping the rows flagged in `keep`."""
    if len(df) <= n:
        return df
    if keep is not None and keep.any():
        flagged = df[keep.reindex(df.index, fill_value=False)]
        rest = df.drop(flagged.index)
        return pd.concat([flagged.head(n // 2), rest.sample(max(n - min(len(flagged), n // 2), 0),
                                                            random_state=0)])
    return df.sample(n, random_state=0)
