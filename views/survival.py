"""Supervivencia (Kaplan-Meier): tiempo hasta un evento, con datos censurados."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eda.config import pal, rgba, style, tok

from ._components import card, page_header
from ._ui import NONE, Ctx, chart, insight, need, pick_index, top_with_other


@st.cache_data(show_spinner=False, max_entries=64)
def _fit_km(_d: pd.DataFrame, key: str):
    from lifelines import KaplanMeierFitter
    from lifelines.utils import median_survival_times, restricted_mean_survival_time
    kmf = KaplanMeierFitter().fit(_d["duration"], _d["event"])
    sf = kmf.survival_function_.iloc[:, 0]
    ci = kmf.confidence_interval_
    med = kmf.median_survival_time_
    if np.isfinite(med):
        m_ci = median_survival_times(kmf.confidence_interval_).iloc[0]
        lo, hi = m_ci.iloc[0], m_ci.iloc[1]
    else:
        lo = hi = np.nan
    return sf, ci, med, lo, hi, restricted_mean_survival_time(kmf, t=_d["duration"].max())


def render(ctx: Ctx) -> None:
    page_header("Modelado", "Análisis de supervivencia",
                "Para datos de «tiempo hasta un evento» (abandono, fallo, alta…) en los que algunos casos aún no han "
                "tenido el evento (censurados). Curvas de Kaplan-Meier y test log-rank.")
    df = ctx.df
    num = [c for c in ctx.numeric if ctx.types.get(c) == "numérica"]
    binaries = [c for c in df.columns if ctx.types.get(c) == "booleana"]
    if not need(bool(num) and bool(binaries), "Hace falta una columna numérica de duración y una binaria de evento "
                                              "(1 = ocurrió, 0 = censurado)."):
        return
    c = st.columns(4)
    dur = c[0].selectbox("Duración", num, key="sv_dur",
                         index=pick_index(num, ("duracion", "duración", "tiempo", "time", "antig", "meses", "dias")))
    ev = c[1].selectbox("Evento", binaries, key="sv_ev",
                        index=pick_index(binaries, ("evento", "event", "abandono", "churn", "muerte", "status")))
    levels = sorted(df[ev].dropna().unique(), key=str)
    ev_val = c[2].selectbox("Valor que indica evento", levels, index=len(levels) - 1, key=f"sv_evval_{ev}")
    grp = c[3].selectbox("Comparar grupos", [NONE] + ctx.categorical, key="sv_grp")

    d = pd.DataFrame({"duration": pd.to_numeric(df[dur], errors="coerce"),
                      "event": (df[ev] == ev_val).astype(int).where(df[ev].notna())}).dropna()
    d = d[d["duration"] >= 0]
    if not need(len(d) >= 5, "No hay filas válidas (duración ≥ 0 y evento conocido)."):
        return
    groups, gser = [("Total", d)], None
    if grp != NONE:
        gser = top_with_other(df.loc[d.index, grp], 5)
        groups = [(str(k), d[gser == k]) for k in gser.value_counts().index]
    colors = pal()
    fig = go.Figure()
    rows = []
    for i, (label, sub) in enumerate(groups):
        if len(sub) < 3:
            continue
        sf, ci, med, lo, hi, rmst = _fit_km(sub, f"{ctx.key}:{dur}:{ev}:{ev_val}:{grp}:{label}")
        col = tok()["subtle"] if label == "Otros" else colors[i]
        fig.add_scatter(x=sf.index, y=ci.iloc[:, 0], mode="lines", line=dict(width=0), line_shape="hv",
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=sf.index, y=ci.iloc[:, 1], mode="lines", line=dict(width=0), line_shape="hv", fill="tonexty",
                        fillcolor=rgba(col, 0.16), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=sf.index, y=sf.values, mode="lines", line_shape="hv", name=label,
                        line=dict(color=col, width=2), hovertemplate="t = %{x:,.4g}: %{y:.1%}<extra>" + label + "</extra>")
        rows.append({"grupo": label, "n": len(sub), "eventos": int(sub["event"].sum()),
                     "censurados": int((sub["event"] == 0).sum()),
                     "mediana": f"{med:.4g}" if np.isfinite(med) else "no alcanzada",
                     "IC 95% mediana": f"[{lo:.4g}, {hi:.4g}]" if np.isfinite(med) else "—", "RMST": rmst})
    with card("Curva de Kaplan-Meier", "Probabilidad de no haber tenido el evento · banda: IC 95%"):
        style(fig, 440, legend=len(rows) > 1)
        fig.update_xaxes(title=dur)
        fig.update_yaxes(title="supervivencia", range=[0, 1.02], tickformat=".0%")
        fig.update_layout(hovermode="x unified")
        chart(fig, "sv_km")
    with card("Resumen por grupo", "Mediana: tiempo en que la mitad ha tenido el evento · RMST: tiempo medio sin "
                                   "evento hasta el último instante observado"):
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     column_config={"RMST": st.column_config.NumberColumn(format="%.4g")})
    if gser is not None and len(rows) >= 2:
        from lifelines.statistics import multivariate_logrank_test
        res = multivariate_logrank_test(d["duration"], gser, d["event"])
        insight(f"**Test log-rank**: {'las curvas difieren significativamente' if res.p_value < 0.05 else 'sin diferencias significativas'} "
                f"entre grupos · χ² = {res.test_statistic:.2f} · p = {res.p_value:.4g}",
                "ok" if res.p_value < 0.05 else "info")
