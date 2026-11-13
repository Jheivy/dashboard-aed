"""Variable objetivo: qué variables se relacionan más con un target (clasificación o regresión)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from eda.charts import bin_edges, box_traces, hist_bar, sample_for_plot
from eda.config import SCALES, pal, style, tok
from eda.stats_utils import apply_scales, compare_groups, correlation_ratio, cramers_v, fmt_p

from ._components import card, kpi_row, page_header
from ._ui import Ctx, chart, es_int, es_pct, insight, need, pick_index, sub_nav, top_with_other

MAX_LEVELS = 30


def render(ctx: Ctx) -> None:
    page_header("Modelado", "Variable objetivo",
                "Elige la variable que quieres explicar o predecir y descubre qué otras se relacionan más con "
                "ella. También detecta posibles fugas de información.")
    df = ctx.df
    cands = [c for c in df.columns if c in ctx.numeric or c in ctx.categorical]
    if not need(bool(cands), "No hay variables utilizables como objetivo."):
        return
    c1, c2 = st.columns([2, 1])
    target = c1.selectbox("Variable objetivo", cands, key="t_target",
                          index=pick_index(cands, ("target", "objetivo", "label", "churn", "abandono", "clase",
                                                   "class")))
    nun = df[target].nunique()
    default_task = "Clasificación" if (target in ctx.categorical or nun <= 10) else "Regresión"
    task = c2.segmented_control("Tipo de problema", ["Clasificación", "Regresión"], default=default_task,
                                key=f"t_task_{target}") or default_task
    if task == "Regresión" and not pd.api.types.is_numeric_dtype(df[target]):
        st.warning("Para regresión el objetivo debe ser numérico.")
        return
    if task == "Clasificación" and nun > MAX_LEVELS:
        st.warning(f"El objetivo tiene {nun} clases; agrúpalas o elige regresión.")
        return
    opts = ["Ranking de variables", "Variable vs objetivo"] + (["Curvas ROC"] if task == "Clasificación" and nun == 2 else [])
    part = sub_nav(opts, key="t_nav")
    if part == "Ranking de variables":
        _ranking(ctx, target, task)
    elif part == "Variable vs objetivo":
        _feature_vs_target(ctx, target, task)
    else:
        _roc(ctx, target)


@st.cache_data(show_spinner="Calculando la relación de cada variable con el objetivo…", max_entries=16)
def _rank_table(_df: pd.DataFrame, key: str, target: str, task: str, numeric: list[str],
                categorical: list[str]) -> pd.DataFrame:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.metrics import roc_auc_score
    df = _df
    y = df[target]
    rows = []
    binary = task == "Clasificación" and y.nunique() == 2
    pos = sorted(y.dropna().unique(), key=str)[-1] if binary else None
    for c in numeric:
        if c == target:
            continue
        d = pd.DataFrame({"x": pd.to_numeric(df[c], errors="coerce"), "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < 10 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
            continue
        r = dict(variable=c, tipo="numérica", n=len(d))
        if task == "Clasificación":
            r.update(asociación=correlation_ratio(d["y"].astype(str), d["x"]), medida="η")
            grp = d.groupby("y", observed=True)["x"]
            res = compare_groups([g.to_numpy() for _, g in grp], [str(k) for k, _ in grp])
            r["p"] = res["p"] if res else np.nan
            if binary:
                auc = roc_auc_score((d["y"] == pos).astype(int), d["x"])
                r["AUC"] = max(auc, 1 - auc)
                r["dirección"] = "↑ más alto → más positiva" if auc >= 0.5 else "↓ más bajo → más positiva"
        else:
            rho, p = stats.spearmanr(d["x"], d["y"])
            r.update(asociación=abs(rho), medida="|ρ Spearman|", p=p, dirección="↑ positiva" if rho >= 0 else "↓ negativa")
        rows.append(r)
    for c in categorical:
        if c == target or df[c].nunique() > 100:
            continue
        x = df[c].astype(object).where(df[c].notna(), "(nulo)")
        d = pd.DataFrame({"x": x, "y": y}).dropna()
        if d["x"].nunique() < 2 or len(d) < 10:
            continue
        r = dict(variable=c, tipo="categórica", n=len(d))
        if task == "Clasificación":
            v, p, _ = cramers_v(d["x"], d["y"])
            r.update(asociación=v, medida="V de Cramér", p=p)
        else:
            r.update(asociación=correlation_ratio(d["x"], d["y"]), medida="η")
            grp = d.groupby("x")["y"]
            res = compare_groups([g.to_numpy() for _, g in grp], [str(k) for k, _ in grp])
            r["p"] = res["p"] if res else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    smp = df.sample(min(len(df), 20000), random_state=0)
    X = pd.DataFrame(index=smp.index)
    disc = []
    for c in out["variable"]:
        if c in numeric:
            v = pd.to_numeric(smp[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            X[c] = v.fillna(v.median())
            disc.append(False)
        else:
            X[c] = smp[c].astype(str).astype("category").cat.codes
            disc.append(True)
    yy = smp[target]
    ok = yy.notna()
    if ok.sum() >= 20:
        fn = mutual_info_classif if task == "Clasificación" else mutual_info_regression
        yv = yy[ok].astype(str) if task == "Clasificación" else yy[ok].astype(float)
        out["información mutua"] = fn(X[ok], yv, discrete_features=disc, random_state=0)
    return out.sort_values("asociación", ascending=False).reset_index(drop=True)


def _target_overview(s: pd.Series, task: str) -> None:
    with card("Distribución del objetivo"):
        if task == "Clasificación":
            vc = s.astype(object).where(s.notna(), "(nulo)").astype(str).value_counts()
            pct = vc / vc.sum() * 100
            fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=pal()[: len(vc)] if len(vc) <= 8 else pal()[0],
                                   text=[es_pct(v) for v in pct], textposition="outside",
                                   hovertemplate="%{x}: %{y:,}<extra></extra>"))
            style(fig, 250, legend=False)
            fig.update_xaxes(type="category")
            chart(fig, "t_dist")
            if pct.min() < 10:
                insight("Clases desbalanceadas: la exactitud (accuracy) engaña; usa AUC, F1 o recall.", "warn")
        else:
            fig = go.Figure(hist_bar(s, 50, s.name, pal()[0]))
            style(fig, 250, legend=False)
            chart(fig, "t_dist")


def _ranking(ctx: Ctx, target: str, task: str) -> None:
    tbl = _rank_table(ctx.df, ctx.key, target, task, ctx.numeric, ctx.categorical)
    a, b = st.columns([1, 2], gap="medium")
    with a:
        _target_overview(ctx.df[target], task)
    if tbl.empty:
        b.info("No hay variables con datos suficientes.", icon=":material/info:")
        return
    with b, card(f"Variables más relacionadas con {target}"):
        metrics = [m for m in ["asociación", "AUC", "información mutua"] if m in tbl.columns]
        metric = st.segmented_control("Ordenar por", metrics, default=metrics[0], key="t_rank_by",
                                      label_visibility="collapsed") or metrics[0]
        top = tbl.sort_values(metric, ascending=False).head(20)
        colors = pal()
        fig = go.Figure()
        for i, tipo in enumerate(["numérica", "categórica"]):
            sub = top[top["tipo"] == tipo]
            if len(sub):
                fig.add_bar(y=sub["variable"], x=sub[metric], name=tipo, orientation="h", marker_color=colors[i],
                            customdata=sub["medida"], hovertemplate="%{y}: %{x:.3f} (%{customdata})<extra></extra>")
        style(fig, max(260, 24 * len(top) + 70))
        fig.update_yaxes(categoryorder="array", categoryarray=top["variable"].tolist()[::-1], title="")
        fig.update_xaxes(title=metric)
        chart(fig, "t_rank_bar")
    show = tbl.copy()
    if "p" in show:
        show["p"] = show["p"].map(fmt_p)
    with card("Tabla completa", "η, V de Cramér y |ρ| van de 0 a 1 · AUC (objetivo binario) usa la variable tal "
                                "cual: 0,5 = no discrimina · la información mutua capta relaciones no lineales"):
        st.dataframe(show, hide_index=True, column_config={
            "asociación": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
            **{c: st.column_config.NumberColumn(format="%.3f") for c in ["AUC", "información mutua"] if c in show}})
    leak = tbl["asociación"] > 0.95
    if "AUC" in tbl:
        leak |= tbl["AUC"] > 0.95
    if leak.any():
        insight("Posible **fuga de información** en: " + ", ".join(tbl.loc[leak, "variable"].head(8)) + ". Una "
                "relación casi perfecta suele indicar que la variable se calculó a partir del objetivo o se mide "
                "después de él.", "warn")


def _feature_vs_target(ctx: Ctx, target: str, task: str) -> None:
    df = ctx.df
    feats = list(dict.fromkeys(c for c in ctx.numeric + ctx.categorical if c != target))
    if not need(bool(feats), "No hay otras variables."):
        return
    f = st.selectbox("Variable", feats, key="t_feat")
    is_num = f in ctx.numeric and ctx.types.get(f) == "numérica"
    colors = pal()
    if task == "Clasificación":
        y = top_with_other(df[target], 7)
        classes = y.value_counts().index.tolist()
        if is_num:
            a, b = st.columns(2, gap="medium")
            with a, card(f"{f} según {target}", "Densidad por clase"):
                vals = [pd.to_numeric(df.loc[y == k, f], errors="coerce") for k in classes]
                edges = bin_edges([v.dropna().to_numpy() for v in vals], 50)
                fig = go.Figure([hist_bar(v, edges, str(k), colors[i % 8], density=True, opacity=0.6, showlegend=True)
                                 for i, (k, v) in enumerate(zip(classes, vals))])
                fig.update_layout(barmode="overlay")
                style(fig, 360)
                fig.update_xaxes(title=f)
                chart(fig, "t_f_hist")
            with b, card(f"% de cada clase por tramo de {f}", "Tramos por cuantiles"):
                nb = st.slider("Tramos", 3, 20, 10, key="t_fbins")
                d = pd.DataFrame({f: pd.to_numeric(df[f], errors="coerce"), "y": y}).dropna()
                try:
                    d["tramo"] = pd.qcut(d[f], nb, duplicates="drop")
                except ValueError:
                    d["tramo"] = d[f]
                rate = pd.crosstab(d["tramo"], d["y"], normalize="index") * 100
                labels = [f"{iv.left:,.3g} – {iv.right:,.3g}" if hasattr(iv, "left") else str(iv) for iv in rate.index]
                fig = go.Figure([go.Bar(x=labels, y=rate[k], name=str(k), marker_color=colors[i % 8],
                                        hovertemplate="%{x}: %{y:.1f}%<extra>" + str(k) + "</extra>")
                                 for i, k in enumerate(classes) if k in rate.columns])
                fig.update_layout(barmode="stack")
                style(fig, 320)
                fig.update_xaxes(tickangle=-35, type="category")
                fig.update_yaxes(title="%", range=[0, 100])
                chart(fig, "t_f_rate")
        else:
            x = top_with_other(df[f], 15)
            ct = pd.crosstab(x, y)
            rate = ct.div(ct.sum(1), axis=0) * 100
            with card(f"% de cada clase de {target} por {f}"):
                fig = go.Figure([go.Bar(y=rate.index, x=rate[k], name=str(k), orientation="h", marker_color=colors[i % 8],
                                        hovertemplate="%{y}: %{x:.1f}%<extra>" + str(k) + "</extra>")
                                 for i, k in enumerate(classes) if k in rate.columns])
                fig.update_layout(barmode="stack")
                style(fig, max(300, 28 * len(rate) + 90))
                fig.update_xaxes(title="%", range=[0, 100])
                chart(fig, "t_f_cat")
            with st.expander("Tabla de contingencia", icon=":material/table:"):
                st.dataframe(ct.assign(total=ct.sum(1)))
    else:
        yv = pd.to_numeric(df[target], errors="coerce")
        if is_num:
            d = pd.DataFrame({f: pd.to_numeric(df[f], errors="coerce"), target: yv}).dropna()
            plot = sample_for_plot(d, 8000)
            with card(f"{target} frente a {f}", "Línea: tendencia LOWESS"):
                fig = go.Figure(go.Scattergl(x=plot[f], y=plot[target], mode="markers", name="datos",
                                             marker=dict(size=5, color=colors[0], opacity=0.4)))
                if len(d) > 10:
                    from statsmodels.nonparametric.smoothers_lowess import lowess
                    base = d.sample(min(len(d), 5000), random_state=0)
                    lw = lowess(base[target], base[f], frac=0.3)
                    fig.add_scatter(x=lw[:, 0], y=lw[:, 1], mode="lines", name="LOWESS",
                                    line=dict(color=tok()["text"], width=2))
                style(fig, 440)
                fig.update_xaxes(title=f)
                fig.update_yaxes(title=target)
                chart(fig, "t_f_scatter")
        else:
            x = top_with_other(df[f], 15)
            order = yv.groupby(x).median().sort_values(ascending=False).index
            with card(f"{target} por {f}", "Ordenado por mediana"):
                fig = go.Figure()
                for k in order:
                    for tr in box_traces(yv[x == k], str(k), colors[0] if k != "Otros" else tok()["subtle"],
                                         horizontal=False, max_outliers=200):
                        fig.add_trace(tr)
                style(fig, 440, legend=False)
                fig.update_yaxes(title=target)
                chart(fig, "t_f_box")


def _roc(ctx: Ctx, target: str) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    df = ctx.df
    classes = sorted(df[target].dropna().unique(), key=str)
    c1, c2, c3 = st.columns([1, 1, 3])
    pos = c1.selectbox("Clase positiva", classes, index=len(classes) - 1, key="t_roc_pos")
    y = (df[target] == pos).astype(int).where(df[target].notna())
    tbl = _rank_table(df, ctx.key, target, "Clasificación", ctx.numeric, ctx.categorical)
    num_rank = tbl[tbl["tipo"] == "numérica"].sort_values("AUC", ascending=False) if "AUC" in tbl else tbl.iloc[0:0]
    if not need(not num_rank.empty, "No hay variables numéricas para trazar curvas ROC."):
        return
    topn = c2.number_input("Curvas individuales", 1, 7, 4, key="t_roc_n")
    combo = c3.multiselect("Modelo logístico combinado (validación cruzada, 5 particiones)",
                           [c for c in ctx.numeric if c != target], default=num_rank["variable"].head(3).tolist(),
                           key="t_roc_combo")
    colors = pal()
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=tok()["subtle"], dash="dot", width=1))
    for i, r in enumerate(num_rank.head(int(topn)).itertuples()):
        d = pd.DataFrame({"x": df[r.variable], "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        auc = roc_auc_score(d["y"], d["x"])
        fpr, tpr, _ = roc_curve(d["y"], d["x"] if auc >= 0.5 else -d["x"])
        fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=colors[i], width=2),
                        name=f"{'↑' if auc >= 0.5 else '↓'} {r.variable} · AUC {max(auc, 1 - auc):.3f}")
    coef, auc_c = None, None
    if combo:
        scales = {}
        sc_cols = st.columns(min(len(combo), 6))
        for i, v in enumerate(combo):
            scales[v] = sc_cols[i % len(sc_cols)].selectbox(f"Escala {v}", SCALES, key=f"t_roc_s_{v}")
        X = apply_scales(df, scales)
        yy = y.loc[X.index].dropna()
        X = X.loc[yy.index]
        if len(X) >= 30 and yy.nunique() == 2 and yy.value_counts().min() >= 5:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            proba = cross_val_predict(model, X.values, yy.values, method="predict_proba",
                                      cv=StratifiedKFold(5, shuffle=True, random_state=0))[:, 1]
            auc_c = roc_auc_score(yy, proba)
            fpr, tpr, _ = roc_curve(yy, proba)
            fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=tok()["text"], width=3),
                            name=f"Combinado ({len(combo)} vars) · AUC validada {auc_c:.3f}")
            model.fit(X.values, yy.values)
            coef = pd.DataFrame({"variable": combo, "escala": [scales[v] for v in combo],
                                 "coef. estandarizado": model[-1].coef_[0]}).sort_values(
                "coef. estandarizado", key=abs, ascending=False)
        else:
            st.caption("No hay filas suficientes para el modelo combinado.")
    if auc_c is not None:
        kpi_row([dict(label="AUC mejor variable", value=f"{num_rank['AUC'].iloc[0]:.3f}"),
                 dict(label="AUC modelo combinado", value=f"{auc_c:.3f}",
                      delta=f"{auc_c - num_rank['AUC'].iloc[0]:+.3f}"),
                 dict(label="Filas del modelo", value=es_int(len(X)))])
    with card(f"Curvas ROC · {target} = {pos}", "Cada curva individual usa el valor bruto de la variable; el "
                                                "combinado se evalúa con validación cruzada (AUC sin sobreajuste)"):
        style(fig, 500)
        fig.update_xaxes(title="tasa de falsos positivos", range=[0, 1], constrain="domain")
        fig.update_yaxes(title="tasa de verdaderos positivos", range=[0, 1.01], scaleanchor="x", scaleratio=1)
        fig.update_layout(legend=dict(orientation="v", x=1.02, y=0, xanchor="left", yanchor="bottom"))
        chart(fig, "t_roc_fig")
    if coef is not None:
        with card("Coeficientes del modelo combinado", "Estandarizados: comparables entre variables"):
            st.dataframe(coef, hide_index=True,
                         column_config={"coef. estandarizado": st.column_config.NumberColumn(format="%.3f")})
