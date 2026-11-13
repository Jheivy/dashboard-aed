"""Valores faltantes: cuánto, dónde, con qué patrón, por qué mecanismo y cómo imputar."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from eda.charts import bin_edges, box_traces, hist_bar
from eda.config import STATUS, div_scale, outlier_color, pal, rgba, seq_scale, style
from eda.missing import (impute_preview, little_mcar_test, missing_dependence, missing_patterns,
                         missing_summary, nullity_correlation)
from eda.profiling import dataset_summary
from eda.stats_utils import cluster_order, fmt_p

from ._components import card, kpi_row, page_header
from ._ui import (NONE, Ctx, chart, download_csv, es_int, es_pct, finite, insight, need, sample_rows,
                  sub_nav)


def render(ctx: Ctx) -> None:
    page_header("Datos", "Valores faltantes",
                "Cuántos datos faltan, dónde se concentran, si faltan juntos, si su ausencia depende de "
                "otras variables (MCAR / MAR / MNAR) y qué efecto tendría imputarlos.")
    summ = missing_summary(ctx.df, ctx.key)
    with_na = summ[summ["nulos"] > 0]
    if with_na.empty:
        st.success("No hay valores faltantes en ninguna columna.", icon=":material/verified:")
        insight("Ojo: los nulos pueden estar disfrazados (−999, «N/A», «-»…). Revisa **Calidad de datos → "
                "Valores sospechosos**.")
        return
    part = sub_nav(["Resumen", "Matriz y patrones", "Correlación de nulidad", "Mecanismo (MCAR/MAR)",
                    "Imputación"], key="m_nav")
    {"Resumen": _summary, "Matriz y patrones": _patterns, "Correlación de nulidad": _nullcorr,
     "Mecanismo (MCAR/MAR)": _mechanism, "Imputación": _imputation}[part](ctx, summ, with_na)


def _summary(ctx: Ctx, summ: pd.DataFrame, with_na: pd.DataFrame) -> None:
    df = ctx.df
    ds = dataset_summary(df, ctx.key)
    per_row = df.isna().sum(axis=1)
    kpi_row([
        dict(label="Celdas vacías", value=es_int(ds["null_cells"])),
        dict(label="% de celdas", value=es_pct(ds["null_pct"], 2)),
        dict(label="Columnas con nulos", value=f"{len(with_na)} / {df.shape[1]}"),
        dict(label="Filas con algún nulo", value=es_pct(100 - ds["complete_rows_pct"])),
        dict(label="Filas completas", value=es_int((per_row == 0).sum()),
             help="Lo que quedaría con listwise deletion (dropna)."),
    ])
    a, b = st.columns([3, 2], gap="medium")
    with a, card("% de valores faltantes por columna", "Rojo ≥ 50% · naranja 5–50% · azul < 5%"):
        d = with_na.head(40)
        colors = [STATUS["critical"] if v >= 50 else STATUS["serious"] if v >= 5 else pal()[0]
                  for v in d["% nulos"]]
        fig = go.Figure(go.Bar(x=d["% nulos"], y=d["columna"], orientation="h", marker_color=colors,
                               text=[es_pct(v) for v in d["% nulos"]], textposition="outside",
                               customdata=d["nulos"],
                               hovertemplate="%{y}<br>%{x:.1f}% · %{customdata:,} nulos<extra></extra>"))
        style(fig, max(240, 24 * len(d) + 60), legend=False)
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_xaxes(range=[0, 112], title="% nulos")
        chart(fig, "m_bar")
    with b, card("Nulos por fila", "¿Cuántas columnas vacías tiene cada fila?"):
        dist = per_row.value_counts().sort_index()
        fig = go.Figure(go.Bar(x=dist.index, y=dist.values, marker_color=pal()[1],
                               hovertemplate="%{x} nulos: %{y:,} filas<extra></extra>"))
        style(fig, 230)
        fig.update_xaxes(title="nulos en la fila", dtick=1 if len(dist) < 15 else None)
        chart(fig, "m_row_hist")
        thr = st.slider("Filas con al menos este % de columnas vacías", 10, 100, 50, 5, key="m_row_thr")
        bad = per_row / df.shape[1] * 100 >= thr
        st.caption(f"{es_int(bad.sum())} filas superan el {thr}%.")
        if bad.any():
            download_csv(lambda: df[bad], "filas_muy_incompletas.csv", "Descargar esas filas", key="m_dl_bad")

    with card("Completitud según las columnas que necesites"):
        pick = st.multiselect("Si tu análisis usa estas columnas…", with_na["columna"].tolist(),
                              default=with_na["columna"].head(3).tolist(), key="m_need_cols")
        if pick:
            complete = df[pick].notna().all(axis=1)
            st.markdown(f"…te quedarían **{es_int(complete.sum())}** filas completas "
                        f"(**{es_pct(complete.mean() * 100)}**) eliminando las incompletas.")

    opts = ([ctx.source_col] if ctx.source_col else []) + [
        c for c in ctx.categorical if c != ctx.source_col and df[c].nunique() <= 30]
    if opts:
        with card("% de nulos por grupo", "Si una columna tiene muchos más nulos en un grupo, no faltan al "
                                          "azar (no es MCAR)"):
            g = st.selectbox("Agrupar por", opts, key="m_group")
            cols = with_na["columna"].head(30).tolist()
            tab = df[cols].isna().groupby(df[g].astype(str)).mean() * 100
            fig = go.Figure(go.Heatmap(z=tab.values, x=tab.columns, y=tab.index, colorscale=seq_scale(),
                                       zmin=0, text=np.round(tab.values, 1), texttemplate="%{text}",
                                       hovertemplate="%{y} · %{x}: %{z:.1f}%<extra></extra>",
                                       colorbar=dict(title="% nulos", thickness=10)))
            style(fig, max(240, 30 * len(tab) + 110))
            fig.update_xaxes(tickangle=-35)
            chart(fig, "m_group_heat")
    with st.expander("Tabla completa de nulos por columna", icon=":material/table:"):
        st.dataframe(summ, hide_index=True, column_config={
            "% nulos": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})


def _patterns(ctx: Ctx, summ: pd.DataFrame, with_na: pd.DataFrame) -> None:
    df = ctx.df
    cols = with_na["columna"].tolist()[:40]
    with card("Matriz de nulos", "Cada columna de píxeles es una fila del dataset; rojo = valor faltante"):
        c1, c2, c3 = st.columns(3)
        sort_by = c1.selectbox("Ordenar filas por", [NONE] + df.columns.tolist(), key="m_sort",
                               help="Ordenar por una fecha o variable muestra si los nulos se concentran en "
                                    "un tramo (p. ej. un periodo sin registro).")
        n_rows = c2.slider("Filas (muestra)", 100, 3000, 800, 100, key="m_mat_rows")
        only_na = c3.toggle("Solo columnas con nulos", True, key="m_only_na")
        show_cols = cols if only_na else df.columns.tolist()[:60]
        smp = sample_rows(df, n_rows)
        try:
            smp = smp.sort_values(sort_by) if sort_by != NONE else smp.sort_index()
        except TypeError:
            pass
        mat = smp[show_cols].isna().astype(np.int8).T
        base = rgba(pal()[0], 0.12)
        fig = go.Figure(go.Heatmap(z=mat.values, y=mat.index, colorscale=[[0, base], [1, outlier_color()]],
                                   showscale=False, hovertemplate="%{y}<extra></extra>"))
        style(fig, max(240, 22 * len(show_cols) + 50))
        fig.update_xaxes(showticklabels=False, title="filas" + (f" ordenadas por {sort_by}" if sort_by != NONE
                                                                else " (muestra)"))
        chart(fig, "m_matrix")

    with card("Patrones de nulos más frecuentes",
              "Cada fila es una combinación de columnas vacías; la barra indica cuántas filas la tienen"):
        pcols = cols[:25]
        pats = missing_patterns(df, ctx.key, pcols, 15)
        labels = [f"patrón {i + 1}" for i in range(len(pats))]
        fig = make_subplots(rows=1, cols=2, column_widths=[0.72, 0.28], shared_yaxes=True,
                            horizontal_spacing=0.02)
        fig.add_trace(go.Heatmap(z=pats[pcols].values, x=pcols, y=labels, showscale=False, xgap=2, ygap=2,
                                 colorscale=[[0, rgba(pal()[0], 0.12)], [1, outlier_color()]],
                                 hovertemplate="%{x}: %{z}<extra></extra>"), 1, 1)
        fig.add_trace(go.Bar(x=pats["% filas"], y=labels, orientation="h", marker_color=pal()[0],
                             text=[es_int(v) for v in pats["filas"]], textposition="auto",
                             hovertemplate="%{x:.1f}% de las filas<extra></extra>", showlegend=False), 1, 2)
        style(fig, max(280, 28 * len(pats) + 130))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickangle=-40, row=1, col=1)
        fig.update_xaxes(title="% filas", row=1, col=2)
        chart(fig, "m_patterns")
        st.caption("Si unas pocas combinaciones concentran casi todos los nulos, suelen tener una causa común "
                   "(un formulario opcional, otra fuente, un periodo sin registro).")


def _nullcorr(ctx: Ctx, summ: pd.DataFrame, with_na: pd.DataFrame) -> None:
    cols = with_na[with_na["% nulos"] < 100]["columna"].tolist()[:50]
    if not need(len(cols) >= 2, "Hacen falta al menos 2 columnas con nulos (y no vacías del todo)."):
        return
    corr = nullity_correlation(ctx.df, ctx.key, cols).fillna(0)
    order = cluster_order(corr)
    corr = corr.iloc[order, order]
    masked = corr.mask(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    with card("Correlación de nulidad", "+1: faltan juntas · −1: cuando falta una, la otra está · 0: "
                                        "independientes. Ordenada por clusters"):
        fig = go.Figure(go.Heatmap(z=masked.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                   colorscale=div_scale(), text=np.round(masked.values, 2),
                                   texttemplate="%{text}" if len(cols) <= 20 else None,
                                   hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>",
                                   colorbar=dict(thickness=10)))
        style(fig, max(360, 26 * len(cols) + 110))
        fig.update_yaxes(autorange="reversed")
        chart(fig, "m_nullcorr")
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    top = upper.reindex(upper.abs().sort_values(ascending=False).index).head(10)
    if len(top):
        with card("Pares más relacionados"):
            st.dataframe(pd.DataFrame({"columna A": [a for a, _ in top.index], "columna B": [b for _, b in top.index],
                                       "correlación de nulidad": top.values.round(3)}), hide_index=True)


def _mechanism(ctx: Ctx, summ: pd.DataFrame, with_na: pd.DataFrame) -> None:
    df = ctx.df
    with st.expander("¿Qué son MCAR, MAR y MNAR?", icon=":material/school:"):
        st.markdown(
            "- **MCAR** (completamente al azar): la ausencia no depende de nada. Eliminar filas no sesga, "
            "solo reduce tamaño.\n"
            "- **MAR** (al azar condicionado): depende de *otras* variables observadas (p. ej. faltan más "
            "ingresos en un canal). Imputar usando esas variables (KNN, regresión, imputación múltiple) "
            "corrige el sesgo.\n"
            "- **MNAR** (no al azar): depende del propio valor que falta (quien gana mucho no lo declara). "
            "No se puede demostrar con los datos; requiere conocimiento del dominio.")

    with card("1 · Test MCAR de Little", "H₀: los datos faltan completamente al azar (variables numéricas)"):
        if any(df[c].isna().any() for c in ctx.numeric):
            cols = st.multiselect("Variables numéricas incluidas", ctx.numeric, default=ctx.numeric[:15],
                                  key="m_little_cols", max_selections=25)
            res = little_mcar_test(df, ctx.key, cols) if len(cols) >= 2 else None
            if res is None:
                st.caption("No aplicable con esta selección (≥ 2 variables con variación, alguna con nulos).")
            else:
                kpi_row([dict(label="χ²", value=f"{res['chi2']:,.1f}".replace(",", ".")),
                         dict(label="Grados de libertad", value=str(res["dof"])),
                         dict(label="p-valor", value=fmt_p(res["p"])),
                         dict(label="Patrones de nulos", value=str(res["n_patterns"]))])
                if res["p"] < 0.05:
                    insight("**Se rechaza MCAR**: la falta de datos no es completamente aleatoria. Eliminar "
                            "filas incompletas probablemente sesgue los resultados; mira abajo de qué "
                            "variables depende.", "warn")
                else:
                    insight("**No hay evidencia contra MCAR.** No rechazar no prueba que sea MCAR, y con "
                            "pocos datos el test tiene poca potencia.", "ok")
                st.caption(f"n = {es_int(res['n'])} filas · medias y covarianzas estimadas por EM bajo "
                           "normalidad multivariante (aproximado con variables muy asimétricas).")
        else:
            st.caption("Ninguna variable numérica tiene nulos; el test de Little no aplica.")

    with card("2 · ¿De qué depende que falte una columna?",
              "Compara cada variable entre las filas donde la columna falta y donde está"):
        target = st.selectbox("Columna con nulos", with_na[with_na["% nulos"] < 100]["columna"].tolist(),
                              key="m_dep_col")
        res = missing_dependence(df, ctx.key, target, ctx.numeric, ctx.categorical)
        if res.empty:
            st.caption("No hay suficientes datos para comparar.")
            return
        related = res[res["relacionada"]]
        if len(related):
            names = ", ".join(f"**{v}**" for v in related["variable"].head(5))
            insight(f"La ausencia de **{target}** está relacionada con {names}. Indica **MAR** (o MNAR): no "
                    "elimines filas sin más; imputa usando estas variables o modela la ausencia.", "warn")
        else:
            insight(f"La ausencia de **{target}** no se asocia de forma relevante con ninguna variable "
                    "observada: compatible con MCAR (no descarta MNAR).", "ok")
        show = res.copy()
        show["p"] = show["p"].map(fmt_p)
        show["p ajustado (Holm)"] = show["p ajustado (Holm)"].map(fmt_p)
        st.dataframe(show, hide_index=True, column_config={
            "efecto": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "relacionada": st.column_config.CheckboxColumn()})
        st.caption("Relacionada si p ajustado (Holm) < 0,05 **y** efecto no trivial (|dif. estandarizada| ≥ "
                   "0,2 o V de Cramér ≥ 0,1). Con muestras grandes casi todo es significativo: mira el efecto.")

        pick = st.selectbox("Comparar una variable entre filas con y sin nulo",
                            [v for v in res["variable"] if not v.startswith("nulos de ")], key="m_dep_var")
        if pick:
            miss = df[target].isna()
            names = (f"{target} falta", f"{target} presente")
            colors = (outlier_color(), pal()[0])
            fig = go.Figure()
            if pick in ctx.numeric:
                for nm, m, col in zip(names, (miss, ~miss), colors):
                    for tr in box_traces(df.loc[m, pick], nm, col):
                        fig.add_trace(tr)
                fig.update_xaxes(title=pick)
            else:
                lab = df[pick].astype(object).where(df[pick].notna(), "(nulo)").astype(str)
                for nm, m, col in zip(names, (miss, ~miss), colors):
                    pct = lab[m].value_counts(normalize=True).head(15) * 100
                    fig.add_bar(y=pct.index, x=pct.values, name=nm, orientation="h", marker_color=col,
                                hovertemplate="%{y}: %{x:.1f}%<extra>" + nm + "</extra>")
                fig.update_layout(barmode="group")
                fig.update_xaxes(title="% de filas")
            style(fig, 320, f"{pick}: filas con {target} faltante vs presente")
            chart(fig, "m_dep_plot")


IMPUTE_NUM = ["Media", "Mediana", "Moda", "Valor constante", "Interpolación lineal",
              "Arrastrar anterior (ffill)", "KNN (vecinos más cercanos)"]
IMPUTE_CAT = ["Moda", "Valor constante", "Arrastrar anterior (ffill)"]


def _imputation(ctx: Ctx, summ: pd.DataFrame, with_na: pd.DataFrame) -> None:
    df = ctx.df
    pct = with_na.set_index("columna")["% nulos"]
    cands = [c for c in with_na["columna"] if (c in ctx.numeric or c in ctx.categorical) and pct[c] < 100]
    if not need(bool(cands), "No hay columnas numéricas o categóricas con nulos para imputar."):
        return
    insight("Las imputaciones simples (media, mediana, moda) **reducen la varianza** y debilitan las "
            "relaciones; KNN usa las demás variables y suele preservar mejor la distribución.")
    c1, c2, c3 = st.columns(3)
    col = c1.selectbox("Columna", cands, key="m_imp_col")
    is_num = col in ctx.numeric
    method = c2.selectbox("Método", IMPUTE_NUM if is_num else IMPUTE_CAT, key="m_imp_method")
    const, k = None, 5
    if method == "Valor constante":
        const = c3.number_input("Valor", value=0.0, key="m_imp_const") if is_num else \
            c3.text_input("Valor", "Desconocido", key="m_imp_const_t")
    elif method.startswith("KNN"):
        k = c3.slider("Vecinos (k)", 1, 30, 5, key="m_imp_k")

    work, wkey = df, ctx.key
    if method.startswith("KNN") and len(df) > 20000:
        work, wkey = df.sample(20000, random_state=0), ctx.key + ":s20k"
        st.caption("KNN calculado sobre una muestra de 20.000 filas.")
    imputed = impute_preview(work, wkey, col, method, ctx.numeric, const, k)
    was_na = work[col].isna()

    if is_num:
        orig, new = finite(work[col]), finite(imputed)
        tbl = pd.DataFrame({"original (sin nulos)": [len(orig), orig.mean(), orig.median(), orig.std(), orig.skew()],
                            "tras imputar": [len(new), new.mean(), new.median(), new.std(), new.skew()]},
                           index=["n", "media", "mediana", "desv. típica", "asimetría"])
        tbl["cambio %"] = (tbl.iloc[:, 1] / tbl.iloc[:, 0] - 1) * 100
        a, b = st.columns([3, 2], gap="medium")
        with a, card(f"{col}: observado vs imputado", "Densidad; las barras rojas son los valores rellenados"):
            edges = bin_edges([orig.to_numpy(), finite(imputed[was_na]).to_numpy()], 50)
            fig = go.Figure([hist_bar(orig, edges, "observados", pal()[0], density=True, opacity=0.75,
                                      showlegend=True),
                             hist_bar(imputed[was_na], edges, "imputados", outlier_color(), density=True,
                                      opacity=0.75, showlegend=True)])
            fig.update_layout(barmode="overlay")
            style(fig, 320)
            chart(fig, "m_imp_hist")
        with b, card("Efecto en los estadísticos"):
            st.dataframe(tbl, column_config={c: st.column_config.NumberColumn(format="%.3f") for c in tbl.columns})
            if tbl.loc["desv. típica", "cambio %"] < -5:
                insight("La desviación típica cae más de un 5%: este método encoge la variabilidad real.", "warn")
            others = [c for c in ctx.numeric if c != col][:6]
            if others:
                corr_o = work[[col] + others].corr()[col].drop(col)
                corr_i = work[others].assign(**{col: imputed}).corr()[col].drop(col)
                st.markdown("**Correlación con otras variables**")
                st.dataframe(pd.DataFrame({"original": corr_o, "tras imputar": corr_i}).round(3))
    else:
        before = work[col].astype(object).where(work[col].notna(), "(nulo)").astype(str).value_counts(normalize=True)
        after = imputed.astype(object).where(imputed.notna(), "(nulo)").astype(str).value_counts(normalize=True)
        comp = pd.DataFrame({"original %": before * 100, "tras imputar %": after * 100}).fillna(0)
        with card("Distribución antes y después"):
            st.dataframe(comp.sort_values("tras imputar %", ascending=False).head(20).round(2))
    st.caption("Vista previa: no modifica los datos del resto del dashboard.")
    if len(work) == len(df):
        download_csv(lambda: df.assign(**{col: imputed}), f"datos_imputado_{col}.csv",
                     "Descargar dataset con esta columna imputada", key="m_dl_imp")
