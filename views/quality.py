"""Calidad de datos: alertas, duplicados, consistencia de texto, valores sospechosos y cambios."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eda.charts import hist_bar
from eda.coltypes import NULL_TOKENS, is_textlike, normalize_text, safe_nunique, uniques_with_counts
from eda.config import SEVERITY, pal, style
from eda.profiling import _dup_mask, dataset_summary, quality_alerts, quality_scores

from ._components import alert_list, card, kpi_row, page_header
from ._ui import Ctx, chart, download_csv, es_int, es_pct, insight, need, sub_nav


def render(ctx: Ctx) -> None:
    page_header("Datos", "Calidad de datos",
                "Revisión automática de los problemas típicos antes de analizar: duplicados, nulos "
                "disfrazados, códigos centinela, categorías mal escritas y tipos mezclados.")
    part = sub_nav(["Alertas", "Duplicados", "Consistencia de texto", "Valores sospechosos",
                    "Cambios aplicados"], key="q_nav")
    {"Alertas": _alerts, "Duplicados": _duplicates, "Consistencia de texto": _text,
     "Valores sospechosos": _suspicious, "Cambios aplicados": _changes}[part](ctx)


def _alerts(ctx: Ctx) -> None:
    df = ctx.df
    alerts = quality_alerts(df, ctx.key, ctx.types, tuple(ctx.prep_log))
    summ = dataset_summary(df, ctx.key)
    scores = quality_scores(summ, alerts, df.columns)
    counts = alerts["gravedad"].value_counts() if len(alerts) else pd.Series(dtype=int)
    kpi_row([dict(label=k, value=es_pct(v * 100)) for k, v in scores.items()] +
            [dict(label="Alertas", value=str(len(alerts)),
                  help=" · ".join(f"{s}: {int(counts.get(s, 0))}" for s in SEVERITY))])
    if alerts.empty:
        st.success("No se han detectado problemas.", icon=":material/verified:")
        return

    left, right = st.columns([2, 3], gap="medium")
    with left, card("Alertas por tipo", "Cada barra apilada por gravedad"):
        by = alerts.groupby(["problema", "gravedad"]).size().unstack(fill_value=0)
        by = by.reindex(columns=[s for s in SEVERITY if s in by.columns])
        by = by.loc[by.sum(axis=1).sort_values().index]
        fig = go.Figure()
        for sev in by.columns:
            color, icon, label = SEVERITY[sev]
            fig.add_bar(y=by.index, x=by[sev], name=f"{icon} {label}", orientation="h", marker_color=color,
                        hovertemplate="%{y}: %{x}<extra>" + label + "</extra>")
        fig.update_layout(barmode="stack")
        style(fig, max(240, 28 * len(by) + 80))
        fig.update_xaxes(title="nº de alertas", dtick=1 if by.values.max() < 10 else None)
        chart(fig, "q_alerts_bar")
    with right, card("Detalle"):
        f1, f2 = st.columns([1, 2])
        sev = f1.pills("Gravedad", list(SEVERITY), selection_mode="multi", default=list(SEVERITY),
                       key="q_sev", label_visibility="collapsed")
        probs = f2.multiselect("Tipo de problema", sorted(alerts["problema"].unique()), key="q_prob",
                               placeholder="Todos los tipos", label_visibility="collapsed")
        shown = alerts[alerts["gravedad"].isin(sev or list(SEVERITY))]
        if probs:
            shown = shown[shown["problema"].isin(probs)]
        view = st.segmented_control("Vista", ["Tarjetas", "Tabla"], default="Tarjetas", key="q_view",
                                    label_visibility="collapsed") or "Tarjetas"
        if view == "Tarjetas":
            with st.container(height=420, border=False):
                alert_list(shown.head(80).itertuples())
        else:
            st.dataframe(shown, hide_index=True, height=420,
                         column_config={"sugerencia": st.column_config.TextColumn(width="large")})
        download_csv(shown, "alertas_calidad.csv", "Descargar alertas", key="q_dl_alerts")


def _duplicates(ctx: Ctx) -> None:
    df = ctx.df
    with card("Buscar duplicados", "En todas las columnas o solo en una clave (p. ej. id de cliente, o "
                                   "cliente + fecha)"):
        key_cols = st.multiselect("Columnas clave", df.columns.tolist(), default=ctx.ids[:1],
                                  key="q_dup_keys", placeholder="Todas las columnas")
    subset = key_cols or None
    dmask = _dup_mask(df, subset, keep=False)
    n_extra = int(_dup_mask(df, subset, keep="first").sum())
    kpi_row([dict(label="Filas implicadas", value=es_int(dmask.sum())),
             dict(label="Copias sobrantes", value=es_int(n_extra),
                  help="Filas que sobrarían dejando una de cada grupo."),
             dict(label="% sobrante", value=es_pct(n_extra / max(len(df), 1) * 100, 2))])
    if not dmask.any():
        st.success("No hay duplicados con este criterio.", icon=":material/verified:")
        return
    dups = df[dmask]
    try:
        dups = dups.sort_values((key_cols or df.columns.tolist())[:5])
    except TypeError:
        pass
    if key_cols:
        conflict = int((~_dup_mask(dups, None, keep=False)).sum())
        if conflict:
            insight(f"**{es_int(conflict)} filas comparten clave pero difieren en otras columnas**: son "
                    "conflictos de datos, no simples copias.", "warn")
    with card("Filas duplicadas"):
        st.dataframe(dups.head(1000), height=360)
        download_csv(dups, "filas_duplicadas.csv", "Descargar duplicados", key="q_dl_dups")
    if key_cols:
        with card("Claves más repetidas"):
            grp = df.groupby(key_cols, dropna=False, observed=True).size()
            grp = grp[grp > 1].sort_values(ascending=False).head(20).reset_index(name="repeticiones")
            st.dataframe(grp, hide_index=True)


def _text(ctx: Ctx) -> None:
    df = ctx.df
    cands = [c for c in df.columns if is_textlike(df[c]) and safe_nunique(df[c]) <= 20000]
    if not need(bool(cands), "No hay columnas de texto que revisar."):
        return
    with card("Variantes de escritura",
              "Valores que solo difieren en mayúsculas, tildes o espacios (p. ej. «Madrid», «madrid », "
              "«MADRID»): normalmente son la misma categoría"):
        col = st.selectbox("Columna", cands, key="q_txt_col")
        u, counts = uniques_with_counts(df[col])
        vc = pd.Series(counts, index=u.values)
        norm = pd.Series(u.map(normalize_text).values, index=u.values)
        groups = []
        for k, members in norm.groupby(norm).groups.items():
            if len(members) > 1:
                groups.append(dict(forma_normalizada=k, variantes=len(members),
                                   valores=" | ".join(f"«{m}» ({vc[m]:,})" for m in members),
                                   filas=int(vc[list(members)].sum()),
                                   sugerido=max(members, key=lambda m: vc[m])))
        kpi_row([dict(label="Valores distintos", value=es_int(len(vc))),
                 dict(label="Distintos tras normalizar", value=es_int(norm.nunique())),
                 dict(label="Grupos con variantes", value=str(len(groups)))])
        if groups:
            st.dataframe(pd.DataFrame(groups).sort_values("filas", ascending=False), hide_index=True)
            st.caption("«sugerido» es la variante más frecuente de cada grupo: buena candidata para unificar.")
        else:
            st.success("No hay variantes de escritura en esta columna.", icon=":material/verified:")
    a, b = st.columns(2, gap="medium")
    with a, card("Longitud de los textos"):
        lens = np.repeat(u.str.len().to_numpy(), counts)
        fig = go.Figure(hist_bar(lens, bins=40, name="textos", color=pal()[0]))
        style(fig, 240)
        fig.update_xaxes(title="nº de caracteres")
        chart(fig, "q_txt_len")
    with b, card("Valores únicos", "Aparecen una sola vez: posibles erratas"):
        rare = vc[vc == 1]
        st.metric("Valores con una sola aparición", es_int(len(rare)), border=False)
        if len(rare):
            st.caption(", ".join(map(str, rare.index[:15])))


def _suspicious(ctx: Ctx) -> None:
    df = ctx.df
    with card("Nulos disfrazados", "Valores como «N/A», «-», «?» o «null» en columnas de texto"):
        rows = []
        for c in df.columns:
            if not is_textlike(df[c]):
                continue
            u, counts = uniques_with_counts(df[c])
            hit = u.str.strip().str.lower().isin(NULL_TOKENS).to_numpy()
            rows += [dict(columna=c, valor=repr(v), apariciones=int(n)) for v, n in zip(u[hit], counts[hit])]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("Añádelos en el panel lateral → Preparación → «Valores a tratar como nulo».")
        else:
            st.success("No se han encontrado.", icon=":material/verified:")

    with card("Acumulaciones en los extremos",
              "Valores muy repetidos justo en el mínimo o el máximo (posibles códigos como -999, 9999 o 0)"):
        rows = []
        for c in ctx.numeric:
            a = df[c].to_numpy(dtype=float, na_value=np.nan)
            a = a[np.isfinite(a)]
            if a.size < 20 or len(np.unique(a[:2000])) < 5:
                continue
            for v in {a.min(), a.max()}:
                share = float((a == v).mean() * 100)
                if share >= 1:
                    rest = a[a != v]
                    rows.append({"columna": c, "valor": v, "extremo": "mínimo" if v == a.min() else "máximo",
                                 "% filas": round(share, 2),
                                 "siguiente valor": (rest.min() if v == a.min() else rest.max()) if rest.size
                                 else np.nan})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            insight("Un valor que acumula muchas filas en el extremo puede ser real (0 compras) o un código "
                    "de «desconocido». Compáralo con el siguiente valor.")
        else:
            st.success("No hay acumulaciones sospechosas.", icon=":material/verified:")

    with card("Mezcla de números y texto"):
        rows = []
        for c in df.columns:
            if not is_textlike(df[c]):
                continue
            u, counts = uniques_with_counts(df[c])
            if u.empty:
                continue
            isnum = pd.to_numeric(u.str.replace(",", ".", regex=False).str.strip(), errors="coerce").notna()
            share = float(isnum.to_numpy() @ counts / counts.sum())
            if 0.05 <= share <= 0.95:
                rows.append({"columna": c, "% numéricos": round(share * 100, 1),
                             "ejemplos no numéricos": ", ".join(u[~isnum].head(5)),
                             "ejemplos numéricos": ", ".join(u[isnum].head(5))})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.success("No hay columnas con tipos mezclados.", icon=":material/verified:")


def _changes(ctx: Ctx) -> None:
    with card("Transformaciones aplicadas al cargar", "Configurables en el panel lateral → Preparación"):
        if ctx.prep_log:
            st.dataframe(pd.DataFrame(ctx.prep_log).drop(columns="perdidos", errors="ignore"), hide_index=True)
        else:
            st.caption("No se ha aplicado ningún cambio.")
    if ctx.load_info:
        with card("Lectura de archivos"):
            st.dataframe(pd.DataFrame(ctx.load_info).astype(str), hide_index=True)
