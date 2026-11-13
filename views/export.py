"""Informe y exportación de los datos preparados (descargas diferidas: se generan al pulsar)."""

import io

import pandas as pd
import streamlit as st

from eda.report import build_report

from ._components import card, page_header
from ._ui import Ctx, es_int

EXCEL_WARN_ROWS = 50_000


def render(ctx: Ctx) -> None:
    page_header("Salida", "Informe y exportación",
                "Un informe HTML con el resumen del análisis para compartir o archivar, y los datos tal y como los "
                "estás analizando (con preparación, tipos y filtros aplicados).")
    df = ctx.df
    a, b = st.columns(2, gap="medium")
    with a, card("Informe HTML", "Métricas de calidad, alertas, nulos, outliers, correlaciones y perfil de columnas. "
                                 "Se abre en cualquier navegador"):
        title = st.text_input("Título", "Informe de análisis exploratorio", key="x_title")
        st.download_button("Descargar informe", lambda: build_report(df, ctx.key, ctx.types, ctx.numeric,
                                                                     ctx.prep_log, title).encode("utf-8"),
                           "informe_eda.html", "text/html", key="x_dl_report", type="primary",
                           icon=":material/description:", on_click="ignore")
    with b, card("Datos preparados"):
        notes = [f"{es_int(len(df))} filas × {df.shape[1]} columnas"]
        if ctx.prep_log:
            notes.append(f"{len(ctx.prep_log)} ajustes de preparación")
        if ctx.n_filtered < ctx.n_loaded:
            notes.append(f"filtros activos ({es_int(ctx.n_filtered)} de {es_int(ctx.n_loaded)})")
        if ctx.sampled:
            notes.append("muestra aleatoria por rendimiento")
        st.caption(" · ".join(notes))
        c1, c2, c3 = st.columns(3)
        c1.download_button("CSV", lambda: df.to_csv(index=False).encode("utf-8-sig"), "datos_preparados.csv",
                           "text/csv", key="x_csv", icon=":material/download:", on_click="ignore", width="stretch")
        c2.download_button("Parquet", lambda: _parquet(df), "datos_preparados.parquet", "application/octet-stream",
                           key="x_pq", icon=":material/download:", on_click="ignore", width="stretch")
        c3.download_button("Excel", lambda: _excel(df), "datos_preparados.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="x_xlsx",
                           icon=":material/download:", on_click="ignore", width="stretch")
        if len(df) > EXCEL_WARN_ROWS:
            st.caption(f"Con {es_int(len(df))} filas, Excel puede tardar minutos en generarse: CSV o Parquet son "
                       "mucho más rápidos.")
    with card("Diccionario de tipos"):
        st.dataframe(pd.DataFrame({"columna": list(ctx.types), "tipo": list(ctx.types.values()),
                                   "dtype": [str(df[c].dtype) for c in ctx.types]}), hide_index=True)


def _parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    out = df.copy()
    for c in out.columns:  # mixed-type object columns can't be written by Arrow
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: v if v is None or isinstance(v, str) else str(v))
    out.to_parquet(buf, index=False)
    return buf.getvalue()


def _excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    out = df.copy()
    for c in out.select_dtypes(include=["datetimetz"]).columns:
        out[c] = out[c].dt.tz_localize(None)
    out.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()
