"""Informe HTML autocontenido con el resumen del EDA."""

import html
from datetime import datetime

import numpy as np
import pandas as pd

from .missing import missing_summary
from .outliers import outlier_overview
from .profiling import column_profile, dataset_summary, quality_alerts, quality_scores

CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:2rem auto;max-width:1100px;
     color:#1d1d1b;background:#fff;padding:0 1rem}
h1{font-size:1.6rem;margin-bottom:.2rem} h2{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid #ddd;
padding-bottom:.3rem} .muted{color:#77756f;font-size:.9rem}
table{border-collapse:collapse;font-size:.82rem;width:100%;margin:.5rem 0}
th,td{border:1px solid #e1e0d9;padding:4px 7px;text-align:left} th{background:#f4f3ef}
.kpis{display:flex;gap:1rem;flex-wrap:wrap}.kpi{border:1px solid #e1e0d9;border-radius:8px;padding:.6rem 1rem}
.kpi b{display:block;font-size:1.3rem}.Alta{color:#c62f2e;font-weight:600}.Media{color:#b07800;font-weight:600}
.Baja{color:#2a78d6}
@media (prefers-color-scheme: dark){body{background:#1b1b1a;color:#eee}th{background:#2a2a28}
th,td,.kpi{border-color:#3a3a37}}
"""


def _table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "<p class='muted'>Sin datos.</p>"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.4g}")
    out = d.to_html(index=False, escape=True, border=0)
    if "gravedad" in d.columns:
        for sev in ("Alta", "Media", "Baja"):
            out = out.replace(f"<td>{sev}</td>", f"<td class='{sev}'>{sev}</td>")
    return out


def build_report(df: pd.DataFrame, key: str, types: dict, numeric: list[str], prep_log: list,
                 title: str = "Informe EDA") -> str:
    alerts = quality_alerts(df, key, types, tuple(prep_log))
    summary = dataset_summary(df, key)
    scores = quality_scores(summary, alerts, df.columns)
    prof = column_profile(df, key, types).drop(columns=["distribución"], errors="ignore")
    miss = missing_summary(df, key)
    miss = miss[miss["nulos"] > 0]
    out_cols = [c for c in numeric if types.get(c) == "numérica"]
    outl = outlier_overview(df, key, out_cols) if out_cols else pd.DataFrame()
    corr_html = "<p class='muted'>Menos de 2 variables numéricas.</p>"
    if len(out_cols) >= 2:
        corr = df[out_cols].corr("spearman")
        up = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
        top = up.reindex(up.abs().sort_values(ascending=False).index).head(15).reset_index()
        top.columns = ["variable A", "variable B", "ρ de Spearman"]
        corr_html = _table(top)
    n_dup = summary["dup_rows"]
    kpis = {"Filas": f"{len(df):,}", "Columnas": df.shape[1],
            "Celdas nulas": f"{df.isna().mean().mean() * 100:.1f}%", "Filas duplicadas": f"{n_dup:,}",
            **{k: f"{v * 100:.1f}%" for k, v in scores.items()}}
    kpi_html = "".join(f"<div class='kpi'>{html.escape(str(k))}<b>{html.escape(str(v))}</b></div>"
                       for k, v in kpis.items())
    tc = pd.Series(types).value_counts().reset_index()
    tc.columns = ["tipo", "columnas"]
    prep = pd.DataFrame(prep_log).drop(columns="perdidos", errors="ignore") if prep_log else pd.DataFrame()
    outl_cols = ["variable", "n", "nº IQR", "% IQR (Tukey)", "% Z-score", "% Z-score robusto (MAD)",
                 "límite inf. IQR", "límite sup. IQR", "mín", "máx", "asimetría"]
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body>
<h1>{html.escape(title)}</h1><p class="muted">Generado el {datetime.now():%d/%m/%Y %H:%M} con el Explorador EDA.</p>
<h2>Resumen</h2><div class="kpis">{kpi_html}</div>
<h2>Tipos de columna</h2>{_table(tc)}
<h2>Problemas de calidad detectados ({len(alerts)})</h2>{_table(alerts)}
<h2>Valores faltantes</h2>{_table(miss)}
<h2>Outliers por variable</h2>{_table(outl[[c for c in outl_cols if c in outl.columns]] if not outl.empty else outl)}
<h2>Correlaciones más fuertes</h2>{corr_html}
<h2>Perfil de columnas</h2>{_table(prof, 500)}
<h2>Preparación aplicada</h2>{_table(prep)}
</body></html>"""
