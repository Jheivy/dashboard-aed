"""Perfil por columna, resumen del dataset y detección automática de problemas de calidad.

Todas las funciones cacheadas reciben el DataFrame como `_df` (no se hashea) y una `key`
que lo identifica: así un clic no re-hashea 1M de filas para encontrar el resultado en caché.
"""

import numpy as np
import pandas as pd
import streamlit as st

from .coltypes import (NULL_TOKENS, has_unhashable, is_textlike, normalize_text, safe_nunique,
                       uniques_with_counts)

SENTINELS = [-999999, -99999, -9999, -999, -99, 99, 999, 9999, 99999, 999999]
NONNEG_HINTS = ("edad", "age", "precio", "price", "importe", "cantidad", "count", "num_",
                "n_", "peso", "weight", "altura", "height", "poblacion", "población",
                "duracion", "duración", "salario", "ingreso", "coste", "costo")
SEV_ORDER = {"Alta": 0, "Media": 1, "Baja": 2}
ALERT_COLS = ["gravedad", "columna", "problema", "detalle", "sugerencia"]


def _dup_mask(df: pd.DataFrame, subset=None, keep="first") -> pd.Series:
    try:
        return df.duplicated(subset=subset, keep=keep)
    except TypeError:
        return df.astype(str).duplicated(subset=subset, keep=keep)


@st.cache_data(show_spinner=False, max_entries=16)
def dataset_summary(_df: pd.DataFrame, key: str) -> dict:
    """Everything the headers/KPIs need, computed once per dataset state."""
    df = _df
    na = df.isna()
    return dict(
        rows=len(df), cols=df.shape[1],
        null_cells=int(na.to_numpy().sum()), null_pct=float(na.to_numpy().mean() * 100) if df.size else 0.0,
        complete_rows_pct=float((~na.any(axis=1)).mean() * 100) if len(df) else 0.0,
        dup_rows=int(_dup_mask(df).sum()),
        mem_mb=float(df.memory_usage(deep=len(df) <= 2_000_000).sum() / 1024 ** 2),
        nulls_by_col=na.sum().to_dict(),
    )


def _hist_counts(x: pd.Series, bins: int = 20) -> list[int]:
    a = x.to_numpy(dtype=float, na_value=np.nan)
    a = a[np.isfinite(a)]
    if a.size == 0 or a.min() == a.max():
        return []
    return np.histogram(a, bins=bins)[0].tolist()


@st.cache_data(show_spinner=False, max_entries=16)
def column_profile(_df: pd.DataFrame, key: str, types: dict) -> pd.DataFrame:
    df = _df
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        n_valid = int(s.notna().sum())
        nu = safe_nunique(s)
        row = {"columna": c, "tipo": types.get(c, ""), "dtype": str(s.dtype), "no nulos": n_valid,
               "% nulos": (1 - n_valid / n) * 100 if n else 0.0,
               "únicos": nu, "% únicos": nu / n_valid * 100 if n_valid else 0.0}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(x):
                row.update({"media": x.mean(), "mediana": x.median(),
                            "desv. típica": x.std() if len(x) > 1 else np.nan,
                            "mín": x.min(), "máx": x.max(),
                            "asimetría": x.skew() if len(x) > 2 else np.nan,
                            "% ceros": (x == 0).mean() * 100, "distribución": _hist_counts(x)})
        elif pd.api.types.is_datetime64_any_dtype(s):
            nn = s.dropna()
            if len(nn):
                row.update({"rango de fechas": f"{str(nn.min())[:10]} → {str(nn.max())[:10]}",
                            "distribución": _hist_counts(nn.astype("int64"))})
        elif not has_unhashable(s):
            vc = s.value_counts()
            vc = vc[vc > 0]
            if len(vc):
                row.update({"moda": str(vc.index[0])[:40], "% moda": vc.iloc[0] / n_valid * 100,
                            "distribución": vc.head(15).tolist()})
        row.setdefault("distribución", [])
        first = s.first_valid_index()
        row["ejemplo"] = str(s.loc[first])[:50] if first is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def _alert(sev, col, problem, detail, suggestion, **extra):
    return dict(gravedad=sev, columna=col, problema=problem, detalle=detail,
                sugerencia=suggestion, **extra)


@st.cache_data(show_spinner="Revisando la calidad de los datos…", max_entries=16)
def quality_alerts(_df: pd.DataFrame, key: str, types: dict, conv_log: tuple = ()) -> pd.DataFrame:
    df = _df
    alerts = []
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=ALERT_COLS)

    # ---- filas duplicadas
    n_dup = int(_dup_mask(df).sum())
    if n_dup:
        pct = n_dup / n * 100
        alerts.append(_alert("Alta" if pct > 5 else "Media", "(todas)", "Filas duplicadas",
                             f"{n_dup:,} filas idénticas a otra ({pct:.1f}%)",
                             "Revisa si son registros repetidos por error; puedes eliminarlos en "
                             "'Preparación' de la barra lateral."))

    # ---- columnas duplicadas (hash del contenido, confirmado con equals)
    hashes = {}
    for c in df.columns:
        try:
            h = int(pd.util.hash_pandas_object(df[c], index=False).sum())
        except TypeError:
            continue
        if h in hashes and df[c].equals(df[hashes[h]]):
            alerts.append(_alert("Media", c, "Columna duplicada", f"Contenido idéntico a «{hashes[h]}»",
                                 "Elimina una de las dos para no contar la información dos veces."))
        else:
            hashes.setdefault(h, c)

    num_cols = []
    for c in df.columns:
        s = df[c]
        t = types.get(c, "")
        n_valid = int(s.notna().sum())
        miss = (1 - n_valid / n) * 100

        if miss >= 50:
            alerts.append(_alert("Alta", c, "Muchos valores faltantes", f"{miss:.1f}% nulos",
                                 "Valora descartar la columna o analizar por qué falta (sección "
                                 "Valores faltantes)."))
        elif miss >= 5:
            alerts.append(_alert("Media", c, "Valores faltantes", f"{miss:.1f}% nulos",
                                 "Estudia el mecanismo (MCAR/MAR) antes de imputar o eliminar filas."))
        elif miss > 0:
            alerts.append(_alert("Baja", c, "Algunos valores faltantes", f"{miss:.2f}% nulos",
                                 "Impacto probablemente pequeño."))

        if t == "constante":
            alerts.append(_alert("Media", c, "Columna constante o vacía",
                                 f"{safe_nunique(s)} valor(es) distinto(s)",
                                 "No aporta información; puedes ignorarla."))
            continue
        if has_unhashable(s) or n_valid == 0:
            continue
        if t == "identificador":
            alerts.append(_alert("Baja", c, "Identificador", "Todos (o casi todos) los valores son distintos",
                                 "Se excluye de los análisis; úsalo para buscar duplicados por clave."))

        if is_textlike(s):
            u, counts = uniques_with_counts(s)
            if counts.sum() and counts.max() / counts.sum() >= 0.95 and t != "booleana":
                alerts.append(_alert("Baja", c, "Casi constante",
                                     f"El valor más frecuente ocupa el {counts.max() / counts.sum() * 100:.1f}%",
                                     "Aporta muy poca variabilidad."))
            low = u.str.strip().str.lower()
            fake = low.isin(NULL_TOKENS).to_numpy()
            if fake.any():
                vals = ", ".join(f"«{v}»" for v in u[fake].head(5))
                alerts.append(_alert("Media", c, "Nulos disfrazados",
                                     f"{int(counts[fake].sum()):,} valores tipo {vals}",
                                     "Márcalos como nulos en 'Preparación → Valores a tratar como nulo'."))
            ws = (u != u.str.strip()).to_numpy()
            if ws.any():
                alerts.append(_alert("Media", c, "Espacios sobrantes",
                                     f"{int(counts[ws].sum()):,} valores con espacios al inicio/final",
                                     "Activa 'Recortar espacios' en Preparación."))
            if t in ("categórica", "booleana") and len(u) <= 5000:
                norm = u.map(normalize_text)
                groups = u.groupby(norm.values).agg(list)
                groups = groups[groups.map(len) > 1]
                if len(groups):
                    ex = " / ".join(f"«{v}»" for v in groups.iloc[0][:4])
                    alerts.append(_alert("Media", c, "Categorías inconsistentes",
                                         f"{len(groups)} grupos de variantes (mayúsculas, tildes o espacios), "
                                         f"p. ej. {ex}",
                                         "Unifica la escritura; ver 'Calidad → Consistencia de texto'."))
            isnum = pd.to_numeric(u.str.replace(",", ".", regex=False).str.strip(), errors="coerce").notna()
            share = float(isnum.to_numpy() @ counts / counts.sum())
            if 0.1 <= share <= 0.9:
                alerts.append(_alert("Media", c, "Mezcla de números y texto",
                                     f"{share * 100:.0f}% de los valores son numéricos",
                                     "Revisa si es una columna numérica con valores erróneos."))
            if t == "categórica" and len(u) > 50:
                alerts.append(_alert("Baja", c, "Alta cardinalidad", f"{len(u):,} categorías",
                                     "Agrupa las categorías poco frecuentes en 'Otros' para analizar."))

        if pd.api.types.is_numeric_dtype(s) and t in ("numérica", "booleana"):
            a = s.to_numpy(dtype=float, na_value=np.nan)
            n_inf = int(np.isinf(a).sum())
            if n_inf:
                alerts.append(_alert("Alta", c, "Valores infinitos", f"{n_inf:,} valores ±inf",
                                     "Suelen venir de divisiones entre cero; corrígelos o trátalos como nulos."))
            a = a[np.isfinite(a)]
            if t == "numérica" and a.size > 10:
                num_cols.append(c)
                q1, q3 = np.percentile(a, [25, 75])
                iqr = q3 - q1
                amin, amax = a.min(), a.max()
                vals, cnts = np.unique(a[(a == amin) | (a == amax)], return_counts=True)
                for v, cnt in zip(vals, cnts):
                    if v in SENTINELS and cnt >= max(3, 0.001 * a.size) and iqr > 0 and \
                            (v < q1 - 3 * iqr or v > q3 + 3 * iqr):
                        alerts.append(_alert("Alta", c, "Posible código de valor faltante",
                                             f"El valor {v:g} aparece {cnt:,} veces en el extremo",
                                             f"Si {v:g} significa 'sin dato', añádelo a 'Valores a tratar "
                                             "como nulo'."))
                if iqr > 0:
                    ext = float(((a < q1 - 3 * iqr) | (a > q3 + 3 * iqr)).mean() * 100)
                    if ext >= 1:
                        alerts.append(_alert("Baja", c, "Outliers extremos", f"{ext:.1f}% de valores fuera de 3·IQR",
                                             "Revísalos en la sección Outliers."))
                sk = pd.Series(a).skew()
                if abs(sk) > 2 and len(np.unique(a[:5000])) > 10:
                    alerts.append(_alert("Baja", c, "Distribución muy asimétrica", f"asimetría = {sk:.2f}",
                                         "Considera escala logarítmica y estadísticos robustos."))
                zeros = float((a == 0).mean() * 100)
                if zeros >= 50:
                    alerts.append(_alert("Baja", c, "Muchos ceros", f"{zeros:.0f}% de ceros",
                                         "Puede ser una variable 'inflada en ceros'; analiza ceros y no-ceros "
                                         "por separado."))
                neg = int((a < 0).sum())
                if neg and any(h in c.lower() for h in NONNEG_HINTS):
                    alerts.append(_alert("Media", c, "Negativos sospechosos",
                                         f"{neg:,} valores negativos en una variable que por su nombre "
                                         "debería ser ≥ 0", "Comprueba si son errores."))
            if t == "booleana" and a.size:
                _, cnts = np.unique(a, return_counts=True)
                minority = cnts.min() / cnts.sum() * 100
                if minority < 5:
                    alerts.append(_alert("Baja", c, "Clases desbalanceadas",
                                         f"La clase minoritaria es el {minority:.1f}%",
                                         "Si es tu variable objetivo, usa métricas como AUC o F1."))

        if pd.api.types.is_datetime64_any_dtype(s):
            nn = s.dropna()
            try:
                tz = nn.dt.tz
                now = pd.Timestamp.now(tz=tz)
                fut = int((nn > now + pd.Timedelta(days=365)).sum())
                old = int((nn < pd.Timestamp("1900-01-01", tz=tz)).sum())
            except (TypeError, ValueError):
                fut = old = 0
            if fut or old:
                alerts.append(_alert("Media", c, "Fechas fuera de rango",
                                     f"{fut:,} en el futuro lejano · {old:,} anteriores a 1900",
                                     "Suelen ser fechas por defecto (01/01/1900, 2099…)."))

    for item in conv_log:
        if item.get("perdidos"):
            alerts.append(_alert("Media", item["columna"], "Valores no convertibles",
                                 f"{item['perdidos']:,} valores no se pudieron leer como número y quedaron "
                                 "nulos", "Revisa esos valores en el archivo original."))

    if 2 <= len(num_cols) <= 150:
        corr = df[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
        for (a_, b_), r in upper[upper > 0.95].sort_values(ascending=False).head(15).items():
            alerts.append(_alert("Baja", f"{a_} · {b_}", "Variables redundantes", f"|r de Pearson| = {r:.3f}",
                                 "Casi la misma información; valora quedarte con una."))

    out = pd.DataFrame(alerts)
    if out.empty:
        return pd.DataFrame(columns=ALERT_COLS)
    out["_o"] = out["gravedad"].map(SEV_ORDER)
    return out.sort_values(["_o", "problema"]).drop(columns="_o").reset_index(drop=True)


def quality_scores(summary: dict, alerts: pd.DataFrame, columns) -> dict:
    bad_cols = set()
    if not alerts.empty:
        for c in alerts.loc[alerts["gravedad"].isin(["Alta", "Media"]), "columna"]:
            bad_cols.update(str(c).split(" · "))
    ncols = max(len(columns), 1)
    return {
        "Completitud": 1 - summary["null_pct"] / 100,
        "Unicidad de filas": 1 - summary["dup_rows"] / max(summary["rows"], 1),
        "Columnas sin problemas graves": 1 - len(bad_cols & set(columns)) / ncols,
    }
