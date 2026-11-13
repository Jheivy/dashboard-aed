"""Detección de outliers univariante y multivariante."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.preprocessing import StandardScaler

UNI_METHODS = {
    "IQR (Tukey)": ("Multiplicador k del IQR", 1.0, 4.0, 1.5, 0.1),
    "Z-score": ("Umbral |z|", 2.0, 6.0, 3.0, 0.1),
    "Z-score robusto (MAD)": ("Umbral |z robusto|", 2.0, 8.0, 3.5, 0.1),
    "Percentiles": ("Percentil inferior / superior (%)", 0.1, 10.0, 1.0, 0.1),
}
UNI_HELP = {
    "IQR (Tukey)": "Fuera de [Q1 − k·IQR, Q3 + k·IQR]. Robusto, no asume normalidad. "
                   "k = 1,5 es el bigote del boxplot; k = 3 marca solo los extremos.",
    "Z-score": "Distancia a la media en desviaciones típicas. Asume normalidad y la propia "
               "media/desviación se ven arrastradas por los outliers.",
    "Z-score robusto (MAD)": "Como el z-score pero con mediana y MAD (Iglewicz y Hoaglin): no se "
                             "deja arrastrar por los propios outliers. Umbral habitual: 3,5.",
    "Percentiles": "Marca el p% más bajo y más alto. Siempre marca algo: útil para recortar "
                   "(winsorizar), no para decidir si hay anomalías.",
}


def uni_bounds(x: pd.Series, method: str, param: float) -> tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return np.nan, np.nan
    if method == "IQR (Tukey)":
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        return q1 - param * iqr, q3 + param * iqr
    if method == "Z-score":
        return x.mean() - param * x.std(), x.mean() + param * x.std()
    if method == "Z-score robusto (MAD)":
        med = x.median()
        mad = stats.median_abs_deviation(x, scale="normal")
        if mad == 0:  # >50% identical values: fall back to the mean absolute deviation
            mad = (x - med).abs().mean() * 1.2533
        return med - param * mad, med + param * mad
    if method == "Percentiles":
        return x.quantile(param / 100), x.quantile(1 - param / 100)
    raise ValueError(method)


@st.cache_data(show_spinner="Calculando outliers de todas las variables…", max_entries=8)
def outlier_overview(_df: pd.DataFrame, key: str, cols: list[str], k_iqr: float = 1.5,
                     z: float = 3.0, z_mad: float = 3.5) -> pd.DataFrame:
    df = _df
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < 5 or x.nunique() < 3:
            continue
        r = {"variable": c, "n": len(x)}
        for name, p in [("IQR (Tukey)", k_iqr), ("Z-score", z), ("Z-score robusto (MAD)", z_mad)]:
            lo, hi = uni_bounds(x, name, p)
            m = (x < lo) | (x > hi)
            r[f"% {name}"] = m.mean() * 100
            if name == "IQR (Tukey)":
                r["nº IQR"] = int(m.sum())
                r["bajos IQR"] = int((x < lo).sum())
                r["altos IQR"] = int((x > hi).sum())
                r["límite inf. IQR"], r["límite sup. IQR"] = lo, hi
        r["mín"], r["máx"] = x.min(), x.max()
        r["asimetría"] = x.skew()
        rows.append(r)
    out = pd.DataFrame(rows)
    return out.sort_values("% IQR (Tukey)", ascending=False).reset_index(drop=True) if len(out) else out


def _prep_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return X.loc[:, X.nunique() > 1]


@st.cache_data(show_spinner="Ajustando detectores multivariantes…", max_entries=8)
def multivariate_scores(_df: pd.DataFrame, key: str, cols: list[str], contamination: float,
                        chi_q: float, methods: tuple[str, ...], max_fit: int = 20000):
    """Returns a DataFrame indexed like the complete rows of `df[cols]` with one
    score + flag column per method, and a dict of method notes."""
    X = _prep_matrix(_df, cols)
    notes = {}
    out = pd.DataFrame(index=X.index)
    if X.shape[1] < 1 or len(X) < 10:
        return out, {"error": "Hacen falta al menos 10 filas completas y 1 variable con variación."}
    Z = StandardScaler().fit_transform(X)
    fit_idx = np.arange(len(Z))
    if len(Z) > max_fit:
        fit_idx = np.random.default_rng(0).choice(len(Z), max_fit, replace=False)
        notes["muestra"] = f"Modelos ajustados sobre una muestra de {max_fit:,} filas y aplicados a todas."
    p = Z.shape[1]
    thr = np.sqrt(stats.chi2.ppf(chi_q, df=p))

    if "Mahalanobis robusta (MCD)" in methods and p >= 2:
        from sklearn.covariance import MinCovDet
        try:
            mcd = MinCovDet(random_state=0).fit(Z[fit_idx])
            d = np.sqrt(mcd.mahalanobis(Z))
            out["Mahalanobis robusta (MCD)"] = d
            out["flag_Mahalanobis robusta (MCD)"] = d > thr
        except (ValueError, np.linalg.LinAlgError) as e:
            notes["Mahalanobis robusta (MCD)"] = f"No se pudo ajustar: {e}"
    if "Mahalanobis clásica" in methods and p >= 2:
        inv = np.linalg.pinv(np.cov(Z, rowvar=False))
        diff = Z - Z.mean(axis=0)
        d = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))
        out["Mahalanobis clásica"] = d
        out["flag_Mahalanobis clásica"] = d > thr
    if "Isolation Forest" in methods:
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=contamination, random_state=0, n_estimators=200)
        iso.fit(Z[fit_idx])
        out["Isolation Forest"] = -iso.score_samples(Z)
        out["flag_Isolation Forest"] = iso.predict(Z) == -1
    if "Local Outlier Factor" in methods:
        from sklearn.neighbors import LocalOutlierFactor
        if len(Z) > max_fit:
            lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
            lof.fit(Z[fit_idx])
            out["Local Outlier Factor"] = -lof.score_samples(Z)
            out["flag_Local Outlier Factor"] = lof.predict(Z) == -1
        else:
            lof = LocalOutlierFactor(n_neighbors=min(20, len(Z) - 1), contamination=contamination)
            flags = lof.fit_predict(Z) == -1
            out["Local Outlier Factor"] = -lof.negative_outlier_factor_
            out["flag_Local Outlier Factor"] = flags
    flag_cols = [c for c in out.columns if c.startswith("flag_")]
    out["nº métodos"] = out[flag_cols].sum(axis=1) if flag_cols else 0
    notes["umbral_chi2"] = thr
    notes["variables"] = list(X.columns)
    return out, notes


def winsorize(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return x.clip(lower=lo, upper=hi)
