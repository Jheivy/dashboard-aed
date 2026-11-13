"""Análisis de valores faltantes: patrones, correlación de nulidad, mecanismo e imputación."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from .stats_utils import cramers_v


@st.cache_data(show_spinner=False, max_entries=8)
def missing_summary(_df: pd.DataFrame, key: str) -> pd.DataFrame:
    df = _df
    n = len(df)
    m = df.isna().sum()
    out = pd.DataFrame({"columna": m.index, "nulos": m.values,
                        "% nulos": (m.values / n * 100) if n else 0.0,
                        "dtype": df.dtypes.astype(str).values})
    return out.sort_values("nulos", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=8)
def missing_patterns(_df: pd.DataFrame, key: str, cols: list[str], top: int = 15) -> pd.DataFrame:
    df = _df
    """Most frequent combinations of missing columns (1 = missing)."""
    ind = df[cols].isna().astype(np.int8)
    pat = ind.value_counts().reset_index(name="filas")
    pat["% filas"] = pat["filas"] / len(df) * 100
    pat["nº columnas nulas"] = pat[cols].sum(axis=1)
    return pat.head(top)


@st.cache_data(show_spinner=False, max_entries=8)
def nullity_correlation(_df: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    df = _df
    """Correlation between the missingness indicators (1 = missing) of each column."""
    return df[cols].isna().astype(float).corr()


@st.cache_data(show_spinner="Calculando dependencia de los nulos…", max_entries=16)
def missing_dependence(_df: pd.DataFrame, key: str, target: str, num_cols: list[str],
                       cat_cols: list[str]) -> pd.DataFrame:
    df = _df
    """For column `target`, does its missingness depend on the other variables?
    Compares each other variable between rows where `target` is missing vs present."""
    miss = df[target].isna()
    rows = []
    for c in num_cols:
        if c == target:
            continue
        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        a, b = x[miss].dropna(), x[~miss].dropna()
        if len(a) < 5 or len(b) < 5 or x.nunique() < 2:
            continue
        try:
            _, p = stats.mannwhitneyu(a, b)
        except ValueError:
            continue
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = (a.mean() - b.mean()) / sd if sd > 0 else 0.0
        rows.append(dict(variable=c, tipo="numérica", test="Mann-Whitney U", p=p,
                         efecto=abs(smd), medida="|diferencia de medias estandarizada|",
                         detalle=f"mediana con nulo = {a.median():.3g} · sin nulo = {b.median():.3g}"))
    for c in cat_cols:
        if c == target or df[c].nunique() > 50:
            continue
        v, p, _ = cramers_v(miss, df[c].astype(object).where(df[c].notna(), "(nulo)"))
        if pd.isna(v):
            continue
        rate = miss.groupby(df[c].astype(object).where(df[c].notna(), "(nulo)")).mean() * 100
        worst = rate.idxmax()
        rows.append(dict(variable=c, tipo="categórica", test="Chi-cuadrado", p=p, efecto=v,
                         medida="V de Cramér",
                         detalle=f"más nulos en «{worst}» ({rate.max():.1f}%) vs "
                                 f"{miss.mean() * 100:.1f}% global"))
    for c in df.columns:
        if c == target or df[c].isna().sum() == 0 or df[c].notna().sum() == 0:
            continue
        v, p, _ = cramers_v(miss, df[c].isna())
        if pd.isna(v):
            continue
        rows.append(dict(variable=f"nulos de {c}", tipo="nulidad", test="Chi-cuadrado", p=p,
                         efecto=v, medida="V de Cramér",
                         detalle="¿faltan a la vez?"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    m = len(out)
    out["p ajustado (Holm)"] = _holm(out["p"].values)
    out["relacionada"] = (out["p ajustado (Holm)"] < 0.05) & (
        ((out["medida"].str.contains("estandarizada")) & (out["efecto"] >= 0.2)) |
        ((out["medida"] == "V de Cramér") & (out["efecto"] >= 0.1)))
    out.attrs["n_tests"] = m
    return out.sort_values(["relacionada", "efecto"], ascending=[False, False]).reset_index(drop=True)


def _holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def _em_mvn(X: np.ndarray, max_iter: int = 200, tol: float = 1e-5):
    """EM estimates of mean and covariance of a multivariate normal with missing data."""
    n, p = X.shape
    obs = ~np.isnan(X)
    mu = np.nanmean(X, axis=0)
    S = np.diag(np.nanvar(X, axis=0) + 1e-9)
    pats, inv = np.unique(obs, axis=0, return_inverse=True)
    inv = inv.ravel()
    for _ in range(max_iter):
        Xh = np.where(obs, X, 0.0)
        C = np.zeros((p, p))
        for k, o in enumerate(pats):
            idx = inv == k
            m = ~o
            if not m.any():
                continue
            if not o.any():
                Xh[np.ix_(idx, m)] = mu[m]
                C[np.ix_(m, m)] += idx.sum() * S[np.ix_(m, m)]
                continue
            B = S[np.ix_(m, o)] @ np.linalg.pinv(S[np.ix_(o, o)])
            Xh[np.ix_(idx, m)] = mu[m] + (X[np.ix_(idx, o)] - mu[o]) @ B.T
            C[np.ix_(m, m)] += idx.sum() * (S[np.ix_(m, m)] - B @ S[np.ix_(o, m)])
        mu_new = Xh.mean(axis=0)
        D = Xh - mu_new
        S_new = (D.T @ D + C) / n
        done = np.max(np.abs(mu_new - mu)) < tol and np.max(np.abs(S_new - S)) < tol
        mu, S = mu_new, S_new
        if done:
            break
    return mu, S, pats, inv


@st.cache_data(show_spinner="Ejecutando el test MCAR de Little…", max_entries=8)
def little_mcar_test(_df: pd.DataFrame, key: str, cols: list[str], max_rows: int = 20000) -> dict | None:
    df = _df
    """Little (1988) test. H0: data are Missing Completely At Random.
    Uses EM estimates of mean/covariance (numeric columns only)."""
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.nunique() > 1]
    X = X.dropna(how="all")
    if X.shape[1] < 2 or len(X) < 10 or not X.isna().any().any():
        return None
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=0)
    X = (X - X.mean()) / X.std()  # scale-invariant statistic; improves conditioning
    arr = X.to_numpy(float)
    mu, S, pats, inv = _em_mvn(arr)
    d2, dof = 0.0, 0
    for k, o in enumerate(pats):
        if not o.any():
            continue
        idx = inv == k
        diff = arr[np.ix_(idx, o)].mean(axis=0) - mu[o]
        d2 += idx.sum() * diff @ np.linalg.pinv(S[np.ix_(o, o)]) @ diff
        dof += int(o.sum())
    dof -= arr.shape[1]
    if dof <= 0:
        return None
    return dict(chi2=float(d2), dof=dof, p=float(stats.chi2.sf(d2, dof)),
                n_patterns=int(len(pats)), n=len(arr), cols=list(X.columns))


def impute_preview(df: pd.DataFrame, key: str, col: str, method: str, num_cols: list[str],
                   const_value=None, k: int = 5) -> pd.Series:
    s = df[col]
    if isinstance(s.dtype, pd.CategoricalDtype):  # fillna with a new label would fail on a category
        s = s.astype(object)
    if method == "Media":
        return s.fillna(s.mean())
    if method == "Mediana":
        return s.fillna(s.median())
    if method == "Moda":
        mode = s.mode(dropna=True)
        return s.fillna(mode.iloc[0]) if len(mode) else s
    if method == "Valor constante":
        return s.fillna(const_value)
    if method == "Interpolación lineal":
        return s.interpolate(limit_direction="both")
    if method == "Arrastrar anterior (ffill)":
        return s.ffill().bfill()
    if method.startswith("KNN"):
        return _knn_impute(df, key, col, num_cols, k)
    return s


@st.cache_data(show_spinner="Imputando con KNN…", max_entries=8)
def _knn_impute(_df: pd.DataFrame, key: str, col: str, num_cols: list[str], k: int) -> pd.Series:
    df = _df
    from sklearn.impute import KNNImputer
    feats = [c for c in num_cols if c != col and df[c].notna().mean() > 0.5][:15]
    X = df[[col] + feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0, 1)
    Z = (X - mu) / sd
    filled = KNNImputer(n_neighbors=k).fit_transform(Z)[:, 0]
    return pd.Series(filled * sd[col] + mu[col], index=df.index, name=col)
