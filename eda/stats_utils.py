"""Estadística reutilizable: descriptivos, normalidad, tests entre grupos y asociación."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

ALPHA = 0.05


def apply_scale(s: pd.Series, scale: str) -> pd.Series:
    s = pd.Series(s, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if scale == "log10":
        return np.log10(s[s > 0])
    if scale == "log1p":
        return np.log1p(s[s > -1])
    if scale == "raíz":
        return np.sqrt(s[s >= 0])
    return s


def apply_scales(df: pd.DataFrame, col_scales: dict[str, str]) -> pd.DataFrame:
    """Several columns, each with its own scale, keeping only rows valid for all."""
    cols = list(col_scales)
    d = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    valid = pd.Series(True, index=d.index)
    for c, sc in col_scales.items():
        if sc == "log10":
            valid &= d[c] > 0
        elif sc == "log1p":
            valid &= d[c] > -1
        elif sc == "raíz":
            valid &= d[c] >= 0
    d = d.loc[valid].copy()
    for c, sc in col_scales.items():
        if sc == "log10":
            d[c] = np.log10(d[c])
        elif sc == "log1p":
            d[c] = np.log1p(d[c])
        elif sc == "raíz":
            d[c] = np.sqrt(d[c])
    return d


def describe_numeric(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    n_total = len(x)
    x = x.dropna()
    if x.empty:
        return {}
    q = x.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    mean, sd = x.mean(), x.std()
    return {
        "n": len(x), "nulos": n_total - len(x), "media": mean, "desv. típica": sd,
        "coef. variación": sd / abs(mean) if mean else np.nan,
        "mín": x.min(), "p1": q[0.01], "p5": q[0.05], "Q1": q[0.25], "mediana": q[0.5],
        "Q3": q[0.75], "p95": q[0.95], "p99": q[0.99], "máx": x.max(),
        "rango": x.max() - x.min(), "IQR": q[0.75] - q[0.25],
        "MAD": stats.median_abs_deviation(x, scale="normal") if len(x) > 1 else np.nan,
        "asimetría": x.skew() if len(x) > 2 else np.nan,
        "curtosis (exceso)": x.kurt() if len(x) > 3 else np.nan,
        "% ceros": (x == 0).mean() * 100, "% negativos": (x < 0).mean() * 100,
        "únicos": x.nunique(),
    }


def skew_text(skew: float) -> str:
    if pd.isna(skew):
        return ""
    a = abs(skew)
    side = "derecha (cola de valores altos)" if skew > 0 else "izquierda (cola de valores bajos)"
    if a < 0.5:
        return "Distribución aproximadamente simétrica."
    if a < 1:
        return f"Asimetría moderada hacia la {side}."
    return (f"Asimetría fuerte hacia la {side}. Considera la escala logarítmica, "
            "estadísticos robustos (mediana, IQR) y tests no paramétricos.")


def normality_tests(s: pd.Series) -> list[dict]:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out = []
    if x.nunique() < 3:
        return out
    if len(x) >= 3:
        smp = x.sample(min(len(x), 5000), random_state=0)
        st_, p = stats.shapiro(smp)
        out.append(dict(test="Shapiro-Wilk" + (" (muestra 5.000)" if len(x) > 5000 else ""),
                        estadístico=st_, p=p))
    if len(x) >= 20:
        st_, p = stats.normaltest(x)
        out.append(dict(test="D'Agostino-Pearson K²", estadístico=st_, p=p))
    if len(x) >= 2:
        st_, p = stats.jarque_bera(x)
        out.append(dict(test="Jarque-Bera", estadístico=st_, p=p))
    return out


def compare_groups(groups: list[np.ndarray], names: list[str]) -> dict | None:
    """Numeric variable across 2+ groups: non-parametric + parametric test and effect size."""
    valid = [(n, np.asarray(g, dtype=float)) for n, g in zip(names, groups) if len(g) >= 3]
    if len(valid) < 2:
        return None
    gs = [g for _, g in valid]
    try:
        if len(gs) == 2:
            u, p = stats.mannwhitneyu(gs[0], gs[1])
            t, p_t = stats.ttest_ind(gs[0], gs[1], equal_var=False)
            n1, n2 = len(gs[0]), len(gs[1])
            pooled = np.sqrt(((n1 - 1) * gs[0].var(ddof=1) + (n2 - 1) * gs[1].var(ddof=1)) / (n1 + n2 - 2))
            d = (gs[0].mean() - gs[1].mean()) / pooled if pooled > 0 else np.nan
            rbc = 1 - 2 * u / (n1 * n2)
            return dict(test="Mann-Whitney U", stat=u, p=p, test_param="t de Welch", p_param=p_t,
                        effect_name="d de Cohen", effect=d, effect2_name="correlación biserial de rangos",
                        effect2=rbc, n_groups=2)
        h, p = stats.kruskal(*gs)
        f, p_f = stats.f_oneway(*gs)
        n = sum(len(g) for g in gs)
        eps2 = (h - len(gs) + 1) / (n - len(gs)) if n > len(gs) else np.nan
        allv = np.concatenate(gs)
        ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in gs)
        ss_t = ((allv - allv.mean()) ** 2).sum()
        return dict(test="Kruskal-Wallis", stat=h, p=p, test_param="ANOVA", p_param=p_f,
                    effect_name="η² (eta cuadrado)", effect=ss_b / ss_t if ss_t else np.nan,
                    effect2_name="ε² (Kruskal)", effect2=eps2, n_groups=len(gs))
    except ValueError:
        return None


def effect_label(name: str, value: float) -> str:
    if pd.isna(value):
        return ""
    v = abs(value)
    if "Cohen" in name:
        cuts = (0.2, 0.5, 0.8)
    elif "η²" in name or "ε²" in name:
        cuts = (0.01, 0.06, 0.14)
    else:  # correlations, Cramér's V, rank-biserial
        cuts = (0.1, 0.3, 0.5)
    if v < cuts[0]:
        return "despreciable"
    if v < cuts[1]:
        return "pequeño"
    if v < cuts[2]:
        return "mediano"
    return "grande"


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    """Bias-corrected Cramér's V. Returns (V, p-value of chi², dof)."""
    ct = pd.crosstab(x, y)
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]  # unused categorical levels
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan, np.nan, 0
    chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
    n = ct.values.sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return (np.sqrt(phi2c / denom) if denom > 0 else np.nan), p, dof


def correlation_ratio(cat: pd.Series, num: pd.Series) -> float:
    """η: how much of a numeric variable's variance is explained by a categorical one (0–1)."""
    d = pd.DataFrame({"c": cat, "x": pd.to_numeric(num, errors="coerce")}).dropna()
    if d.empty or d["c"].nunique() < 2:
        return np.nan
    grand = d["x"].mean()
    g = d.groupby("c", observed=True)["x"].agg(["mean", "count"])
    ss_b = (g["count"] * (g["mean"] - grand) ** 2).sum()
    ss_t = ((d["x"] - grand) ** 2).sum()
    return float(np.sqrt(ss_b / ss_t)) if ss_t > 0 else np.nan


@st.cache_data(show_spinner="Calculando asociaciones…", max_entries=16)
def association_matrix(_df: pd.DataFrame, key: str, num_cols: list[str], cat_cols: list[str],
                       max_levels: int = 50) -> pd.DataFrame:
    """Mixed association matrix in [0, 1]: |Spearman| num–num, η num–cat, Cramér's V cat–cat."""
    df = _df
    cat_cols = [c for c in cat_cols if df[c].nunique() <= max_levels and c not in num_cols]
    cols = list(num_cols) + list(cat_cols)
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    if num_cols:
        sp = df[num_cols].corr(method="spearman").abs()
        m.loc[num_cols, num_cols] = sp.values
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a in num_cols and b in num_cols:
                continue
            if a in cat_cols and b in cat_cols:
                v = cramers_v(df[a], df[b])[0]
            elif a in cat_cols:
                v = correlation_ratio(df[a], df[b])
            else:
                v = correlation_ratio(df[b], df[a])
            m.loc[a, b] = m.loc[b, a] = v
    return m


def cluster_order(mat: pd.DataFrame, is_similarity: bool = True) -> list[int]:
    """Leaf order of a hierarchical clustering so similar variables sit together."""
    if len(mat) < 3:
        return list(range(len(mat)))
    sim = mat.abs().fillna(0).values
    dist = np.clip(1 - sim if is_similarity else sim, 0, None)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    link = hierarchy.linkage(squareform(dist, checks=False), method="average")
    return hierarchy.leaves_list(link).tolist()


def pearson_ci(x, y, conf: float = 0.95):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r, p = stats.pearsonr(x, y)
    n = len(x)
    if n <= 3:
        return r, np.nan, np.nan, p, n
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1 / np.sqrt(n - 3)
    zc = stats.norm.ppf((1 + conf) / 2)
    return r, np.tanh(z - zc * se), np.tanh(z + zc * se), p, n


@st.cache_data(show_spinner=False, max_entries=32)
def corr_matrix(_df: pd.DataFrame, key: str, cols: tuple, method: str) -> pd.DataFrame:
    """Correlation matrix, cached per data state + columns + method (Spearman ranks are costly)."""
    return _df[list(cols)].corr(method=method)


@st.cache_data(show_spinner=False, max_entries=16)
def corr_pvalues(_df: pd.DataFrame, key: str, method: str) -> pd.DataFrame:
    df = _df
    cols = df.columns
    p = pd.DataFrame(np.nan, index=cols, columns=cols)
    fn = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            d = df[[a, b]].dropna()
            if len(d) > 3 and d[a].nunique() > 1 and d[b].nunique() > 1:
                p.loc[a, b] = p.loc[b, a] = fn(d[a], d[b])[1]
    return p


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "—"
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def fmt_num(v, dec: int = 2) -> str:
    """Spanish number format: 1.234,56"""
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}".replace(",", ".")
    s = f"{v:,.{dec}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")
