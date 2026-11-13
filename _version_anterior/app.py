"""
Generic, reusable EDA dashboard.
Run:  streamlit run app.py
(or just ./ejecutar.sh / ejecutar.bat)

Detects types at runtime (including Decimal and numeric strings) and lets
you reclassify columns manually from the sidebar.
"""

import re
import warnings
from collections import Counter

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import median_survival_times, restricted_mean_survival_time
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Page config + compact CSS (avoids scrolling)
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="EDA Dashboard", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    html, body {
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    .stApp {background-color: #f9f9f7;}
    .block-container {padding-top: 1.0rem; padding-bottom: 2.5rem;}
    [data-testid="stMetricValue"] {font-size: 1.05rem; color: #0b0b0b;}
    [data-testid="stMetricLabel"] {font-size: 0.72rem; color: #52514e;}
    h1 {font-size: 1.3rem; margin-bottom: 0.1rem; color: #0b0b0b;}
    h3 {font-size: 0.95rem; margin-bottom: 0.2rem; color: #0b0b0b;}
    .stTabs [data-baseweb="tab"] {padding: 3px 10px; font-size: 0.85rem;}
    [data-testid="stCaptionContainer"] {color: #898781;}
</style>
""", unsafe_allow_html=True)

# Shared Plotly template so every chart in the app (px.* and go.Figure alike)
# picks up the same font, categorical colors, and recessive grid/axis styling
# without editing each chart individually.
pio.templates["professional"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  color="#0b0b0b", size=12),
        title=dict(font=dict(size=14, color="#0b0b0b")),
        colorway=["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#e1e0d9", linecolor="#c3c2b7",
                  tickfont=dict(color="#898781"), zerolinecolor="#e1e0d9"),
        yaxis=dict(gridcolor="#e1e0d9", linecolor="#c3c2b7",
                  tickfont=dict(color="#898781"), zerolinecolor="#e1e0d9"),
    )
)
pio.templates.default = "professional"

PALETTE = ["rgb(42,120,214)", "rgb(235,104,52)", "rgb(27,175,122)", "rgb(237,161,0)",
          "rgb(232,123,164)", "rgb(0,131,0)", "rgb(74,58,167)", "rgb(227,73,72)"]
LAYOUT = dict(margin=dict(l=0, r=0, t=25, b=0),
              legend=dict(orientation="h", y=-0.18, font=dict(size=10)))


# --------------------------------------------------------------------------- #
# Data loading (one or several files)
# --------------------------------------------------------------------------- #
@st.cache_data
def load_data(file) -> pd.DataFrame:
    name = file.name.lower()
    return pd.read_parquet(file) if name.endswith(".parquet") else pd.read_csv(file)


def dedupe_names(names: list[str]) -> list[str]:
    """Two uploads can share a base filename (e.g. both named
    churn_labels.parquet from different folders) -- Streamlit only exposes
    the base name, not the original path, so disambiguate by upload order
    instead of colliding on widget keys / __source__ values."""
    counts = Counter(names)
    seen = Counter()
    out = []
    for name in names:
        if counts[name] == 1:
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name} ({seen[name]})")
    return out


def combine_files(files, labels: list[str]) -> pd.DataFrame:
    """Stacks several files into a single DataFrame and tags each row's origin.

    The __source__ column lets you later distinguish/overlay the files in
    any chart (e.g. time series grouped by origin). `labels` (from
    dedupe_names) are used instead of raw filenames so two same-named
    uploads still end up distinguishable.
    """
    frames = []
    for f, label in zip(files, labels):
        d = load_data(f).copy()
        d["__source__"] = label
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _has_unhashable(s: pd.Series) -> bool:
    """True if the column contains lists/arrays (unhashable cells)."""
    val = s.dropna()
    if val.empty:
        return False
    return isinstance(val.iloc[0], (list, np.ndarray, dict, set))


def safe_nunique(s: pd.Series) -> int:
    """nunique tolerant to unhashable cells (nested arrays/lists)."""
    try:
        return s.nunique(dropna=True)
    except TypeError:
        return s.astype(str).nunique()


def auto_split(df: pd.DataFrame):
    """Classifies columns by dtype (already set during Kedro cleaning)."""
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical = [c for c in df.select_dtypes(
        include=["category", "object", "string", "bool"]).columns
        if not _has_unhashable(df[c])]  # nested arrays excluded: not groupable
    return numeric, categorical, datetime_cols


def _default_index(options: list[str], keyword: str) -> int:
    """Index of the first option containing `keyword` (0 if none)."""
    for i, c in enumerate(options):
        if keyword in c.lower():
            return i
    return 0


def _rgba(rgb_str: str, alpha: float) -> str:
    """'rgb(r,g,b)' (PALETTE format) -> 'rgba(r,g,b,alpha)' for CI shading."""
    return rgb_str.replace("rgb(", "rgba(").replace(")", f",{alpha})")


def _pearson_ci(x, y, conf: float = 0.95):
    """Pearson r with a Fisher z-transform confidence interval.
    Returns (r, lo, hi, p, n)."""
    r, p = stats.pearsonr(x, y)
    n = len(x)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf((1 + conf) / 2)
    return r, np.tanh(z - zcrit * se), np.tanh(z + zcrit * se), p, n


def transform_scale(s, scale: str) -> pd.Series:
    s = pd.Series(s).dropna()
    if scale == "log":
        s = s[s > 0]
        return np.log10(s)
    if scale == "log1p":
        s = s[s >= 0]
        return np.log1p(s)
    if scale == "sqrt":
        s = s[s >= 0]
        return np.sqrt(s)
    return s


def transform_columns(df: pd.DataFrame, cols: list[str], scale: str) -> pd.DataFrame:
    """Like transform_scale but for several columns at once, keeping only
    the rows valid for all of them (needed so the pairplot doesn't misalign
    columns transformed with a different cutoff each)."""
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if scale == "log":
        d = d[(d > 0).all(axis=1)]
        return np.log10(d)
    if scale == "log1p":
        d = d[(d >= 0).all(axis=1)]
        return np.log1p(d)
    if scale == "sqrt":
        d = d[(d >= 0).all(axis=1)]
        return np.sqrt(d)
    return d


def transform_columns_mixed(df: pd.DataFrame, col_scales: dict[str, str]) -> pd.DataFrame:
    """Like transform_columns but each column can have its own scale."""
    cols = list(col_scales)
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    valid = pd.Series(True, index=d.index)
    for col, scale in col_scales.items():
        if scale == "log":
            valid &= d[col] > 0
        elif scale in ("log1p", "sqrt"):
            valid &= d[col] >= 0
    d = d.loc[valid].copy()
    for col, scale in col_scales.items():
        if scale == "log":
            d[col] = np.log10(d[col])
        elif scale == "log1p":
            d[col] = np.log1p(d[col])
        elif scale == "sqrt":
            d[col] = np.sqrt(d[col])
    return d


# --------------------------------------------------------------------------- #
# Cached heavy computations -- Streamlit reruns the WHOLE script on every
# widget interaction, including every OTHER tab's body, not just the one
# being looked at. These are the computations expensive enough (scale with
# rows x columns, or fit a real model) that redoing them on every unrelated
# rerun was the main source of lag; caching keys on the actual inputs, so a
# widget change elsewhere in the app is a cache hit here, not a recompute.
# --------------------------------------------------------------------------- #
@st.cache_data
def _fit_pca(data: pd.DataFrame, n_comp: int):
    Z = StandardScaler().fit_transform(data)
    pca = PCA(n_components=n_comp).fit(Z)
    return pca.transform(Z), pca.explained_variance_ratio_, pca.components_


@st.cache_data
def _mahalanobis_distances(data: pd.DataFrame) -> np.ndarray:
    Xv = data.values
    inv = np.linalg.pinv(np.cov(Xv, rowvar=False))  # pinv: robust if singular
    diff = Xv - Xv.mean(axis=0)
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))


@st.cache_data
def _fit_isolation_forest(data: pd.DataFrame, contamination: float):
    Z = StandardScaler().fit_transform(data)
    iso = IsolationForest(contamination=contamination, random_state=0)
    mask = iso.fit_predict(Z) == -1
    return mask, -iso.score_samples(Z)


@st.cache_data
def _clustered_corr(data: pd.DataFrame, method: str) -> pd.DataFrame:
    """Correlation matrix reordered by hierarchical clustering (correlated
    blocks sit next to each other instead of raw column order)."""
    corr = data.corr(method=method)
    # clip(lower=0): near-perfectly correlated columns can compute to corr
    # slightly above 1 (float rounding), making 1-|corr| a tiny negative
    # number that scipy's linkage rejects outright.
    dist = (1 - corr.abs()).clip(lower=0)
    condensed = squareform(dist.values, checks=False)
    linkage = hierarchy.linkage(condensed, method="average")
    order = hierarchy.dendrogram(linkage, no_plot=True)["leaves"]
    return corr.iloc[order, order]


@st.cache_data
def _auc_table(df: pd.DataFrame, target_col: str, numeric_cols: list[str],
              categorical_cols: list[str], max_levels: int) -> pd.DataFrame:
    """AUC of every numeric column (as-is) and every categorical level (as a
    0/1 indicator) against a binary target -- the expensive "score every
    column" part of ROC-style tabs, split out so it's not redone on every
    unrelated rerun. `level` is NaN for numeric rows -- reconstruct the
    actual score Series/mask from (col, level) in the caller, since a
    pd.Series isn't a clean thing to round-trip through the cache."""
    values = sorted(df[target_col].dropna().unique(), key=str)
    if len(values) != 2:
        return pd.DataFrame(columns=["label", "col", "level", "auc", "inverted"])
    positive = values[-1]
    y = (df[target_col] == positive).astype(int)

    rows = []
    for col in numeric_cols:
        if col == target_col:
            continue
        s = df[col]
        mask = s.notna() & y.notna()
        if mask.sum() < 2 or s[mask].nunique() < 2:
            continue
        auc = roc_auc_score(y[mask], s[mask])
        inverted = auc < 0.5
        rows.append(dict(label=col, col=col, level=None,
                         auc=(1 - auc if inverted else auc), inverted=inverted))
    for col in categorical_cols:
        if col == target_col or df[col].nunique(dropna=True) > max_levels:
            continue
        for level in sorted(df[col].dropna().unique(), key=str):
            s = (df[col] == level).astype(int)
            mask = s.notna() & y.notna()
            if mask.sum() < 2 or s[mask].nunique() < 2:
                continue
            auc = roc_auc_score(y[mask], s[mask])
            inverted = auc < 0.5
            rows.append(dict(label=f"{col} = {level}", col=col, level=level,
                             auc=(1 - auc if inverted else auc), inverted=inverted))
    return pd.DataFrame(rows)


@st.cache_data
def _fit_km(duration: pd.Series, event: pd.Series):
    """Kaplan-Meier fit -> plain data (not the fitted object, which isn't a
    clean cache_data return type): survival function, its CI, median survival
    time + CI, and RMST bounded to this group's own last observed time
    (unbounded defaults to t=inf, which returns literally "inf" once the
    curve's last segment is censored -- see the RMST caption below)."""
    kmf = KaplanMeierFitter().fit(duration, event)
    sf = kmf.survival_function_.iloc[:, 0]
    ci = kmf.confidence_interval_
    median = kmf.median_survival_time_
    if np.isfinite(median):
        m_ci = median_survival_times(kmf.confidence_interval_).iloc[0]
        median_lo, median_hi = m_ci.iloc[0], m_ci.iloc[1]
    else:
        median_lo, median_hi = np.nan, np.nan
    rmst = restricted_mean_survival_time(kmf, t=duration.max())
    return sf, ci, median, median_lo, median_hi, rmst


@st.cache_data
def _bucket_correlations(sub_df: pd.DataFrame, x_col: str, y_col: str, z_col: str,
                         scales: dict[str, str], n_buckets: int):
    """abs/scale-adjusted (x, y, z) rows -> (pooled r/lo/hi/p/n, per-bucket
    DataFrame), or None if there isn't enough clean data to say anything."""
    cleaned = transform_columns_mixed(sub_df, {x_col: scales[x_col], y_col: scales[y_col],
                                               z_col: scales[z_col]})
    if len(cleaned) < 10:
        return None
    r_pooled, lo_p, hi_p, p_pooled, n_pooled = _pearson_ci(cleaned[x_col], cleaned[y_col])
    try:
        cleaned = cleaned.assign(_bucket=pd.qcut(cleaned[z_col], n_buckets, duplicates="drop"))
    except ValueError:
        return None
    rows = []
    for pos, (interval, g) in enumerate(cleaned.groupby("_bucket", observed=True), start=1):
        if len(g) < 5:
            continue
        r, lo, hi, p, n = _pearson_ci(g[x_col], g[y_col])
        rows.append(dict(position=pos, bucket=str(interval), r=r, lo=lo, hi=hi, p=p, n=n))
    if not rows:
        return None
    return (r_pooled, lo_p, hi_p, p_pooled, n_pooled), pd.DataFrame(rows)


@st.cache_data
def _churn_definition(active_sub: pd.DataFrame, pctl: float):
    """(subject, date) rows -> (per_subject, gaps, threshold_days), or None if too little data.

    The gap-per-subject diff used to be a groupby().apply(lambda ...) --
    correct, but it invokes the lambda once per subject in pure Python.
    With one row per subject (e.g. churn_labels.parquet, not the
    daily_player_metrics-like table this tab expects) that's ~30k Python
    calls to do work numpy could vectorize in one pass: 15-20s vs <20ms,
    verified identical output on both a 1-row/subject and 6-rows/subject
    shape. groupby(...).diff() computes the same "gap since previous row
    in this subject's own sorted history" without any Python-level loop.
    """
    if active_sub.empty or active_sub["subject"].nunique() < 2:
        return None
    window_end = active_sub["date"].max()
    g = active_sub.sort_values("date").groupby("subject")["date"]
    per_subject = pd.DataFrame({
        "first_active": g.min(), "last_active": g.max(), "n_active_periods": g.nunique(),
    })
    sorted_sub = active_sub.sort_values(["subject", "date"])
    gaps = sorted_sub.groupby("subject")["date"].diff().dropna().dt.days
    if len(gaps) < 5:
        return None
    threshold_days = gaps.quantile(pctl)
    per_subject["trailing_inactive"] = (window_end - per_subject["last_active"]).dt.days
    per_subject["churn"] = (per_subject["trailing_inactive"] >= threshold_days).astype(int)
    return per_subject, gaps, threshold_days


# --------------------------------------------------------------------------- #
# Sidebar: data + type controls
# --------------------------------------------------------------------------- #
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("CSV or Parquet", type=["csv", "parquet"],
                                    accept_multiple_files=True)

if not uploaded:
    st.title("EDA Dashboard")
    st.info("Upload one or more CSV/Parquet files in the sidebar to get started.")
    st.stop()

if len(uploaded) == 1:
    df = load_data(uploaded[0])
    source_col = None
else:
    upload_labels = dedupe_names([f.name for f in uploaded])
    st.sidebar.caption("Include in analysis:")
    active = [(f, label) for f, label in zip(uploaded, upload_labels)
             if st.sidebar.checkbox(label, value=True, key=f"include_{label}")]

    if not active:
        st.title("EDA Dashboard")
        st.info("Check at least one dataset in the sidebar to continue.")
        st.stop()
    elif len(active) == 1:
        df = load_data(active[0][0])
        source_col = None
    else:
        active_files, active_labels = zip(*active)
        df = combine_files(active_files, active_labels)
        source_col = "__source__"


st.sidebar.header("SQL (optional)")
sql_query = st.sidebar.text_area(
    "Query the loaded table (table name: df)", value="", height=100,
    placeholder="SELECT * FROM df WHERE churn = 1", key="custom_sql")
if sql_query.strip():
    try:
        df = duckdb.sql(sql_query).df()
    except Exception as e:
        st.sidebar.error(f"SQL error: {e}")

st.sidebar.header("Column types")
auto_num, auto_cat, auto_dt = auto_split(df)

with st.sidebar.expander("Adjust manually"):
    numeric_cols = st.multiselect("Numeric", df.columns.tolist(), default=auto_num)
    categorical_cols = st.multiselect("Categorical", df.columns.tolist(), default=auto_cat)
    datetime_cols = st.multiselect("Dates", df.columns.tolist(), default=auto_dt)

# --------------------------------------------------------------------------- #
# Sidebar: filters (applied before every tab — including the row-count sampler
# below, so filtering down to a small subgroup isn't first crushed by it)
# --------------------------------------------------------------------------- #
st.sidebar.header("Filters")
total_rows_unfiltered = len(df)
filter_mask = pd.Series(True, index=df.index)

churn_like_cols = [c for c in df.columns
                   if "churn" in c.lower() and safe_nunique(df[c]) == 2]
if churn_like_cols:
    churn_shortcut_col = churn_like_cols[0]
    shortcut = st.sidebar.radio(f"Quick filter — {churn_shortcut_col}",
                                ["All", "Active only", "Churned only"],
                                key="filter_churn_shortcut")
    if shortcut != "All":
        positive = sorted(df[churn_shortcut_col].dropna().unique())[-1]
        is_positive = df[churn_shortcut_col] == positive
        filter_mask &= is_positive if shortcut == "Churned only" else ~is_positive

filter_cols = st.sidebar.multiselect("Filter by other columns", df.columns.tolist(),
                                     key="filter_cols")
for col in filter_cols:
    if col in numeric_cols:
        lo, hi = float(df[col].min()), float(df[col].max())
        if lo < hi:
            sel = st.sidebar.slider(col, lo, hi, (lo, hi), key=f"filter_num_{col}")
            filter_mask &= df[col].between(sel[0], sel[1])
        else:
            st.sidebar.caption(f"{col}: constant value, nothing to filter.")
    elif col in datetime_cols:
        lo, hi = df[col].min(), df[col].max()
        if pd.notna(lo) and pd.notna(hi) and lo < hi:
            sel = st.sidebar.date_input(col, (lo.date(), hi.date()), key=f"filter_dt_{col}")
            if isinstance(sel, tuple) and len(sel) == 2:
                filter_mask &= (df[col] >= pd.Timestamp(sel[0])) & \
                               (df[col] <= pd.Timestamp(sel[1]) + pd.Timedelta(days=1))
        else:
            st.sidebar.caption(f"{col}: not enough date range to filter.")
    else:
        vals = df[col].dropna().unique().tolist()
        if 1 < len(vals) <= 200:
            vals_sorted = sorted(vals, key=str)
            sel = st.sidebar.multiselect(col, vals_sorted, default=vals_sorted,
                                         key=f"filter_cat_{col}")
            filter_mask &= df[col].isin(sel)
        else:
            st.sidebar.caption(f"{col}: {len(vals)} distinct values, too many/few to filter.")

df = df[filter_mask].reset_index(drop=True)
if len(df) < total_rows_unfiltered:
    st.sidebar.caption(f"**{len(df):,} of {total_rows_unfiltered:,} rows** after filters.")

MAX_ROWS_DEFAULT = 200_000
total_rows = len(df)
if total_rows > MAX_ROWS_DEFAULT:
    st.sidebar.header("Performance")
    max_rows = st.sidebar.number_input(
        "Max rows to work with", min_value=10_000, max_value=total_rows,
        value=MAX_ROWS_DEFAULT, step=10_000,
        help="Every tab in this dashboard runs on this sample, not the full "
             "file — keeps things fast and light on memory for large files.")
    df = df.sample(int(max_rows), random_state=0).reset_index(drop=True)
    st.sidebar.caption(f"Working with a random sample of {len(df):,} of "
                       f"{total_rows:,} rows.")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("EDA Dashboard")
st.caption("Churn study on a short observation window — flow: Overview -> Explore -> "
          "Relationships -> Churn (define & describe it) -> Predict (what signals it early).")

churn_col_for_kpi = next((c for c in df.columns
                          if "churn" in c.lower() and safe_nunique(df[c]) == 2), None)
hdr = st.columns(6 if churn_col_for_kpi else 5)
hdr[0].metric("Rows", f"{len(df):,}")
hdr[1].metric("Columns", df.shape[1])
hdr[2].metric("Numeric", len(numeric_cols))
hdr[3].metric("Categorical", len(categorical_cols))
last_idx = 4
if churn_col_for_kpi:
    positive = sorted(df[churn_col_for_kpi].dropna().unique())[-1]
    rate = (df[churn_col_for_kpi] == positive).mean() * 100
    hdr[4].metric(f"{churn_col_for_kpi} rate", f"{rate:.1f}%")
    last_idx = 5
if source_col:
    hdr[last_idx].metric("Files", df[source_col].nunique())
else:
    hdr[last_idx].metric("% null", f"{df.isna().mean().mean() * 100:.1f}%")

top = st.tabs(['Overview', 'Explore', 'Relationships', 'Churn', 'Predict'])

# =========================================================================== #
# GROUP 0: OVERVIEW
# =========================================================================== #
with top[0]:
    subtabs = st.tabs(['Data', 'Summary'])

    with subtabs[0]:
        # --------------------------------------------------------------------------- #
        # 0. DATA (table + head)
        # --------------------------------------------------------------------------- #
        ctrl, view = st.columns([1, 5])
        with ctrl:
            n_rows = st.slider("Rows to show", 5, 200, 20, 5, key="tbl_rows")
            cols_show = st.multiselect("Columns", df.columns.tolist(),
                                       default=df.columns.tolist(), key="tbl_cols")
            if source_col:
                srcs = st.multiselect("Filter by file",
                                      df[source_col].unique().tolist(),
                                      default=df[source_col].unique().tolist(),
                                      key="tbl_src")
            else:
                srcs = None

        with view:
            shown = df if srcs is None else df[df[source_col].isin(srcs)]
            if cols_show:
                shown = shown[cols_show]
            st.dataframe(shown.head(n_rows), use_container_width=True, height=360)
            st.caption(f"Showing {min(n_rows, len(shown))} of {len(shown):,} rows "
                       f"· {shown.shape[1]} columns")

            with st.expander(f"Column names ({df.shape[1]})", expanded=True):
                list_tab, detail_tab = st.tabs(["Copyable list", "Detail"])
                with list_tab:
                    st.code(repr(df.columns.tolist()), language="python")
                with detail_tab:
                    info = pd.DataFrame({
                        "column": df.columns,
                        "dtype": df.dtypes.astype(str).values,
                        "non_null": df.notna().sum().values,
                        "unique": [safe_nunique(df[c]) for c in df.columns],
                        "example": [str(df[c].dropna().iloc[0])[:60] if df[c].notna().any()
                                    else None for c in df.columns],
                    })
                    st.dataframe(info, use_container_width=True, height=260,
                                 hide_index=True)

    with subtabs[1]:
        # --------------------------------------------------------------------------- #
        # 1. SUMMARY
        # --------------------------------------------------------------------------- #
        left, right = st.columns([3, 2])

        with left:
            st.markdown("### Statistics")
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe().T.round(3),
                             use_container_width=True, height=240)
            else:
                st.caption("No numeric columns.")

            st.markdown("### Detected types")
            types_df = pd.DataFrame({
                "column": df.columns,
                "dtype": df.dtypes.astype(str).values,
                "unique": [safe_nunique(df[c]) for c in df.columns],
                "% null": (df.isna().mean() * 100).round(1).values,
            })
            st.dataframe(types_df, use_container_width=True, height=180, hide_index=True)

        with right:
            st.markdown("### Missing values")
            miss = df.isna().sum()
            miss = miss[miss > 0].sort_values(ascending=False)
            if len(miss):
                miss_df = pd.DataFrame({
                    "column": miss.index,
                    "% null": (miss.values / len(df) * 100).round(1),
                })
                fig = px.bar(miss_df, x="% null", y="column", orientation="h",
                             text="% null", color_discrete_sequence=PALETTE)
                fig.update_layout(height=230, yaxis=dict(autorange="reversed", title=""),
                                  **LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Null pattern")
                sample = df.sample(min(len(df), 500), random_state=0)
                fig = px.imshow(sample[miss.index].isna().T.astype(int),
                                aspect="auto", color_continuous_scale="Blues")
                fig.update_layout(height=190, coloraxis_showscale=False,
                                  xaxis_title="rows (sample)", **LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values.")

# =========================================================================== #
# GROUP 1: EXPLORE
# =========================================================================== #
with top[1]:
    subtabs = st.tabs(['Distributions', 'Categorical', 'Time Series'])

    with subtabs[0]:
        # --------------------------------------------------------------------------- #
        # 2. DISTRIBUTIONS
        # --------------------------------------------------------------------------- #
        ctrl, view = st.columns([1, 4])

        with ctrl:
            var = st.selectbox("Variable", numeric_cols, key="dist_var")
            scale = st.radio("Scale", ["linear", "log", "log1p", "sqrt"], key="dist_scale")
            group = st.selectbox("Group by", ["(none)"] + categorical_cols,
                                 key="dist_group")
            max_groups = st.number_input("Max groups", 2, 20, 8, key="dist_maxg")

        MAX_DIST_SAMPLE = 50_000

        def _sampled(s: pd.Series) -> pd.Series:
            return s.sample(MAX_DIST_SAMPLE, random_state=0) if len(s) > MAX_DIST_SAMPLE else s

        with view:
            if not var:
                st.caption("Select a numeric variable.")
            else:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.28, 0.72], vertical_spacing=0.02)
                groups_data = None

                if group == "(none)":
                    s = transform_scale(_sampled(df[var]), scale)
                    fig.add_trace(go.Box(x=s, orientation="h", showlegend=False,
                                         marker_color=PALETTE[0]), row=1, col=1)
                    fig.add_trace(go.Histogram(x=s, showlegend=False, nbinsx=60,
                                               marker_color=PALETTE[0]), row=2, col=1)
                else:
                    top_cats = df[group].value_counts().head(int(max_groups)).index.tolist()
                    groups_data = []
                    for i, cat in enumerate(top_cats):
                        s = transform_scale(_sampled(df.loc[df[group] == cat, var]), scale)
                        color = PALETTE[i % len(PALETTE)]
                        fig.add_trace(go.Box(x=s, name=str(cat), orientation="h",
                                             showlegend=False, marker_color=color),
                                      row=1, col=1)
                        fig.add_trace(go.Histogram(x=s, name=str(cat), opacity=0.6,
                                                   nbinsx=60, marker_color=color),
                                      row=2, col=1)
                        groups_data.append(s.values)
                    fig.update_layout(barmode="overlay")

                xlabel = var + ("" if scale == "linear" else f"  ({scale})")
                fig.update_layout(height=400, **LAYOUT)
                fig.update_xaxes(title_text=xlabel, row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)
                if len(df) > MAX_DIST_SAMPLE:
                    st.caption(f"Showing a random sample of {MAX_DIST_SAMPLE:,} "
                              f"of {len(df):,} rows for performance.")

                if groups_data is not None:
                    valid = [g for g in groups_data if len(g) >= 3]
                    stat = p = name = None
                    test_error = None
                    try:
                        if len(valid) == 2:
                            stat, p = stats.mannwhitneyu(valid[0], valid[1])
                            name = "Mann-Whitney U"
                        elif len(valid) > 2:
                            stat, p = stats.kruskal(*valid)
                            name = "Kruskal-Wallis"
                    except ValueError as e:
                        test_error = str(e)

                    if p is not None:
                        verdict = ("Significant difference" if p < 0.05
                                   else "No significant difference")
                        box = st.success if p < 0.05 else st.warning
                        box(f"**{name}** · statistic = {stat:,.2f} · p = {p:.4g} → "
                            f"{verdict} (α = 0.05)")
                        st.caption("Non-parametric tests: they don't assume normality. "
                                   "With very large samples, tiny differences can "
                                   "come out significant.")
                    elif test_error:
                        st.caption(f"Test not applicable: {test_error}")
                    else:
                        st.warning("Not enough groups with data for the test.")

    with subtabs[1]:
        # --------------------------------------------------------------------------- #
        # 5. CATEGORICAL
        # --------------------------------------------------------------------------- #
        if not categorical_cols:
            st.caption("No categorical columns. Raise the threshold in the sidebar.")
        else:
            ctrl, view = st.columns([1, 4])
            with ctrl:
                cat = st.selectbox("Variable", categorical_cols, key="cat_var")
                topn = st.slider("Top N categories", 3, 30, 10, key="cat_topn")
                cross = st.selectbox("Cross with",
                                     ["(none)"] + [c for c in categorical_cols if c != cat],
                                     key="cat_cross")
                normalize = st.checkbox("Show as %", key="cat_norm")

            with view:
                vc = df[cat].value_counts(dropna=False).head(topn)
                v_left, v_right = st.columns(2)

                with v_left:
                    plot_df = pd.DataFrame({"category": vc.index.astype(str), "n": vc.values})
                    plot_df["%"] = (plot_df["n"] / len(df) * 100).round(1)
                    xcol = "%" if normalize else "n"
                    fig = px.bar(plot_df, x=xcol, y="category", orientation="h",
                                 text=xcol, color_discrete_sequence=PALETTE)
                    fig.update_layout(height=330,
                                      yaxis=dict(autorange="reversed", title=""),
                                      title=f"Frequencies · {cat}", **LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                with v_right:
                    if cross == "(none)":
                        card = pd.DataFrame({
                            "column": categorical_cols,
                            "unique": [safe_nunique(df[c]) for c in categorical_cols],
                        }).sort_values("unique", ascending=False)
                        fig = px.bar(card, x="unique", y="column", orientation="h",
                                     color_discrete_sequence=PALETTE)
                        fig.update_layout(height=330,
                                          yaxis=dict(autorange="reversed", title=""),
                                          title="Cardinality by column", **LAYOUT)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        ct = pd.crosstab(df[cat], df[cross])
                        ct = ct.loc[[i for i in vc.index if i in ct.index]]
                        show = ct.div(ct.sum(axis=1), axis=0) * 100 if normalize else ct
                        fig = px.imshow(show.round(1), text_auto=True, aspect="auto",
                                        color_continuous_scale="Blues")
                        fig.update_layout(height=330,
                                          title=f"{cat} × {cross}"
                                                + (" (% by row)" if normalize else ""),
                                          **LAYOUT)
                        st.plotly_chart(fig, use_container_width=True)

                if cross != "(none)":
                    ct_full = pd.crosstab(df[cat], df[cross])
                    if ct_full.shape[0] > 1 and ct_full.shape[1] > 1:
                        chi2, p, dof, _ = stats.chi2_contingency(ct_full)
                        n = ct_full.values.sum()
                        cramer = np.sqrt(chi2 / (n * (min(ct_full.shape) - 1)))
                        verdict = "Associated" if p < 0.05 else "Independent"
                        box = st.success if p < 0.05 else st.warning
                        box(f"**Chi-square** · χ² = {chi2:,.1f} · df = {dof} · "
                            f"p = {p:.4g} · Cramér's V = {cramer:.3f} → {verdict} (α = 0.05)")
                        st.caption("Cramér's V measures the strength of the association "
                                   "(0 = none, 1 = total); the p-value only tells you "
                                   "whether one exists.")

    with subtabs[2]:
        # --------------------------------------------------------------------------- #
        # 5. TIME SERIES
        # --------------------------------------------------------------------------- #
        if not datetime_cols:
            st.caption("No date columns detected. "
                       "You can mark them manually in the sidebar.")
        else:
            ctrl, view = st.columns([1, 4])
            with ctrl:
                tcol = st.selectbox("Date column", datetime_cols, key="ts_col")
                agg = st.selectbox("Metric", ["count", "sum", "mean", "median"],
                                   key="ts_agg")
                metric = (st.selectbox("Variable", numeric_cols, key="ts_metric")
                          if agg != "count" else None)
                freq_label = st.selectbox("Frequency", ["Hour", "Day", "Week", "Month"],
                                          index=1, key="ts_freq")
                ts_group = st.selectbox("Split by", ["(none)"] + categorical_cols,
                                        key="ts_group")

            freq = {"Hour": "h", "Day": "D", "Week": "W", "Month": "MS"}[freq_label]
            d = df[df[tcol].notna()].copy()

            if d.empty:
                st.warning("That column has no valid dates.")
            else:
                func = {"count": "size", "sum": "sum",
                        "mean": "mean", "median": "median"}[agg]

                with view:
                    top_row = st.columns([3, 1])

                    if ts_group == "(none)":
                        g = d.set_index(tcol).resample(freq)
                        serie = g.size() if agg == "count" else getattr(g[metric], func)()
                        serie = serie.reset_index(name="value")
                        fig = px.line(serie, x=tcol, y="value", markers=True,
                                      color_discrete_sequence=PALETTE)
                    else:
                        top_cats = d[ts_group].value_counts().head(8).index
                        d2 = d[d[ts_group].isin(top_cats)]
                        g = d2.set_index(tcol).groupby(ts_group).resample(freq)
                        serie = g.size() if agg == "count" else getattr(g[metric], func)()
                        serie = serie.reset_index(name="value")
                        fig = px.line(serie, x=tcol, y="value", color=ts_group,
                                      markers=True, color_discrete_sequence=PALETTE)

                    fig.update_layout(height=265,
                                      yaxis_title=f"{agg} of {metric or 'records'}",
                                      title=f"{agg.capitalize()} by {freq_label.lower()}",
                                      **LAYOUT)
                    with top_row[0]:
                        st.plotly_chart(fig, use_container_width=True)
                    with top_row[1]:
                        st.metric("From", f"{d[tcol].min():%Y-%m-%d}")
                        st.metric("To", f"{d[tcol].max():%Y-%m-%d}")
                        st.metric("Periods", f"{len(serie):,}")

                    s_left, s_right = st.columns(2)
                    with s_left:
                        by_hour = d.groupby(d[tcol].dt.hour).size().reset_index()
                        by_hour.columns = ["hour", "n"]
                        fig = px.bar(by_hour, x="hour", y="n",
                                     color_discrete_sequence=PALETTE)
                        fig.update_layout(height=195, title="By hour of day", **LAYOUT)
                        st.plotly_chart(fig, use_container_width=True)
                    with s_right:
                        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                        by_dow = d.groupby(d[tcol].dt.dayofweek).size().reset_index()
                        by_dow.columns = ["dow", "n"]
                        by_dow["day"] = by_dow["dow"].map(lambda i: days[i])
                        fig = px.bar(by_dow, x="day", y="n",
                                     color_discrete_sequence=PALETTE)
                        fig.update_layout(height=195, title="By day of week", **LAYOUT)
                        st.plotly_chart(fig, use_container_width=True)

# =========================================================================== #
# GROUP 2: RELATIONSHIPS
# =========================================================================== #
with top[2]:
    subtabs = st.tabs(['Correlation', 'Stratified Correlation', 'Pairplot', 'PCA', 'Outliers'])

    with subtabs[0]:
        # --------------------------------------------------------------------------- #
        # 3. CORRELATION
        # --------------------------------------------------------------------------- #
        corr_cols_all = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
        dropped_const = set(numeric_cols) - set(corr_cols_all)

        if len(corr_cols_all) < 2:
            st.caption("At least 2 non-constant numeric columns are needed.")
        else:
            corr_cols = st.multiselect("Variables to include", corr_cols_all,
                                       default=corr_cols_all, key="corr_vars")
            method = st.radio("Method", ["pearson", "spearman", "kendall"],
                              key="corr_method", horizontal=True)

            if len(corr_cols) < 2:
                st.caption("Choose at least 2 variables.")
            else:
                corr = _clustered_corr(df[corr_cols], method)

                # The matrix is symmetric — only show the lower triangle + diagonal.
                upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
                corr_display = corr.mask(upper)

                fig = px.imshow(corr_display.round(2), text_auto=True, aspect="auto",
                                color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                fig.update_layout(height=650,
                                  title=f"Correlation matrix ({method}, clustered)", **LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
                if dropped_const:
                    st.caption(f"Excluded (constant, correlation undefined): "
                               f"{', '.join(sorted(dropped_const))}.")

                cm = corr.where(~np.eye(len(corr), dtype=bool))
                pairs = (cm.stack().reset_index()
                         .rename(columns={"level_0": "var1", "level_1": "var2", 0: "r"}))
                pairs["abs"] = pairs["r"].abs()
                pairs = pairs.sort_values("abs", ascending=False).drop_duplicates("abs").head(5)
                if len(pairs):
                    txt = " · ".join(f"{r.var1}–{r.var2}: {r.r:.2f}" for r in pairs.itertuples())
                    st.caption(f"**Strongest correlations** — {txt}")

    with subtabs[1]:
        # --------------------------------------------------------------------------- #
        # 3b. STRATIFIED CORRELATION (does a confound hide a real relationship?)
        # --------------------------------------------------------------------------- #
        st.caption("Checks whether the pooled correlation between X and Y is misleading "
                  "because both are driven by a third variable (Z). Splits Z into quantile "
                  "buckets and recomputes the X-Y correlation *within* each bucket — a real "
                  "relationship often only shows up once Z is controlled for (Simpson's "
                  "paradox). Buckets are computed within each group's own data when split by "
                  "a second file/category, not on a shared global cutoff.")

        strat_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
        if len(strat_cols) < 3:
            st.caption("At least 3 non-constant numeric columns are needed (X, Y, Z).")
        else:
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                strat_x = st.selectbox("X", strat_cols, index=_default_index(strat_cols, "bet"),
                                       key="strat_x")
            with s2:
                y_opts = [c for c in strat_cols if c != strat_x]
                strat_y = st.selectbox("Y", y_opts, index=_default_index(y_opts, "game"),
                                       key="strat_y")
            with s3:
                z_opts = [c for c in strat_cols if c not in (strat_x, strat_y)]
                strat_z = st.selectbox("Stratify by (Z)", z_opts,
                                       index=_default_index(z_opts, "round"), key="strat_z")
            with s4:
                strat_nbuckets = st.slider("Quantile buckets", 3, 8, 5, key="strat_nbuckets")

            st.caption("Per-variable transform:")
            sc1, sc2, sc3 = st.columns(3)
            strat_scales, strat_abs = {}, {}
            for col_widget, var in [(sc1, strat_x), (sc2, strat_y), (sc3, strat_z)]:
                with col_widget:
                    st.markdown(f"*{var}*")
                    default_abs = (df[var].dropna() < 0).mean() > 0.5
                    strat_abs[var] = st.checkbox("Use |x| (sign-flipped column)", value=default_abs,
                                                 key=f"strat_abs_{var}")
                    strat_scales[var] = st.selectbox("Scale", ["linear", "log", "log1p", "sqrt"],
                                                     index=1 if strat_abs[var] else 0,
                                                     key=f"strat_scale_{var}")

            sp1, sp2 = st.columns(2)
            with sp1:
                strat_split_opts = ["(none)"] + categorical_cols
                strat_split_default = (strat_split_opts.index(source_col)
                                       if source_col in strat_split_opts else 0)
                strat_split_col = st.selectbox("Split by (optional)", strat_split_opts,
                                               index=strat_split_default, key="strat_split")
            with sp2:
                strat_maxg = st.number_input("Max groups", 2, 10, 6, key="strat_maxg")

            strat_base = df[[strat_x, strat_y, strat_z]].copy()
            for var in (strat_x, strat_y, strat_z):
                if strat_abs[var]:
                    strat_base[var] = strat_base[var].abs()

            if strat_split_col == "(none)":
                groups_to_plot = [("Total", strat_base)]
            else:
                split_vals = df.loc[strat_base.index, strat_split_col]
                top_groups = split_vals.value_counts().head(int(strat_maxg)).index.tolist()
                groups_to_plot = [(str(cat), strat_base[split_vals == cat]) for cat in top_groups]

            strat_fig = go.Figure()
            strat_fig.add_hline(y=0, line_color="black", line_width=0.8)
            all_tables = []
            any_plotted = False
            for i, (label, sub) in enumerate(groups_to_plot):
                result = _bucket_correlations(sub, strat_x, strat_y, strat_z,
                                              strat_scales, strat_nbuckets)
                if result is None:
                    continue
                (r_pooled, lo_p, hi_p, p_pooled, n_pooled), bucket_df = result
                color = PALETTE[i % len(PALETTE)]
                strat_fig.add_trace(go.Scatter(
                    x=bucket_df["position"], y=bucket_df["r"], mode="markers+lines",
                    error_y=dict(type="data", symmetric=False,
                                array=bucket_df["hi"] - bucket_df["r"],
                                arrayminus=bucket_df["r"] - bucket_df["lo"]),
                    marker=dict(size=9, color=color), line=dict(color=color, width=1.5),
                    name=f"{label} (pooled r={r_pooled:.2f}, n={n_pooled:,})"))
                bucket_df.insert(0, "group", label)
                all_tables.append(bucket_df)
                any_plotted = True

            if not any_plotted:
                st.warning("Not enough clean data (after the selected scales/transforms) to "
                          "compute this for any group.")
            else:
                strat_fig.update_xaxes(title_text=f"{strat_z} — quantile bucket (1 = lowest)",
                                       tickmode="linear", dtick=1)
                strat_fig.update_yaxes(title_text=f"Within-bucket correlation ({strat_x} vs. {strat_y})")
                strat_fig.update_layout(height=460,
                                        title=f"{strat_x} vs. {strat_y}, stratified by {strat_z}",
                                        **LAYOUT)
                st.plotly_chart(strat_fig, use_container_width=True)
                st.dataframe(pd.concat(all_tables, ignore_index=True), use_container_width=True,
                            hide_index=True)
                st.caption("If the pooled correlation (in each trace's legend label) looks weak "
                          "but the within-bucket points are consistently on one side of zero, "
                          f"{strat_z} is likely a confound hiding a real {strat_x}-{strat_y} "
                          "relationship.")

    with subtabs[2]:
        # --------------------------------------------------------------------------- #
        # 4. PAIRPLOT
        # --------------------------------------------------------------------------- #
        if len(numeric_cols) < 2:
            st.caption("At least 2 numeric columns are needed.")
        else:
            MAX_PAIR_VARS = 8
            c1, c2 = st.columns([3, 1])
            with c1:
                pair_vars = st.multiselect("Pairplot variables", numeric_cols,
                                           default=numeric_cols[:5],
                                           max_selections=MAX_PAIR_VARS, key="corr_pair")
            with c2:
                pair_color = st.selectbox("Color", ["(none)"] + categorical_cols,
                                          key="corr_color")

            PAIR_MAX_GROUPS = 8

            if len(pair_vars) < 2:
                st.caption("Choose at least 2 variables for the pairplot.")
            else:
                pair_cols = pair_vars[:MAX_PAIR_VARS]

                scale_cols = st.columns(len(pair_cols))
                col_scales = {}
                for sc, col in zip(scale_cols, pair_cols):
                    with sc:
                        col_scales[col] = st.selectbox(
                            col, ["linear", "log", "log1p", "sqrt"],
                            key=f"pair_scale_{col}")

                sample = df.sample(min(len(df), 2000), random_state=0)
                plot_data = transform_columns_mixed(sample, col_scales)

                if plot_data.empty:
                    st.warning("No rows left after applying the selected scales — "
                               "check log/sqrt on columns that can be zero or negative "
                               "(e.g. bet_amount is always <= 0).")
                else:
                    # Order variables by hierarchical clustering, same idea as the
                    # Correlation tab, so related variables sit next to each other.
                    # Constant columns (correlation undefined) are left at the end.
                    usable = [c for c in pair_cols if plot_data[c].nunique() > 1]
                    if len(usable) >= 2:
                        # clip(lower=0): see the Correlation tab for why this is needed
                        dist = (1 - plot_data[usable].corr().abs()).clip(lower=0)
                        condensed = squareform(dist.values, checks=False)
                        linkage = hierarchy.linkage(condensed, method="average")
                        order = hierarchy.dendrogram(linkage, no_plot=True)["leaves"]
                        pair_cols = ([usable[i] for i in order]
                                    + [c for c in pair_cols if c not in usable])
                    n = len(pair_cols)

                    if pair_color == "(none)":
                        cats = [None]
                        color_full = None
                    else:
                        color_full = sample.loc[plot_data.index, pair_color].astype(str)
                        cats = color_full.value_counts().head(PAIR_MAX_GROUPS).index.tolist()

                    # Built by hand instead of px.scatter_matrix: plotly's scatter_matrix
                    # diagonal is just the variable against itself (a straight line), it
                    # does not draw histograms — so real per-variable histograms on the
                    # diagonal need a manual grid. Sharing x-axes per column keeps the
                    # scale consistent down each column (every cell in a column plots
                    # the same variable on x, including the diagonal histogram).
                    # Y-axes are NOT shared per row: the diagonal cell's y-axis is a
                    # count (histogram), not the variable's value like the scatter
                    # cells in that row — sharing it would squash the scatter panels.
                    fig = make_subplots(rows=n, cols=n, horizontal_spacing=0.02,
                                        vertical_spacing=0.02, shared_xaxes=True)
                    legend_shown = set()
                    for i, yi in enumerate(pair_cols):
                        for j, xj in enumerate(pair_cols):
                            if j > i:
                                continue
                            for k, cat in enumerate(cats):
                                idx = (plot_data.index if cat is None else
                                       plot_data.index[(color_full.loc[plot_data.index] == cat).values])
                                col_color = PALETTE[k % len(PALETTE)]
                                label = str(cat) if cat is not None else "Total"
                                show_legend = label not in legend_shown
                                legend_shown.add(label)
                                if i == j:
                                    fig.add_trace(go.Histogram(
                                        x=plot_data.loc[idx, xj], marker_color=col_color,
                                        name=label, legendgroup=label,
                                        showlegend=show_legend, opacity=0.6),
                                        row=i + 1, col=j + 1)
                                else:
                                    fig.add_trace(go.Scatter(
                                        x=plot_data.loc[idx, xj], y=plot_data.loc[idx, yi],
                                        mode="markers", marker=dict(size=5, color=col_color, opacity=0.6),
                                        name=label, legendgroup=label, showlegend=False),
                                        row=i + 1, col=j + 1)

                            # Trend line + correlation annotation, on the whole panel
                            # (all groups combined), not per color group.
                            if i != j:
                                x_all, y_all = plot_data[xj].values, plot_data[yi].values
                                if len(x_all) >= 2 and np.std(x_all) > 0 and np.std(y_all) > 0:
                                    m, b = np.polyfit(x_all, y_all, 1)
                                    xs = np.array([x_all.min(), x_all.max()])
                                    fig.add_trace(go.Scatter(
                                        x=xs, y=m * xs + b, mode="lines",
                                        line=dict(color="black", width=1, dash="dash"),
                                        showlegend=False, hoverinfo="skip"),
                                        row=i + 1, col=j + 1)
                                    r = np.corrcoef(x_all, y_all)[0, 1]
                                    fig.add_annotation(
                                        text=f"r={r:.2f}", xref="x domain", yref="y domain",
                                        x=0.05, y=0.95, xanchor="left", yanchor="top",
                                        showarrow=False, font=dict(size=8),
                                        row=i + 1, col=j + 1)

                    for i in range(n):
                        for j in range(n):
                            if j > i:
                                fig.update_xaxes(visible=False, row=i + 1, col=j + 1)
                                fig.update_yaxes(visible=False, row=i + 1, col=j + 1)
                            else:
                                fig.update_xaxes(tickfont=dict(size=8), row=i + 1, col=j + 1)
                                fig.update_yaxes(tickfont=dict(size=8), row=i + 1, col=j + 1)
                        y_suffix = "" if col_scales[pair_cols[i]] == "linear" else f" ({col_scales[pair_cols[i]]})"
                        fig.update_yaxes(title_text=pair_cols[i] + y_suffix, title_font=dict(size=9),
                                         row=i + 1, col=1)
                    for j in range(n):
                        x_suffix = "" if col_scales[pair_cols[j]] == "linear" else f" ({col_scales[pair_cols[j]]})"
                        fig.update_xaxes(title_text=pair_cols[j] + x_suffix, title_font=dict(size=9),
                                         row=n, col=j + 1)

                    fig.update_layout(height=140 * n + 100, barmode="overlay",
                                      showlegend=(color_full is not None),
                                      title=f"Pairplot (max. {MAX_PAIR_VARS} vars, clustered)",
                                      **LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
                    dropped = len(sample) - len(plot_data)
                    if dropped and any(s != "linear" for s in col_scales.values()):
                        st.caption(f"{dropped} rows out of domain for the selected "
                                   f"per-variable scales were excluded.")

    with subtabs[3]:
        # --------------------------------------------------------------------------- #
        # 6. PCA
        # --------------------------------------------------------------------------- #
        if len(numeric_cols) < 2:
            st.caption("At least 2 numeric columns are needed.")
        else:
            ctrl, view = st.columns([1, 4])
            with ctrl:
                pca_vars = st.multiselect("Variables", numeric_cols,
                                          default=numeric_cols, key="pca_vars")
                color_by = st.selectbox("Color by", ["(none)"] + categorical_cols,
                                        key="pca_color")
                st.caption("Complete rows, standard scaling.")

            if len(pca_vars) < 2:
                st.caption("Select at least 2 variables.")
            else:
                data = df[pca_vars].replace([np.inf, -np.inf], np.nan).dropna()
                if len(data) < 3:
                    st.warning(f"Only {len(data)} complete rows (no nulls) in "
                               "the chosen variables. PCA needs more rows; "
                               "try other columns or more data.")
                else:
                    # n_comp bounded by rows and columns: can't exceed min(n, p)
                    n_comp = min(len(pca_vars), 10, len(data))
                    scores, explained_variance_ratio, components = _fit_pca(data, n_comp)

                    with view:
                        p1, p2 = st.columns(2)
                        with p1:
                            color = (df.loc[data.index, color_by].astype(str)
                                     if color_by != "(none)" else None)
                            var1, var2 = explained_variance_ratio[:2] * 100
                            fig = px.scatter(x=scores[:, 0], y=scores[:, 1], color=color,
                                             opacity=0.6, color_discrete_sequence=PALETTE)
                            fig.update_traces(marker=dict(size=5))
                            fig.update_layout(height=345, title="PC1–PC2 projection",
                                              xaxis_title=f"PC1 ({var1:.1f}%)",
                                              yaxis_title=f"PC2 ({var2:.1f}%)", **LAYOUT)
                            st.plotly_chart(fig, use_container_width=True)

                        with p2:
                            vr = explained_variance_ratio
                            labels = [f"PC{i + 1}" for i in range(n_comp)]
                            fig = go.Figure()
                            fig.add_bar(x=labels, y=vr, name="Individual",
                                        marker_color=PALETTE[0])
                            fig.add_scatter(x=labels, y=np.cumsum(vr), name="Cumulative",
                                            mode="lines+markers", marker_color=PALETTE[1])
                            fig.add_hline(y=0.8, line_dash="dot", line_color="grey")
                            fig.update_layout(height=345, title="Explained variance",
                                              yaxis_tickformat=".0%", **LAYOUT)
                            st.plotly_chart(fig, use_container_width=True)

                        load = pd.DataFrame(components[:2].T, index=pca_vars,
                                            columns=["PC1", "PC2"]).round(3)
                        st.caption("**Loadings** — " + " · ".join(
                            f"{i}: PC1={r.PC1}, PC2={r.PC2}" for i, r in load.iterrows()))

    with subtabs[4]:
        # --------------------------------------------------------------------------- #
        # 7. OUTLIERS
        # --------------------------------------------------------------------------- #
        if not numeric_cols:
            st.caption("No numeric columns.")
        else:
            ctrl, view = st.columns([1, 4])
            with ctrl:
                method = st.radio("Method", ["IQR (boxplot)", "Z-score", "Mahalanobis",
                                             "Isolation Forest"], key="out_method")

            if method in ("IQR (boxplot)", "Z-score"):
                with ctrl:
                    var = st.selectbox("Variable", numeric_cols, key="out_var")
                s = df[var].dropna()

                if method == "IQR (boxplot)":
                    with ctrl:
                        k = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1, key="out_k")
                    q1, q3 = s.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lo, hi = q1 - k * iqr, q3 + k * iqr
                else:
                    with ctrl:
                        z = st.slider("|z| threshold", 2.0, 5.0, 3.0, 0.1, key="out_z")
                    lo, hi = s.mean() - z * s.std(), s.mean() + z * s.std()

                mask = (s < lo) | (s > hi)
                with view:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        row_heights=[0.3, 0.7], vertical_spacing=0.03)
                    fig.add_trace(go.Box(x=s, orientation="h", showlegend=False,
                                         marker_color=PALETTE[0]), row=1, col=1)
                    fig.add_trace(go.Histogram(x=s, nbinsx=60, showlegend=False,
                                               marker_color=PALETTE[0]), row=2, col=1)
                    for b in (lo, hi):
                        fig.add_vline(x=b, line_dash="dash", line_color="crimson")
                    fig.update_layout(height=340, **LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f"Outliers: **{int(mask.sum()):,}** ({mask.mean() * 100:.2f}%) "
                            f"· bounds [{lo:,.2f}, {hi:,.2f}]")

            elif method == "Mahalanobis":
                with ctrl:
                    mvars = st.multiselect("Variables", numeric_cols,
                                           default=numeric_cols, key="out_mvars")
                    alpha = st.slider("χ² quantile", 0.90, 0.999, 0.975, 0.005, key="out_alpha")
                if len(mvars) < 2:
                    st.caption("Select at least 2 variables.")
                else:
                    data = df[mvars].replace([np.inf, -np.inf], np.nan).dropna()
                    md = _mahalanobis_distances(data)
                    thr = np.sqrt(stats.chi2.ppf(alpha, df=len(mvars)))
                    mask = md > thr
                    with view:
                        m1, m2 = st.columns(2)
                        with m1:
                            fig = px.histogram(x=md, nbins=60,
                                               color_discrete_sequence=PALETTE,
                                               labels={"x": "Mahalanobis distance"})
                            fig.add_vline(x=thr, line_dash="dash", line_color="crimson",
                                          annotation_text=f"χ² {alpha:.1%}")
                            fig.update_layout(height=330,
                                              title="Distance distribution", **LAYOUT)
                            st.plotly_chart(fig, use_container_width=True)
                        with m2:
                            if len(data) >= 3:
                                sc, _, _ = _fit_pca(data, 2)
                                fig = px.scatter(x=sc[:, 0], y=sc[:, 1],
                                                 color=np.where(mask, "outlier", "normal"),
                                                 opacity=0.6,
                                                 color_discrete_map={"outlier": "crimson",
                                                                     "normal": PALETTE[0]},
                                                 labels={"x": "PC1", "y": "PC2"})
                                fig.update_traces(marker=dict(size=5))
                                fig.update_layout(height=330, title="PCA view", **LAYOUT)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.caption("Too few rows for the PCA view.")
                        st.info(f"Multivariate outliers: **{int(mask.sum()):,}** "
                                f"({mask.mean() * 100:.2f}%) · threshold = {thr:.2f}")

            else:  # Isolation Forest
                with ctrl:
                    ivars = st.multiselect("Variables", numeric_cols,
                                           default=numeric_cols, key="out_ivars")
                    contamination = st.slider("Contamination", 0.01, 0.25, 0.05, 0.01,
                                              key="out_contam")
                if not ivars:
                    st.caption("Select at least 1 variable.")
                else:
                    data = df[ivars].replace([np.inf, -np.inf], np.nan).dropna()
                    mask, scores_if = _fit_isolation_forest(data, contamination)
                    with view:
                        i1, i2 = st.columns(2)
                        with i1:
                            fig = px.histogram(x=scores_if, nbins=60,
                                               color_discrete_sequence=PALETTE,
                                               labels={"x": "Anomaly score"})
                            fig.update_layout(height=330, title="Anomaly scores", **LAYOUT)
                            st.plotly_chart(fig, use_container_width=True)
                        with i2:
                            if len(ivars) >= 2 and len(data) >= 3:
                                sc, _, _ = _fit_pca(data, 2)
                                fig = px.scatter(x=sc[:, 0], y=sc[:, 1],
                                                 color=np.where(mask, "outlier", "normal"),
                                                 opacity=0.6,
                                                 color_discrete_map={"outlier": "crimson",
                                                                     "normal": PALETTE[0]},
                                                 labels={"x": "PC1", "y": "PC2"})
                                fig.update_traces(marker=dict(size=5))
                                fig.update_layout(height=330, title="PCA view", **LAYOUT)
                                st.plotly_chart(fig, use_container_width=True)
                        st.info(f"Outliers: **{int(mask.sum()):,}** "
                                f"({mask.mean() * 100:.2f}%)")

# =========================================================================== #
# GROUP 3: CHURN
# =========================================================================== #
with top[3]:
    subtabs = st.tabs(['Definition', 'Survival'])

    with subtabs[0]:
        # --------------------------------------------------------------------------- #
        # 12. CHURN (data-driven reconnection threshold from raw daily-activity rows)
        # --------------------------------------------------------------------------- #
        st.caption("Computes a data-driven churn threshold: how long players who DO return "
                  "normally take (a percentile of that gap distribution), then flags anyone "
                  "currently past that threshold as churned. Expects one row per subject per "
                  "active period, e.g. daily_player_metrics.parquet — not a pre-aggregated "
                  "one-row-per-subject table like churn_labels.parquet.")

        all_cols = df.columns.tolist()
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            churn_id_col = st.selectbox("Subject ID", all_cols,
                                        index=_default_index(all_cols, "party"), key="churn_id")
        with c2:
            date_candidates = datetime_cols or all_cols
            churn_date_col = st.selectbox("Date", date_candidates,
                                          index=_default_index(date_candidates, "date"), key="churn_date")
        with c3:
            churn_activity_col = st.selectbox("Activity column (optional, >0 = active)",
                                              ["(every row is active)"] + numeric_cols, key="churn_activity")
        with c4:
            churn_group_opts = ["(none)"] + categorical_cols
            churn_group_default = churn_group_opts.index(source_col) if source_col in churn_group_opts else 0
            churn_group_col = st.selectbox("Split by (optional)", churn_group_opts,
                                           index=churn_group_default, key="churn_def_group")
        with c5:
            churn_max_groups = st.number_input("Max groups", 2, 10, 6, key="churn_def_maxg")

        churn_pctl = st.slider("Reconnection percentile (threshold)", 0.50, 0.99, 0.95, 0.01,
                               key="churn_pctl")

        dates = pd.to_datetime(df[churn_date_col], errors="coerce")
        active_mask = dates.notna()
        if churn_activity_col != "(every row is active)":
            active_mask &= df[churn_activity_col] > 0

        active_df = pd.DataFrame({
            "subject": df.loc[active_mask, churn_id_col],
            "date": dates[active_mask].dt.normalize(),
        }).drop_duplicates()

        if active_df.empty or active_df["subject"].nunique() < 2:
            st.warning("Not enough active rows with the selected columns.")
        else:
            pooled = _churn_definition(active_df, churn_pctl)
            if pooled is None:
                st.warning("Not enough subjects with 2+ active dates to estimate a "
                          "reconnection gap distribution.")
            else:
                per_subject, gaps, threshold_days = pooled

                k1, k2, k3 = st.columns(3)
                k1.metric("Threshold", f"{threshold_days:.0f} days")
                k2.metric("Churn rate", f"{per_subject['churn'].mean() * 100:.1f}%")
                k3.metric("Subjects", f"{len(per_subject):,}")

                churn_fig = make_subplots(rows=1, cols=2, subplot_titles=(
                    "How long do returning subjects normally take?", "Final churn classification"))

                gap_counts = gaps.value_counts().sort_index()
                churn_fig.add_trace(go.Bar(x=gap_counts.index, y=gap_counts.values,
                                           marker_color=PALETTE[0], showlegend=False), row=1, col=1)
                churn_fig.add_vline(x=threshold_days, line_dash="dash", line_color=PALETTE[5], row=1, col=1)

                for label, sub, color in [("Active/censored", per_subject[per_subject["churn"] == 0], PALETTE[0]),
                                          ("Churn", per_subject[per_subject["churn"] == 1], PALETTE[5])]:
                    vc = sub["trailing_inactive"].value_counts().sort_index()
                    churn_fig.add_trace(go.Bar(x=vc.index, y=vc.values, name=label,
                                               marker_color=color), row=1, col=2)
                churn_fig.add_vline(x=threshold_days - 0.5, line_dash="dash", line_color="black", row=1, col=2)

                churn_fig.update_xaxes(title_text="Days between active periods", row=1, col=1)
                churn_fig.update_yaxes(title_text="Returns observed", row=1, col=1)
                churn_fig.update_xaxes(title_text="Days inactive at cutoff", row=1, col=2)
                churn_fig.update_yaxes(title_text="Subjects", row=1, col=2)
                churn_fig.update_layout(height=460, title=f"Churn definition (percentile={churn_pctl:.2f})", **LAYOUT)
                st.plotly_chart(churn_fig, use_container_width=True)

                st.caption(f"{len(gaps):,} observed returns from "
                          f"{(per_subject['n_active_periods'] >= 2).sum():,} subjects with 2+ active "
                          f"periods. Threshold = percentile {churn_pctl:.0%} of that return-gap "
                          f"distribution, in the same date units as the Date column (days, if a "
                          f"calendar date).")

                if churn_group_col != "(none)":
                    group_vals = df.loc[active_df.index, churn_group_col]
                    top_groups = group_vals.value_counts().head(int(churn_max_groups)).index.tolist()
                    cmp_rows = []
                    for cat in top_groups:
                        result = _churn_definition(active_df[group_vals == cat], churn_pctl)
                        if result is None:
                            continue
                        per_subject_g, gaps_g, threshold_g = result
                        cmp_rows.append({
                            "Group": str(cat),
                            "Threshold (days)": round(float(threshold_g), 1),
                            "Churn rate (%)": round(per_subject_g["churn"].mean() * 100, 1),
                            "Subjects": len(per_subject_g),
                            "Observed returns": len(gaps_g),
                        })
                    st.markdown(f"**By {churn_group_col}**")
                    if not cmp_rows:
                        st.caption(f"Not enough data per group of {churn_group_col} to compute "
                                  f"a definition for any group.")
                    else:
                        cmp_df = pd.DataFrame(cmp_rows)
                        st.dataframe(cmp_df, use_container_width=True, hide_index=True)
                        b1, b2 = st.columns(2)
                        with b1:
                            fig_thr = px.bar(cmp_df, x="Group", y="Threshold (days)", color="Group",
                                             color_discrete_sequence=PALETTE)
                            fig_thr.update_layout(height=280, showlegend=False, **LAYOUT)
                            st.plotly_chart(fig_thr, use_container_width=True)
                        with b2:
                            fig_rate = px.bar(cmp_df, x="Group", y="Churn rate (%)", color="Group",
                                              color_discrete_sequence=PALETTE)
                            fig_rate.update_layout(height=280, showlegend=False, **LAYOUT)
                            st.plotly_chart(fig_rate, use_container_width=True)

    with subtabs[1]:
        # --------------------------------------------------------------------------- #
        # 8. SURVIVAL (Kaplan-Meier)
        # --------------------------------------------------------------------------- #
        if not numeric_cols:
            st.caption("No numeric columns to use as duration.")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                duration_col = st.selectbox("Duration", numeric_cols,
                                            index=_default_index(numeric_cols, "duration"),
                                            key="surv_duration")
            with c2:
                event_opts = [c for c in df.columns if safe_nunique(df[c]) <= 2] or df.columns.tolist()
                event_col = st.selectbox("Event (1 = event, 0 = censored)", event_opts,
                                         index=_default_index(event_opts, "event"),
                                         key="surv_event")
            with c3:
                group_col = st.selectbox("Split by (optional)", ["(none)"] + categorical_cols,
                                         key="surv_group")
            with c4:
                max_groups = st.number_input("Max groups", 2, 10, 6, key="surv_maxg")
            with c5:
                n_show = st.slider("Nº players (lifetimes)", 5, 300, 60, key="surv_n_lifetimes")

            d = df[[duration_col, event_col]].apply(pd.to_numeric, errors="coerce")
            d.columns = ["duration", "event"]
            d = d.dropna()

            if d.empty:
                st.warning("No valid rows with those columns.")
            else:
                curves = None
                km_fig = go.Figure()
                summary_rows = []

                def _add_km_curve(sub, color, label, showlegend=True):
                    sf, ci, median, median_lo, median_hi, rmst = _fit_km(
                        sub["duration"], sub["event"])
                    km_fig.add_trace(go.Scatter(
                        x=sf.index, y=ci.iloc[:, 0], mode="lines", line_shape="hv",
                        line=dict(width=0), showlegend=False, hoverinfo="skip"))
                    km_fig.add_trace(go.Scatter(
                        x=sf.index, y=ci.iloc[:, 1], mode="lines", line_shape="hv",
                        line=dict(width=0), fill="tonexty", fillcolor=_rgba(color, 0.2),
                        showlegend=False, hoverinfo="skip"))
                    km_fig.add_trace(go.Scatter(
                        x=sf.index, y=sf.values, mode="lines", line_shape="hv",
                        name=label, line_color=color, showlegend=showlegend))

                    if np.isfinite(median):
                        median_str = f"{median:.4g}"
                        median_ci = f"[{median_lo:.4g}, {median_hi:.4g}]"
                    else:
                        median_str, median_ci = "not reached", "—"
                    summary_rows.append({
                        "Group": label,
                        "Subjects": len(sub),
                        "Events": int(sub["event"].sum()),
                        "Censored": int((sub["event"] == 0).sum()),
                        "Median": median_str,
                        "95% CI median": median_ci,
                        # bounded to this group's own last observed time: without `t=`,
                        # lifelines defaults to t=inf, which returns literally "inf" the
                        # moment the curve's last segment is censored (never reaches 0) --
                        # exactly the case here, since churn can't happen past duration=4
                        # in an 8-day window but plenty of subjects are censored out to 7.
                        "RMST": f"{rmst:.4g}",
                    })

                if group_col == "(none)":
                    _add_km_curve(d, PALETTE[0], "Total", showlegend=False)
                else:
                    group = df.loc[d.index, group_col]
                    top_cats = group.value_counts().head(int(max_groups)).index
                    curves = group.isin(top_cats)
                    for i, cat in enumerate(top_cats):
                        sub = d[group == cat]
                        if len(sub) < 2:
                            continue
                        _add_km_curve(sub, PALETTE[i % len(PALETTE)], str(cat))

                km_fig.update_layout(height=420, title="Kaplan-Meier curve",
                                     xaxis_title=duration_col,
                                     yaxis_title="Survival probability",
                                     yaxis_range=[0, 1.02], **LAYOUT)

                # Lifetimes: each player is a line from 0 to their duration (red = event,
                # with a dot at the end; blue = censored, no dot: "still alive").
                sample = d.sample(min(n_show, len(d)), random_state=0).sort_values(
                    "duration").reset_index(drop=True)
                y = np.arange(len(sample))
                observed = sample["event"] == 1

                def _lines(mask):
                    xs, ys = [], []
                    for i in np.flatnonzero(mask.values):
                        xs += [0, sample["duration"].iloc[i], None]
                        ys += [y[i], y[i], None]
                    return xs, ys

                life_fig = go.Figure()
                xs, ys = _lines(~observed)
                if xs:
                    life_fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines", line=dict(color="#348ABD", width=1.5),
                        name="Censored", hoverinfo="skip"))
                xs, ys = _lines(observed)
                if xs:
                    life_fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines", line=dict(color="#A60628", width=1.5),
                        name="Event", hoverinfo="skip"))
                    life_fig.add_trace(go.Scatter(
                        x=sample.loc[observed, "duration"], y=y[observed.values],
                        mode="markers", marker=dict(color="#A60628", size=5),
                        showlegend=False, hoverinfo="skip"))
                life_fig.update_layout(height=420, title="Individual lifetimes",
                                       xaxis_title=duration_col,
                                       yaxis=dict(showticklabels=False,
                                                  title=f"Players (sample of {len(sample)})"),
                                       **LAYOUT)

                c_left, c_right = st.columns(2)
                with c_left:
                    st.plotly_chart(km_fig, use_container_width=True)
                with c_right:
                    st.plotly_chart(life_fig, use_container_width=True)
                st.caption("Lifetimes: red = observed event (probable churn) · "
                           "blue = censored (still active when the data was cut).")

                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True,
                            hide_index=True)
                st.caption("Median = duration at which half of the subjects have "
                           "experienced the event (95% CI). RMST = restricted mean "
                           "survival time (area under the curve up to the last "
                           "observed instant).")

                if curves is not None and curves.sum() >= 2 and group[curves].nunique() >= 2:
                    res = multivariate_logrank_test(d.loc[curves, "duration"],
                                                    group[curves], d.loc[curves, "event"])
                    verdict = ("Significant difference" if res.p_value < 0.05
                               else "No significant difference")
                    box = st.success if res.p_value < 0.05 else st.warning
                    box(f"**Log-rank test** · statistic = {res.test_statistic:,.2f} · "
                        f"p = {res.p_value:.4g} → {verdict} (α = 0.05)")


# =========================================================================== #
# GROUP 4: PREDICT
# =========================================================================== #
with top[4]:
    subtabs = st.tabs(['ROC', 'Trajectory', 'Flips'])

    with subtabs[0]:
        # --------------------------------------------------------------------------- #
        # 10. ROC (univariate signal per variable + optional combined score)
        # --------------------------------------------------------------------------- #
        ROC_MAX_OVERLAY = 10
        ROC_MAX_LEVELS = 20  # cap on distinct values for a categorical to be level-expanded

        roc_target_opts = [c for c in df.columns if safe_nunique(df[c]) == 2]
        if not roc_target_opts:
            st.caption("No binary column available to use as target.")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                roc_target = st.selectbox("Target (binary)", roc_target_opts,
                                          index=_default_index(roc_target_opts, "churn"),
                                          key="roc_target")
            with c2:
                roc_topn = st.slider("Curves to overlay", 1, ROC_MAX_OVERLAY, 6, key="roc_topn")

            roc_values = sorted(df[roc_target].dropna().unique(), key=str)
            roc_positive = roc_values[-1]
            y_roc = (df[roc_target] == roc_positive).astype(int)

            # Scorable items: numeric columns as-is, plus one 0/1 indicator per level
            # of each categorical column (mirrors what a one-hot-encoded model would
            # see) -- same idea as churnpipeline's target association tab. The scoring
            # loop itself (cached) is the expensive part; series/mask are cheap to
            # rebuild here for whichever rows actually get used below.
            auc_table = _auc_table(df, roc_target, numeric_cols, categorical_cols, ROC_MAX_LEVELS)
            roc_scored = []
            for row in auc_table.itertuples():
                s = df[row.col] if row.level is None else (df[row.col] == row.level).astype(int)
                mask = s.notna() & y_roc.notna()
                roc_scored.append(dict(label=row.label, series=s, mask=mask,
                                       auc=row.auc, inverted=row.inverted))

            if not roc_scored:
                st.caption("No feature has enough variation to score.")
            else:
                roc_scored.sort(key=lambda r: r["auc"], reverse=True)

                roc_fig = go.Figure()
                roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                  line=dict(color="#c3c2b7", dash="dash", width=1))
                for i, row in enumerate(roc_scored[:roc_topn]):
                    score = -row["series"] if row["inverted"] else row["series"]
                    fpr, tpr, _ = roc_curve(y_roc[row["mask"]], score[row["mask"]])
                    arrow = "↓" if row["inverted"] else "↑"
                    roc_fig.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        name=f"{arrow} {row['label']} ({row['auc']:.3f})",
                        line=dict(color=PALETTE[i % len(PALETTE)], width=2)))

                st.markdown("**Combined score (optional)**")
                combine_vars = st.multiselect(
                    "Variables to combine into one logistic-regression score",
                    [c for c in numeric_cols if c != roc_target], key="roc_combine")

                combined_ok = False
                if combine_vars:
                    scale_cols_roc = st.columns(len(combine_vars))
                    combine_scales = {}
                    for sc, col in zip(scale_cols_roc, combine_vars):
                        with sc:
                            combine_scales[col] = st.selectbox(
                                col, ["linear", "log", "log1p", "sqrt"], key=f"roc_scale_{col}")

                    X_roc = transform_columns_mixed(df, combine_scales)
                    common_idx = X_roc.index.intersection(y_roc.dropna().index)
                    X_roc = X_roc.loc[common_idx]
                    y_roc_sub = y_roc.loc[common_idx]

                    if len(X_roc) < 10 or y_roc_sub.nunique() < 2:
                        st.warning("Not enough valid rows for the selected scales/target "
                                  "to fit a combined score.")
                    else:
                        Xs = StandardScaler().fit_transform(X_roc.values)
                        clf = LogisticRegression().fit(Xs, y_roc_sub.values)
                        combined_score = clf.decision_function(Xs)
                        combined_auc = roc_auc_score(y_roc_sub, combined_score)
                        inverted_c = combined_auc < 0.5
                        if inverted_c:
                            combined_auc = 1 - combined_auc
                            combined_score = -combined_score
                        fpr_c, tpr_c, _ = roc_curve(y_roc_sub, combined_score)
                        arrow_c = "↓" if inverted_c else "↑"
                        roc_fig.add_trace(go.Scatter(
                            x=fpr_c, y=tpr_c, mode="lines",
                            name=f"{arrow_c} Combined ({len(combine_vars)} vars) ({combined_auc:.3f})",
                            line=dict(color="black", width=3.5)))

                        coef_df = pd.DataFrame({
                            "variable": combine_vars,
                            "scale": [combine_scales[v] for v in combine_vars],
                            "standardized coef.": clf.coef_[0],
                        }).sort_values("standardized coef.", key=abs, ascending=False)
                        combined_ok = True

                        dropped = len(df) - len(X_roc)
                        if dropped:
                            st.caption(f"{dropped} rows out of domain for the selected "
                                      f"per-variable scales were excluded from the combined score.")

                roc_fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                                      xaxis_range=[0, 1], yaxis_range=[0, 1],
                                      title=f"ROC — target: {roc_target} = {roc_positive}",
                                      height=560, **LAYOUT)
                st.plotly_chart(roc_fig, use_container_width=True)
                st.caption("Each individual curve scores that variable's raw value directly — no "
                          "classifier, no scale dependence (ROC only depends on rank order, so linear "
                          "vs. log gives the exact same curve for a single variable). The combined "
                          "curve is different: it fits a logistic regression on the selected variables "
                          "*with the scale chosen per variable*, so scale choice changes that curve.")

                leak_suspects = [r["label"] for r in roc_scored[:roc_topn] if r["auc"] > 0.95]
                if leak_suspects:
                    st.warning("AUC > 0.95 for " + ", ".join(leak_suspects) + " — check whether "
                              "this variable was used to *construct* the target itself (e.g. a "
                              "survival/inactivity column feeding a churn label). That gives a "
                              "near-perfect but circular score, not a real early signal.")

                if combined_ok:
                    st.dataframe(coef_df, use_container_width=True, hide_index=True)

                st.markdown("**Per-group comparison (optional)**")
                g1, g2, g3 = st.columns([2, 1, 1])
                with g1:
                    roc_group_var = st.selectbox("Variable", [r["label"] for r in roc_scored],
                                                 key="roc_group_var")
                with g2:
                    roc_group_opts = ["(none)"] + categorical_cols
                    roc_group_default = roc_group_opts.index(source_col) if source_col in roc_group_opts else 0
                    roc_group_col = st.selectbox("Split by", roc_group_opts,
                                                 index=roc_group_default, key="roc_group_col")
                with g3:
                    roc_group_maxg = st.number_input("Max groups", 2, 10, 6, key="roc_group_maxg")

                if roc_group_col != "(none)":
                    row = next(r for r in roc_scored if r["label"] == roc_group_var)
                    score = -row["series"] if row["inverted"] else row["series"]
                    top_g = df.loc[row["mask"], roc_group_col].value_counts().head(
                        int(roc_group_maxg)).index.tolist()
                    group_fig = go.Figure()
                    group_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                        line=dict(color="#c3c2b7", dash="dash", width=1))
                    any_group = False
                    for i, cat in enumerate(top_g):
                        sub_mask = row["mask"] & (df[roc_group_col] == cat)
                        if sub_mask.sum() < 10 or y_roc[sub_mask].nunique() < 2:
                            continue
                        fpr, tpr, _ = roc_curve(y_roc[sub_mask], score[sub_mask])
                        auc_g = roc_auc_score(y_roc[sub_mask], score[sub_mask])
                        group_fig.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode="lines",
                            name=f"{cat} ({auc_g:.3f}, n={int(sub_mask.sum()):,})",
                            line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
                        any_group = True
                    if not any_group:
                        st.caption(f"Not enough data per group of {roc_group_col} to score "
                                  f"{roc_group_var} for any group.")
                    else:
                        group_fig.update_layout(xaxis_title="False positive rate",
                                                yaxis_title="True positive rate",
                                                xaxis_range=[0, 1], yaxis_range=[0, 1],
                                                title=f"{roc_group_var} — by {roc_group_col}",
                                                height=420, **LAYOUT)
                        st.plotly_chart(group_fig, use_container_width=True)

    with subtabs[1]:
        # --------------------------------------------------------------------------- #
        # 11. TRAJECTORY (AUC evolution across "<name>_dayN" checkpoint columns)
        # --------------------------------------------------------------------------- #
        DAY_COL_RE = re.compile(r"^(.+)_day(\d+)$")
        day_families = {}
        for col in numeric_cols:
            m = DAY_COL_RE.match(col)
            if m:
                base, day_n = m.group(1), int(m.group(2))
                day_families.setdefault(base, {})[day_n] = col

        if not day_families:
            st.caption("No columns matching the '<name>_day<N>' pattern were found — this tab "
                      "expects checkpoint columns like avg_bet_amount_day1, cum_avg_bet_day2, "
                      "days_since_last_active_day3, etc.")
        else:
            traj_target_opts = [c for c in df.columns if safe_nunique(df[c]) == 2]
            if not traj_target_opts:
                st.caption("No binary column available to use as target.")
            else:
                c1, c2 = st.columns([1, 2])
                with c1:
                    traj_target = st.selectbox("Target (binary)", traj_target_opts,
                                               index=_default_index(traj_target_opts, "churn"),
                                               key="traj_target")
                with c2:
                    family_opts = sorted(day_families, key=lambda b: -len(day_families[b]))
                    traj_families = st.multiselect(
                        "Day-by-day variables to track", family_opts,
                        default=family_opts[:4], key="traj_families")
                show_combined = st.checkbox(
                    "Add combined score (logistic regression across selected variables, "
                    "at checkpoints where 2+ of them have data)", value=True, key="traj_combined")

                if not traj_families:
                    st.caption("Pick at least one variable family to plot.")
                else:
                    traj_values = sorted(df[traj_target].dropna().unique(), key=str)
                    traj_positive = traj_values[-1]
                    y_traj = (df[traj_target] == traj_positive).astype(int)

                    all_days = sorted({d for b in traj_families for d in day_families[b]})
                    per_day_cols = {d: {} for d in all_days}  # day -> {base: column name}
                    leak_suspects_traj = []

                    traj_fig = go.Figure()
                    for i, base in enumerate(traj_families):
                        days_avail = sorted(day_families[base])
                        aucs, ns = [], []
                        for d in days_avail:
                            col = day_families[base][d]
                            s = df[col]
                            mask = s.notna() & y_traj.notna()
                            if mask.sum() < 20 or s[mask].nunique() < 2:
                                aucs.append(None)
                                ns.append(int(mask.sum()))
                                continue
                            a = roc_auc_score(y_traj[mask], s[mask])
                            a = max(a, 1 - a)
                            if a > 0.95:
                                leak_suspects_traj.append(f"{base}_day{d} (AUC={a:.3f})")
                            aucs.append(a)
                            ns.append(int(mask.sum()))
                            per_day_cols[d][base] = col

                        traj_fig.add_trace(go.Scatter(
                            x=days_avail, y=aucs, mode="lines+markers", name=base,
                            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                            text=[f"n={n:,}" for n in ns], hoverinfo="x+y+text+name"))

                    if show_combined:
                        combo_days, combo_aucs = [], []
                        for d in all_days:
                            cols_here = per_day_cols[d]
                            if len(cols_here) < 2:
                                continue
                            sub = df[list(cols_here.values()) + [traj_target]].copy()
                            sub.columns = list(cols_here.keys()) + ["__target__"]
                            sub["__y__"] = (sub["__target__"] == traj_positive).astype(int)
                            sub = sub.drop(columns="__target__").replace(
                                [np.inf, -np.inf], np.nan).dropna()
                            if len(sub) < 20 or sub["__y__"].nunique() < 2:
                                continue
                            X = np.column_stack([
                                np.sign(sub[b]) * np.log1p(np.abs(sub[b])) for b in cols_here.keys()])
                            Xs = StandardScaler().fit_transform(X)
                            clf = LogisticRegression(max_iter=1000).fit(Xs, sub["__y__"].values)
                            a = roc_auc_score(sub["__y__"], clf.decision_function(Xs))
                            combo_days.append(d)
                            combo_aucs.append(max(a, 1 - a))
                        if combo_days:
                            traj_fig.add_trace(go.Scatter(
                                x=combo_days, y=combo_aucs, mode="lines+markers", name="Combined",
                                line=dict(color="black", width=3.5)))

                    traj_fig.add_hline(y=0.5, line_dash="dot", line_color="#c3c2b7")
                    traj_fig.update_layout(
                        xaxis_title="Day (checkpoint)", yaxis_title=f"AUC vs. {traj_target}",
                        title="How does the prediction improve as days accumulate?",
                        yaxis_range=[0.45, 1.02], height=540, **LAYOUT)
                    st.plotly_chart(traj_fig, use_container_width=True)
                    st.caption("Each point: that variable's own value at that checkpoint, used "
                              "directly as a score — no classifier, same method as the ROC tab. "
                              "'Combined' fits a logistic regression across the selected families, "
                              "only at checkpoints where 2+ of them have data.")

                    if leak_suspects_traj:
                        st.warning("AUC > 0.95 for: " + ", ".join(leak_suspects_traj) + " — likely "
                                  "a checkpoint close enough to the end of the window that it's "
                                  "nearly restating the target itself, not predicting it.")

    with subtabs[2]:
        # --------------------------------------------------------------------------- #
        # 13. FLIPS (early prediction vs. actual outcome, and what explains the miss)
        # --------------------------------------------------------------------------- #
        st.caption("Fits a classifier from early ('day 1'-style) variables, compares its "
                  "prediction against the actual outcome, and ranks which *later* behaviors "
                  "best explain the cases where the early prediction turned out wrong.")

        FLIPS_DAY_RE = re.compile(r"^(.+)_day(\d+)$")

        flips_target_opts = [c for c in df.columns if safe_nunique(df[c]) == 2]
        if not flips_target_opts:
            st.caption("No binary column available to use as target.")
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                flips_target = st.selectbox("Target (binary)", flips_target_opts,
                                            index=_default_index(flips_target_opts, "churn"),
                                            key="flips_target")
            with c2:
                day1_suggested = [c for c in numeric_cols
                                  if FLIPS_DAY_RE.match(c) and FLIPS_DAY_RE.match(c).group(2) == "1"
                                  and c != flips_target]
                early_vars = st.multiselect(
                    "Early variables (build the day-1 prediction from these)",
                    [c for c in numeric_cols if c != flips_target],
                    default=day1_suggested or numeric_cols[:3], key="flips_early_vars")

            later_suggested = sorted({
                c for c in numeric_cols
                if (m := FLIPS_DAY_RE.match(c)) and int(m.group(2)) >= 2 and c != flips_target
            })
            later_vars = st.multiselect(
                "Later variables to test as explanations for a wrong early prediction",
                [c for c in numeric_cols if c != flips_target],
                default=later_suggested or [c for c in numeric_cols if c != flips_target][:5],
                key="flips_later_vars")

            if not early_vars:
                st.caption("Pick at least one early variable.")
            else:
                sub_df = df.replace([np.inf, -np.inf], np.nan).dropna(
                    subset=early_vars + [flips_target]).copy()
                flips_values = sorted(sub_df[flips_target].dropna().unique(), key=str)
                flips_positive = flips_values[-1]
                y_flips = (sub_df[flips_target] == flips_positive).astype(int)

                if len(sub_df) < 20 or y_flips.nunique() < 2:
                    st.warning("Not enough valid rows to fit the early classifier.")
                else:
                    X = np.column_stack([np.sign(sub_df[v]) * np.log1p(np.abs(sub_df[v]))
                                         for v in early_vars])
                    Xs = StandardScaler().fit_transform(X)
                    clf = LogisticRegression(max_iter=1000).fit(Xs, y_flips.values)
                    sub_df["_pred"] = (clf.predict_proba(Xs)[:, 1] >= 0.5).astype(int)
                    sub_df["_actual"] = y_flips.values

                    TP = sub_df[(sub_df["_pred"] == 1) & (sub_df["_actual"] == 1)]
                    FP = sub_df[(sub_df["_pred"] == 1) & (sub_df["_actual"] == 0)]
                    FN = sub_df[(sub_df["_pred"] == 0) & (sub_df["_actual"] == 1)]
                    TN = sub_df[(sub_df["_pred"] == 0) & (sub_df["_actual"] == 0)]

                    cm1, cm2 = st.columns([1, 2])
                    with cm1:
                        st.markdown("**Confusion matrix** (early prediction vs. actual)")
                        ct = pd.DataFrame(
                            [[len(TN), len(FN)], [len(FP), len(TP)]],
                            index=["Predicted: 0", "Predicted: 1"],
                            columns=["Actual: 0", "Actual: 1"])
                        st.dataframe(ct, use_container_width=True)
                        acc = (sub_df["_pred"] == sub_df["_actual"]).mean() * 100
                        st.caption(f"Accuracy: {acc:.1f}% · n={len(sub_df):,}")
                    with cm2:
                        qc1, qc2 = st.columns(2)
                        qc1.metric("Recovered (pred. 1, was 0)", f"{len(FP):,}",
                                  help="Looked positive early, actually negative")
                        qc2.metric("Surprise (pred. 0, was 1)", f"{len(FN):,}",
                                  help="Looked negative early, actually positive")
                        qc1.metric("Confirmed positive (pred. 1, was 1)", f"{len(TP):,}")
                        qc2.metric("Confirmed negative (pred. 0, was 0)", f"{len(TN):,}")

                    if not later_vars:
                        st.caption("Pick at least one later variable to explain the flips.")
                    else:
                        def _rank(g_a, g_b):
                            combo = pd.concat([g_a.assign(_g=1), g_b.assign(_g=0)])
                            rows = []
                            for col in later_vars:
                                s = combo[col]
                                mask = s.notna()
                                if mask.sum() < 20 or s[mask].nunique() < 2:
                                    continue
                                a = roc_auc_score(combo.loc[mask, "_g"], s[mask])
                                rows.append((col, max(a, 1 - a)))
                            rows.sort(key=lambda x: x[1])
                            return pd.DataFrame(rows, columns=["variable", "AUC"])

                        r1, r2 = st.columns(2)
                        with r1:
                            st.markdown(f"**What explains recovery?** "
                                       f"({len(FP):,} recovered vs {len(TP):,} confirmed — "
                                       f"both predicted positive)")
                            if len(FP) >= 10 and len(TP) >= 10:
                                rank_fp = _rank(FP, TP)
                                fig1 = px.bar(rank_fp, x="AUC", y="variable", orientation="h",
                                             color_discrete_sequence=PALETTE)
                                fig1.update_layout(height=max(220, 28 * len(rank_fp)),
                                                  yaxis_title="", **LAYOUT)
                                st.plotly_chart(fig1, use_container_width=True)
                            else:
                                st.caption("Not enough rows in one of the two groups.")
                        with r2:
                            st.markdown(f"**What explains a surprise?** "
                                       f"({len(FN):,} surprise vs {len(TN):,} confirmed — "
                                       f"both predicted negative)")
                            if len(FN) >= 10 and len(TN) >= 10:
                                rank_fn = _rank(FN, TN)
                                fig2 = px.bar(rank_fn, x="AUC", y="variable", orientation="h",
                                             color_discrete_sequence=PALETTE)
                                fig2.update_layout(height=max(220, 28 * len(rank_fn)),
                                                  yaxis_title="", **LAYOUT)
                                st.plotly_chart(fig2, use_container_width=True)
                            else:
                                st.caption("Not enough rows in one of the two groups.")

                        st.caption("AUC here measures how well each *later* variable tells the two "
                                  "outcomes apart within a group that all got the *same* early "
                                  "prediction — closer to 1.0 means that behavior is what actually "
                                  "decided the outcome, not the early signal.")

# --------------------------------------------------------------------------- #
# Footer (outside the tabs, always visible at the bottom of the page)
# --------------------------------------------------------------------------- #
st.markdown("<hr style='margin-top: 2rem; margin-bottom: 0.5rem; opacity: 0.2;'>",
           unsafe_allow_html=True)
