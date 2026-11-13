"""Preparación de datos y detección de tipos semánticos de columna.

Rendimiento: todas las operaciones de texto se hacen sobre los valores *únicos* de cada
columna (pd.factorize) y el resultado se proyecta a las filas con un `take` vectorizado.
Una columna de 1M de filas con 30 categorías procesa 30 cadenas, no 1M.
"""

import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

TYPE_OPTIONS = ["numérica", "categórica", "booleana", "fecha", "texto",
                "identificador", "constante", "ignorar"]

NULL_TOKENS = ["", "na", "n/a", "nan", "null", "none", "-", "--", "?", "nd", "n.d.",
               "s/d", "sin dato", "sin datos", "#n/a", "missing", "desconocido", "unknown"]

BOOL_TOKENS = {"true", "false", "sí", "si", "no", "yes", "y", "n", "s",
               "verdadero", "falso", "t", "f"}

_RE_STD = r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$"
_RE_EU = r"^[+-]?(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?$"
_RE_US = r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$"
_RE_DATE = (r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
            r"(?:[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?$")
_RE_STRIP = r"[%€$£\s ]"
ID_NAME = re.compile(r"(^id$|^id[_\s-]|[_\s-]id$|^cod(igo)?[_\s-]|^key$|uuid|^dni|^nif)", re.I)
CATEGORY_MAX_RATIO = 0.5


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def has_unhashable(s: pd.Series) -> bool:
    if s.dtype != object:
        return False
    v = s.dropna()
    return (not v.empty) and isinstance(v.iloc[0], (list, np.ndarray, dict, set))


def safe_nunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return int(s.astype(str).nunique())


def is_textlike(s: pd.Series) -> bool:
    """Object, string or categorical column holding scalar values."""
    return (s.dtype == object or isinstance(s.dtype, (pd.StringDtype, pd.CategoricalDtype))) \
        and not has_unhashable(s)


def normalize_text(v: str) -> str:
    """Lowercase, no accents, collapsed whitespace — to spot inconsistent spellings."""
    v = unicodedata.normalize("NFKD", str(v))
    v = "".join(ch for ch in v if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", v).strip().lower()


def uniques_with_counts(s: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """(unique non-null values as strings, their row counts) — the base of every text check."""
    vc = s.value_counts(dropna=True)
    vc = vc[vc > 0]
    return pd.Series(vc.index.astype(str), dtype=object), vc.to_numpy()


def map_uniques(s: pd.Series, fn) -> pd.Series:
    """Applies `fn` (vectorized, on a Series of unique values) and projects to every row."""
    codes, uniq = pd.factorize(s, use_na_sentinel=True)
    vals = fn(pd.Series(uniq, dtype=object)).reset_index(drop=True)
    tail = vals.iloc[:0].reindex([len(vals)])          # one NA of the right dtype
    full = pd.concat([vals, tail]) if len(vals) else tail
    out = full.iloc[codes].to_numpy()                   # code -1 (null) -> last element (NA)
    return pd.Series(out, index=s.index, name=s.name)


def _weighted_share(mask: pd.Series | np.ndarray, counts: np.ndarray) -> float:
    total = counts.sum()
    return float(np.asarray(mask, bool) @ counts / total) if total else 0.0


# --------------------------------------------------------------------------- #
# Conversión de texto a número / fecha
# --------------------------------------------------------------------------- #
def _quick_reject(u: pd.Series, patterns: list[str], min_share: float) -> bool:
    """High-cardinality columns: test a sample of uniques first and bail out early."""
    if len(u) <= 5000:
        return False
    smp = u.sample(2000, random_state=0).str.replace(_RE_STRIP, "", regex=True)
    return max(smp.str.match(p).mean() for p in patterns) < min_share - 0.1


def _numeric_format(u: pd.Series, counts: np.ndarray, min_share: float):
    if u.empty or _quick_reject(u, [_RE_STD, _RE_EU, _RE_US], min_share):
        return None
    if _weighted_share(u.str.match(r"^0\d"), counts) > 0.05:   # codes with leading zeros stay text
        return None
    c = u.str.replace(_RE_STRIP, "", regex=True)
    formats = [("estándar", _RE_STD), ("europeo (1.234,5)", _RE_EU), ("anglosajón (1,234.5)", _RE_US)]
    if c.str.contains(",", regex=False).any():
        formats = [formats[1], formats[2], formats[0]]
    for fmt, rx in formats:
        if _weighted_share(c.str.match(rx), counts) >= min_share:
            return fmt
    return None


def _parse_numbers(u: pd.Series, fmt: str) -> pd.Series:
    c = u.astype(str).str.strip().str.replace(_RE_STRIP, "", regex=True)
    if fmt.startswith("europeo"):
        c = c.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    elif fmt.startswith("anglosajón"):
        c = c.str.replace(",", "", regex=False)
    return pd.to_numeric(c, errors="coerce").astype(float)


def numeric_from_strings(s: pd.Series, min_share: float = 0.95, uc=None):
    """Reads a text column as numbers (1.234,5 / 1,234.5 / 12% / 3 €).
    Returns (converted, format_name, values_lost) or (None, None, 0)."""
    u, counts = uc if uc is not None else uniques_with_counts(s)
    u = u.str.strip()
    keep = (u != "").to_numpy()
    fmt = _numeric_format(u[keep].reset_index(drop=True), counts[keep], min_share)
    if fmt is None:
        return None, None, 0
    out = map_uniques(s, lambda v: _parse_numbers(v, fmt)).astype(float)
    blank = int(counts[~keep].sum())
    lost = int(s.notna().sum() - out.notna().sum() - blank)
    if out.notna().all() and np.all(np.mod(out.to_numpy(), 1) == 0):
        out = out.astype("int64")
    return out, fmt, max(lost, 0)


def dates_from_strings(s: pd.Series, min_share: float = 0.95, uc=None):
    u, counts = uc if uc is not None else uniques_with_counts(s)
    u = u.str.strip()
    if u.empty or _quick_reject(u, [_RE_DATE], min_share) or (
            _weighted_share(u.str.match(_RE_DATE), counts) < min_share):
        return None, None
    parts = u.str.extract(r"^(\d{1,4})[-/.](\d{1,2})[-/.](\d{1,4})").astype(float)
    if (parts[0] > 31).any():
        dayfirst, fmt = False, "año-mes-día"
    elif (parts[0] > 12).any():
        dayfirst, fmt = True, "día/mes/año"
    elif (parts[1] > 12).any():
        dayfirst, fmt = False, "mes/día/año"
    else:
        dayfirst, fmt = True, "día/mes/año (ambiguo)"
    out = map_uniques(s, lambda v: pd.to_datetime(v.astype(str).str.strip(), errors="coerce",
                                                  dayfirst=dayfirst, format="mixed"))
    out = pd.to_datetime(out, errors="coerce")
    if out.notna().sum() < 0.9 * s.notna().sum():
        return None, None
    return out, fmt


def to_label(s: pd.Series) -> pd.Series:
    """Any column -> text labels (keeps nulls), with 3.0 shown as '3'."""
    def fmt(v: pd.Series) -> pd.Series:
        return v.map(lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x))
    return map_uniques(s, fmt).astype(object)


# --------------------------------------------------------------------------- #
# Tipos semánticos
# --------------------------------------------------------------------------- #
def semantic_type(s: pd.Series, name: str = "", uc=None) -> str:
    if has_unhashable(s):
        return "ignorar"
    n = int(s.notna().sum())
    nu = safe_nunique(s)
    if n == 0 or nu <= 1:
        return "constante"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "fecha"
    if pd.api.types.is_bool_dtype(s) or nu == 2:
        return "booleana"
    if pd.api.types.is_timedelta64_dtype(s):
        return "ignorar"
    if pd.api.types.is_numeric_dtype(s):
        if nu == n and n >= 20:
            nn = s.dropna().to_numpy(dtype=float)
            if np.all(np.mod(nn, 1) == 0):
                row_counter = nn[0] in (0, 1) and bool(np.all(np.diff(nn) == 1))
                if ID_NAME.search(name) or row_counter:
                    return "identificador"
        return "numérica"
    if nu == n and n >= 20:
        return "identificador"
    u, counts = uc if uc is not None else uniques_with_counts(s)
    avg_len = float((u.str.len().to_numpy() * counts).sum() / counts.sum())
    if avg_len > 40 and nu / n > 0.5:
        return "texto"
    if nu / n > 0.9 and n > 100:
        return "identificador" if ID_NAME.search(name) or u.str.len().std() < 2 else "texto"
    return "categórica"


@st.cache_resource(show_spinner="Preparando datos y detectando tipos…", max_entries=4)
def prepare(_df: pd.DataFrame, key: str, null_tokens: tuple[str, ...], strip_text: bool,
            drop_empty: bool, drop_duplicates: bool, auto_convert: bool):
    """Applies the preparation options. `key` identifies `_df` (not hashed for speed).
    Returns (df, types, log) where `log` lists every automatic change."""
    df = _df.copy()
    log: list[dict] = []

    if drop_empty:
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            df = df.drop(columns=empty_cols)
            log.append(dict(columna=", ".join(empty_cols[:10]) + ("…" if len(empty_cols) > 10 else ""),
                            cambio="Columnas totalmente vacías eliminadas",
                            detalle=f"{len(empty_cols)} columnas"))
        n0 = len(df)
        df = df.dropna(how="all")
        if len(df) < n0:
            log.append(dict(columna="(filas)", cambio="Filas totalmente vacías eliminadas",
                            detalle=f"{n0 - len(df):,} filas"))

    text_cols = [c for c in df.columns if is_textlike(df[c]) and df[c].dtype == object]
    tokens = {t.strip().lower() for t in null_tokens}
    for c in text_cols:
        s = df[c]
        changed = {}
        if strip_text or tokens:
            u, counts = uniques_with_counts(s)
            stripped = u.str.strip()
            if strip_text:
                ws = (stripped != u).to_numpy()
                if ws.any():
                    changed["strip"] = int(counts[ws].sum())
            if tokens:
                hit = stripped.str.lower().isin(tokens).to_numpy()
                if hit.any():
                    changed["null"] = int(counts[hit].sum())
            if changed:
                def fix(v: pd.Series) -> pd.Series:
                    sv = v.astype(str)
                    out = sv.str.strip() if strip_text else sv
                    if tokens:
                        out = out.where(~sv.str.strip().str.lower().isin(tokens), np.nan)
                    return out.astype(object)
                df[c] = map_uniques(s, fix).astype(object)
        if "strip" in changed:
            log.append(dict(columna=c, cambio="Espacios al inicio/final eliminados",
                            detalle=f"{changed['strip']:,} valores"))
        if "null" in changed:
            log.append(dict(columna=c, cambio="Marcadores de nulo → vacío",
                            detalle=f"{changed['null']:,} valores"))
    num_tokens = []
    for t in tokens:
        try:
            num_tokens.append(float(t))
        except ValueError:
            pass
    if num_tokens:
        for c in df.select_dtypes(include=np.number).columns:
            hit = df[c].isin(num_tokens)
            if hit.any():
                df.loc[hit, c] = np.nan
                log.append(dict(columna=c, cambio="Código numérico de nulo → vacío",
                                detalle=f"{int(hit.sum()):,} valores"))

    if drop_duplicates:
        n0 = len(df)
        try:
            df = df.drop_duplicates()
        except TypeError:
            df = df.loc[~df.astype(str).duplicated()]
        if len(df) < n0:
            log.append(dict(columna="(filas)", cambio="Filas duplicadas eliminadas",
                            detalle=f"{n0 - len(df):,} filas"))

    if auto_convert:
        for c in [c for c in df.columns if is_textlike(df[c]) and df[c].dtype == object]:
            uc = uniques_with_counts(df[c])
            num, fmt, lost = numeric_from_strings(df[c], uc=uc)
            if num is not None:
                df[c] = num
                det = f"formato {fmt}" + (f" · {lost:,} valores no convertibles → nulo" if lost else "")
                log.append(dict(columna=c, cambio="Texto → número", detalle=det, perdidos=lost))
                continue
            dt, fmt = dates_from_strings(df[c], uc=uc)
            if dt is not None:
                df[c] = dt
                log.append(dict(columna=c, cambio="Texto → fecha", detalle=f"formato {fmt}"))
        for c in df.columns:
            if pd.api.types.is_bool_dtype(df[c]):
                df[c] = df[c].astype(int)

    df = df.reset_index(drop=True)
    types = {c: semantic_type(df[c], c) for c in df.columns}
    # Low-cardinality text -> category: faster groupby/value_counts/duplicated, less memory.
    for c, t in types.items():
        s = df[c]
        if t in ("categórica", "booleana") and s.dtype == object and not has_unhashable(s):
            n = s.notna().sum()
            if n and safe_nunique(s) / n <= CATEGORY_MAX_RATIO:
                df[c] = s.astype("category")
    return df, types, log


def apply_type_overrides(df: pd.DataFrame, types: dict, overrides: dict) -> tuple[pd.DataFrame, dict]:
    """Converts the columns whose type the user changed by hand."""
    df = df.copy()
    types = dict(types)
    for c, t in overrides.items():
        if c not in df.columns or t == types.get(c):
            continue
        s = df[c]
        if t == "numérica" and not pd.api.types.is_numeric_dtype(s):
            num, _, _ = numeric_from_strings(s.astype(object), min_share=0.0)
            df[c] = num if num is not None else pd.to_numeric(s.astype(object), errors="coerce")
        elif t == "fecha" and not pd.api.types.is_datetime64_any_dtype(s):
            if pd.api.types.is_numeric_dtype(s):  # e.g. a year column
                df[c] = pd.to_datetime(s.astype("Int64").astype(str), format="%Y", errors="coerce")
            else:
                df[c] = pd.to_datetime(s.astype(object), errors="coerce", dayfirst=True, format="mixed")
        elif t in ("categórica", "texto", "identificador") and not is_textlike(s):
            lab = to_label(s)
            df[c] = lab.astype("category") if t == "categórica" else lab
        types[c] = t
    return df, types


def split_columns(df: pd.DataFrame, types: dict) -> dict[str, list[str]]:
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
           and types.get(c) in ("numérica", "booleana")]
    cat = [c for c in df.columns if types.get(c) in ("categórica", "booleana")]
    dt = [c for c in df.columns if types.get(c) == "fecha"
          and pd.api.types.is_datetime64_any_dtype(df[c])]
    return dict(numeric=num, categorical=cat, datetime=dt,
                text=[c for c in df.columns if types.get(c) == "texto"],
                id=[c for c in df.columns if types.get(c) == "identificador"],
                other=[c for c in df.columns if types.get(c) in ("constante", "ignorar")])
