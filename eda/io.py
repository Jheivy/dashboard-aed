"""Lectura de archivos (CSV/TSV/TXT, Excel, Parquet, JSON) y dataset de ejemplo."""

import csv
import io
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

TEXT_EXT = (".csv", ".tsv", ".txt")
EXCEL_EXT = (".xlsx", ".xlsm")
PARQUET_EXT = (".parquet", ".pq")
JSON_EXT = (".json", ".jsonl")
UPLOAD_TYPES = ["csv", "tsv", "txt", "xlsx", "xlsm", "parquet", "pq", "json", "jsonl"]
SEPARATORS = {"Automático": "auto", "Coma (,)": ",", "Punto y coma (;)": ";",
              "Tabulador": "\t", "Barra vertical (|)": "|"}
ENCODINGS = ["Automática", "utf-8", "latin-1", "cp1252", "utf-16"]


def _detect_encoding(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    sample = raw[:2_000_000]
    try:
        sample.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError as e:
        # A truncated multi-byte character at the very end of the sample is fine.
        if e.start >= len(sample) - 4 and len(raw) > len(sample):
            return "utf-8"
        return "latin-1"


def _detect_sep(text: str) -> str:
    """Picks the delimiter whose per-line count is most consistent (and > 0)."""
    lines = [ln for ln in text.splitlines()[:50] if ln.strip()]
    if not lines:
        return ","
    best, best_score = ",", -1.0
    for d in [",", ";", "\t", "|"]:
        counts = [ln.count(d) for ln in lines]
        if max(counts) == 0:
            continue
        mode, freq = Counter(counts).most_common(1)[0]
        score = (freq / len(lines)) * (1 if mode > 0 else 0) + min(mode, 50) / 1000
        if score > best_score:
            best, best_score = d, score
    if best_score < 0:
        try:
            return csv.Sniffer().sniff(text[:20000], delimiters=",;\t|").delimiter
        except csv.Error:
            return ","
    return best


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Column names as unique, non-empty strings."""
    names, seen = [], Counter()
    for i, c in enumerate(df.columns):
        name = str(c).strip() or f"columna_{i + 1}"
        seen[name] += 1
        names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    out = df.copy()
    out.columns = names
    return out


@st.cache_data(show_spinner=False, max_entries=16)
def excel_sheets(_raw: bytes, key: str) -> list[str]:
    return pd.ExcelFile(io.BytesIO(_raw), engine="openpyxl").sheet_names


@st.cache_resource(show_spinner="Leyendo archivo…", max_entries=8)
def read_file(name: str, _raw: bytes, key: str, sep: str = "auto", encoding: str = "Automática",
              decimal: str = ".", sheet: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns (DataFrame, info). `key` identifies the bytes (they are not hashed on every rerun)."""
    raw = _raw
    lname = name.lower()
    info: dict = {"archivo": name}
    if lname.endswith(PARQUET_EXT):
        df = pd.read_parquet(io.BytesIO(raw))
        info["formato"] = "Parquet"
    elif lname.endswith(EXCEL_EXT):
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet or 0)
        info["formato"] = "Excel"
        info["hoja"] = sheet or excel_sheets(raw, key)[0]
    elif lname.endswith(JSON_EXT):
        try:
            df = pd.read_json(io.BytesIO(raw), lines=lname.endswith(".jsonl"))
        except ValueError:
            df = pd.read_json(io.BytesIO(raw), lines=True)
        info["formato"] = "JSON"
    else:
        enc = _detect_encoding(raw) if encoding == "Automática" else encoding
        text_head = raw[:200_000].decode(enc, errors="ignore")
        if sep == "auto":
            sep = "\t" if lname.endswith(".tsv") else _detect_sep(text_head)
        # Codes with leading zeros (postal codes, IDs…) would lose them if parsed as numbers:
        # peek at the first rows as text and keep those columns as strings.
        try:
            head = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, dtype=str, nrows=2000)
            keep_str = {c: str for c in head.columns
                        if head[c].dropna().str.strip().str.match(r"^0\d").mean() > 0.05}
        except Exception:  # noqa: BLE001
            keep_str = {}
        df = None
        engine = "c"
        if decimal == "." and enc.replace("-sig", "") in ("utf-8", "latin-1", "cp1252"):
            try:  # multi-threaded Arrow reader: several times faster on large files
                df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, dtype=keep_str or None,
                                 engine="pyarrow")
                engine = "pyarrow"
            except Exception:  # noqa: BLE001 — ragged rows, odd quoting… fall back to the C parser
                df = None
        if df is None:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, decimal=decimal, dtype=keep_str or None,
                             low_memory=False, on_bad_lines="warn")
        info.update(formato="Texto delimitado", codificación=enc, motor=engine,
                    separador={"\t": "tabulador"}.get(sep, sep), decimal=decimal)
    df = clean_column_names(df)
    info["filas"], info["columnas"] = df.shape
    return df, info


# --------------------------------------------------------------------------- #
# Dataset de ejemplo: sintético y con problemas de calidad a propósito
# (nulos MCAR/MAR, outliers, centinelas, categorías sucias, duplicados…)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def demo_dataset(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    region = rng.choice(["Norte", "Sur", "Este", "Oeste", "Centro"], n, p=[.25, .2, .2, .15, .2])
    plan = rng.choice(["Básico", "Estándar", "Premium"], n, p=[.5, .35, .15])
    canal = rng.choice(["Web", "Tienda", "Teléfono", "App"], n, p=[.4, .25, .1, .25]).astype(object)
    edad = rng.normal(42, 12, n).clip(18, 90).round()
    ingresos = (rng.lognormal(10.2, 0.5, n) * np.where(plan == "Premium", 1.6, 1.0)).round(0)
    antiguedad = rng.exponential(30, n).round().clip(1, 120)
    visitas = rng.poisson(6, n) + (plan == "Premium") * 3
    gasto = (20 + 0.004 * ingresos + 4 * visitas + rng.normal(0, 25, n)).round(2)
    satisfaccion = (rng.normal(7, 1.5, n) - 0.6 * (plan == "Básico")).clip(0, 10).round(1)
    logit = (-1.0 - 0.35 * (satisfaccion - 7) - 0.02 * (antiguedad - 30)
             + 0.6 * (plan == "Básico") - 0.08 * (visitas - 6))
    abandono = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    fecha_alta = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, n), unit="D")
    valoracion = np.round(rng.normal(3.8, 0.7, n).clip(1, 5), 1)

    df = pd.DataFrame({
        "id_cliente": [f"C{100000 + i}" for i in range(n)],
        "fecha_alta": fecha_alta,
        "region": region.astype(object),
        "plan": plan,
        "canal": canal,
        "edad": edad,
        "ingresos_anuales": ingresos,
        "antiguedad_meses": antiguedad,
        "visitas_mes": visitas,
        "gasto_mensual": gasto,
        "satisfaccion": satisfaccion,
        "valoracion_web": [f"{v:.1f}".replace(".", ",") for v in valoracion],
        "pais": "España",
        "abandono": abandono,
    })
    # Nulos MCAR (al azar) en edad
    df.loc[rng.random(n) < 0.04, "edad"] = np.nan
    # Nulos MAR: los ingresos faltan mucho más en el canal Teléfono
    p_miss = np.where(df["canal"] == "Teléfono", 0.45, 0.04)
    df.loc[rng.random(n) < p_miss, "ingresos_anuales"] = np.nan
    # Nulos MNAR-like: la satisfacción falta más entre quienes abandonan
    p_miss = np.where(df["abandono"] == 1, 0.25, 0.05)
    df.loc[rng.random(n) < p_miss, "satisfaccion"] = np.nan
    # Outliers en gasto y un código centinela -999 en edad
    idx = rng.choice(n, 12, replace=False)
    df.loc[idx, "gasto_mensual"] = df.loc[idx, "gasto_mensual"] * rng.uniform(6, 12, 12)
    df.loc[rng.choice(n, 15, replace=False), "edad"] = -999
    # Categorías sucias y nulos disfrazados
    dirty = rng.choice(n, 40, replace=False)
    df.loc[dirty[:20], "region"] = df.loc[dirty[:20], "region"].str.lower()
    df.loc[dirty[20:], "region"] = df.loc[dirty[20:], "region"] + " "
    df.loc[rng.choice(n, 30, replace=False), "canal"] = "N/A"
    # Filas duplicadas
    dup = df.sample(15, random_state=seed)
    df = pd.concat([df, dup], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
