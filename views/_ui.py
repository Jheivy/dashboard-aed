"""Helpers de interfaz compartidos: contexto, selección, gráficos, descargas, formato."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from eda.config import PLOTLY_CONFIG
from eda.stats_utils import fmt_p

NONE = "(ninguna)"


@dataclass
class Ctx:
    df: pd.DataFrame
    key: str                      # fingerprint of the exact data state (cache key)
    types: dict
    numeric: list
    categorical: list
    datetime: list
    text: list
    ids: list
    source_col: str | None = None
    prep_log: list = field(default_factory=list)
    load_info: list = field(default_factory=list)
    n_loaded: int = 0
    n_filtered: int = 0
    sampled: bool = False


def sub_nav(options: list[str], key: str) -> str:
    """Sub-section selector. Unlike st.tabs, only the chosen sub-section is computed."""
    choice = st.segmented_control("Apartado", options, default=options[0], key=key,
                                  label_visibility="collapsed")
    return choice or options[0]


def chart(fig, key: str) -> None:
    st.plotly_chart(fig, key=key, config=PLOTLY_CONFIG)


def need(cond: bool, msg: str) -> bool:
    if not cond:
        st.info(msg, icon=":material/info:")
    return cond


def top_with_other(s: pd.Series, n: int, other: str = "Otros") -> pd.Series:
    s = s.astype(object).where(s.notna(), "(nulo)").astype(str)
    top = s.value_counts().head(n).index
    return s.where(s.isin(top), other)


def verdict(p: float, yes: str, no: str, alpha: float = 0.05, extra: str = "") -> None:
    if pd.isna(p):
        return
    msg = f"**{yes if p < alpha else no}** · p = {fmt_p(p)}" + (f" · {extra}" if extra else "")
    if p < alpha:
        st.success(msg, icon=":material/check_circle:")
    else:
        st.info(msg, icon=":material/remove_circle_outline:")


def insight(text: str, kind: str = "info") -> None:
    """Interpretation callout."""
    icons = {"info": ":material/lightbulb:", "warn": ":material/warning:", "ok": ":material/check_circle:"}
    fn = {"info": st.info, "warn": st.warning, "ok": st.success}[kind]
    fn(text, icon=icons[kind])


def download_csv(df_or_fn, filename: str, label: str = "Descargar CSV", key: str | None = None) -> None:
    """Deferred download: the CSV is built only when the button is clicked."""
    def build() -> bytes:
        d = df_or_fn() if callable(df_or_fn) else df_or_fn
        return d.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, build, file_name=filename, mime="text/csv", key=key,
                       icon=":material/download:", on_click="ignore")


def sample_rows(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    return df.sample(n, random_state=seed) if len(df) > n else df


def num_format_config(df: pd.DataFrame, dec: int = 3) -> dict:
    return {c: st.column_config.NumberColumn(format=f"%.{dec}f") for c in df.columns
            if pd.api.types.is_float_dtype(df[c])}


def pick_index(options: list, keywords: tuple[str, ...] = (), fallback: int = 0) -> int:
    for kw in keywords:
        for i, o in enumerate(options):
            if kw in str(o).lower():
                return i
    return min(fallback, max(len(options) - 1, 0))


def finite(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def es_int(v) -> str:
    return f"{int(v):,}".replace(",", ".")


def es_pct(v: float, dec: int = 1) -> str:
    return f"{v:.{dec}f}%".replace(".", ",")


def es_num(v: float, sig: int = 4) -> str:
    if v is None or pd.isna(v):
        return "—"
    s = f"{v:,.{sig}g}" if abs(v) < 1e6 else f"{v:,.0f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")
