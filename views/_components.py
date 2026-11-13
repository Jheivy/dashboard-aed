"""Componentes visuales del sistema de diseño (cabeceras, KPIs, alertas, tarjetas, chips)."""

import html

import streamlit as st

from eda.config import SEVERITY


def page_header(eyebrow: str, title: str, subtitle: str | None = None) -> None:
    sub = f"<div class='eda-sub'>{html.escape(subtitle)}</div>" if subtitle else ""
    st.markdown(f"<div class='eda-head'><div class='eda-eyebrow'>{html.escape(eyebrow)}</div>"
                f"<div class='eda-title'>{html.escape(title)}</div>{sub}</div>", unsafe_allow_html=True)


def chips(items: list[tuple[str, str]], accent_first: bool = True) -> None:
    """Row of dataset chips: [(label, value)]; value may be empty for plain labels."""
    out = []
    for i, (label, value) in enumerate(items):
        cls = "eda-chip accent" if (accent_first and i == 0) else "eda-chip"
        body = (f"{html.escape(label)} <b>{html.escape(value)}</b>" if value
                else f"<b>{html.escape(label)}</b>")
        out.append(f"<span class='{cls}'>{body}</span>")
    st.markdown(f"<div class='eda-chips'>{''.join(out)}</div>", unsafe_allow_html=True)


def kpi_row(items: list[dict]) -> None:
    """items: dict(label, value, help=None, delta=None, delta_color='normal', chart=None)."""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.metric(it["label"], it["value"], it.get("delta"), help=it.get("help"), border=True,
                   delta_color=it.get("delta_color", "normal"), chart_data=it.get("chart"),
                   chart_type=it.get("chart_type", "bar"))


def severity_pill(sev: str, count: int | None = None) -> str:
    color, icon, label = SEVERITY.get(sev, ("#888", "●", sev))
    txt = f"{label}" + (f" · {count}" if count is not None else "")
    return f"<span class='eda-pill' style='--sev:{color}'><i>{icon}</i>{html.escape(txt)}</span>"


def alert_card(sev: str, problem: str, column: str, detail: str) -> str:
    color, icon, label = SEVERITY.get(sev, ("#888", "●", sev))
    return (f"<div class='eda-alert' style='--sev:{color}'>"
            f"<span class='eda-sev'><i>{icon}</i>{label}</span>"
            f"<span class='eda-alert-title'>{html.escape(problem)} · <code>{html.escape(str(column))}</code></span>"
            f"<span class='eda-alert-detail'>{html.escape(str(detail))}</span></div>")


def alert_list(rows) -> None:
    st.markdown("".join(alert_card(r.gravedad, r.problema, r.columna, r.detalle) for r in rows),
                unsafe_allow_html=True)


def card(title: str | None = None, subtitle: str | None = None):
    """Bordered section container with an optional title. Use as a context manager."""
    c = st.container(border=True)
    if title:
        sub = f"<div class='eda-card-sub'>{html.escape(subtitle)}</div>" if subtitle else ""
        c.markdown(f"<div class='eda-card-title'>{html.escape(title)}</div>{sub}", unsafe_allow_html=True)
    return c


def feature_grid(features: list[tuple[str, str, str]]) -> None:
    """[(icon glyph, title, text)] -> responsive grid of feature cards."""
    items = "".join(f"<div class='eda-feature'><div class='ic'>{ic}</div><h4>{html.escape(t)}</h4>"
                    f"<div class='txt'>{html.escape(p)}</div></div>" for ic, t, p in features)
    st.markdown(f"<div class='eda-grid'>{items}</div>", unsafe_allow_html=True)
