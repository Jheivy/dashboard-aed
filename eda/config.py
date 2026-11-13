"""Sistema de diseño compartido: tokens por tema, paletas validadas, plantilla de Plotly y CSS.

Paleta categórica (8 slots, orden fijo) validada con el validador de la skill dataviz en
modo claro y oscuro. Los colores de estado (gravedad) están reservados y nunca se usan
como color de serie; siempre van acompañados de icono y texto.
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SEQ_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#3987e5",
            "#2a78d6", "#256abf", "#184f95", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEVERITY = {  # gravedad -> (color de estado, icono, etiqueta)
    "Alta": (STATUS["critical"], "▲", "Alta"),
    "Media": (STATUS["serious"], "◆", "Media"),
    "Baja": (STATUS["warning"], "●", "Baja"),
}
SEVERITY_COLORS = {k: v[0] for k, v in SEVERITY.items()}
SCALES = ["lineal", "log10", "log1p", "raíz"]

TOKENS = {
    "light": dict(bg="#f7f8fa", card="#ffffff", border="#e2e6ec", text="#111827", muted="#5b6474",
                  subtle="#8a93a3", accent="#2a78d6", accent_soft="#eaf2fc", mid="#e9ebee",
                  grid="rgba(17,24,39,0.07)", axis="rgba(17,24,39,0.25)", surface="#ffffff"),
    "dark": dict(bg="#0f1115", card="#161a21", border="#272c36", text="#e6e8ec", muted="#a3abb9",
                 subtle="#7a8394", accent="#3987e5", accent_soft="rgba(57,135,229,0.14)", mid="#383835",
                 grid="rgba(230,232,236,0.08)", axis="rgba(230,232,236,0.28)", surface="#161a21"),
}

PLOTLY_CONFIG = {"displayModeBar": "hover", "displaylogo": False, "responsive": True,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
                 "toImageButtonOptions": {"format": "png", "scale": 2}}


def theme_type() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # noqa: BLE001 — outside a session (tests, bare mode)
        return "light"


def tok() -> dict:
    return TOKENS[theme_type()]


def pal() -> list[str]:
    """Categorical palette for the active theme (fixed order, never cycled past 8)."""
    return PAL_DARK if theme_type() == "dark" else PAL_LIGHT


def seq_scale() -> list[str]:
    # light: near-zero recedes to the white surface; dark: near-zero recedes to the dark surface
    return SEQ_RAMP if theme_type() == "light" else SEQ_RAMP[::-1]


def div_scale() -> list[list]:
    """Diverging blue <-> red with a neutral gray midpoint (never a hue at the center)."""
    mid = tok()["mid"]
    return [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, mid], [0.75, "#ec8b8a"], [1.0, "#b42f2e"]]


def outlier_color() -> str:
    return STATUS["critical"]


def normal_color() -> str:
    return pal()[0]


CSS = """
<style>
:root {{
  --eda-bg:{bg}; --eda-card:{card}; --eda-border:{border}; --eda-text:{text};
  --eda-muted:{muted}; --eda-subtle:{subtle}; --eda-accent:{accent}; --eda-accent-soft:{accent_soft};
}}
.block-container {{padding-top: 4.6rem; padding-bottom: 4rem; max-width: 1440px;}}
[data-testid="stMetric"] {{background: var(--eda-card);}}
[data-testid="stMetricValue"] {{font-size: 1.45rem; font-weight: 650; letter-spacing: -0.01em;
  font-variant-numeric: tabular-nums;}}
[data-testid="stMetricLabel"] p {{font-size: 0.78rem; color: var(--eda-muted); font-weight: 500;}}
[data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {{gap: 0.7rem;}}
div[data-testid="stExpander"] details {{background: var(--eda-card);}}
[data-testid="stSidebar"] [data-testid="stExpander"] details {{background: transparent;}}

/* Cabecera de página */
.eda-head {{margin: 0 0 1.1rem 0;}}
.eda-eyebrow {{font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--eda-accent); margin-bottom: 0.25rem;}}
.eda-title {{font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2;
  color: var(--eda-text); margin: 0;}}
.eda-sub {{color: var(--eda-muted); font-size: 0.95rem; margin: 0.35rem 0 0 0; max-width: 820px;}}

/* Chips del dataset */
.eda-chips {{display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0 0 1.3rem 0;}}
.eda-chip {{display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.22rem 0.65rem;
  border: 1px solid var(--eda-border); border-radius: 999px; background: var(--eda-card);
  font-size: 0.78rem; color: var(--eda-muted); white-space: nowrap;}}
.eda-chip b {{color: var(--eda-text); font-weight: 600; font-variant-numeric: tabular-nums;}}
.eda-chip.accent {{background: var(--eda-accent-soft); border-color: transparent; color: var(--eda-accent);}}
.eda-chip.accent b {{color: var(--eda-accent);}}

/* Tarjetas de alerta */
.eda-alert {{display: grid; grid-template-columns: auto 1fr; gap: 0.15rem 0.7rem; align-items: start;
  padding: 0.65rem 0.85rem; border: 1px solid var(--eda-border); border-left: 3px solid var(--sev);
  border-radius: 0.5rem; background: var(--eda-card); margin-bottom: 0.45rem;}}
.eda-sev {{display: inline-flex; align-items: center; gap: 0.3rem; font-size: 0.7rem; font-weight: 650;
  text-transform: uppercase; letter-spacing: 0.04em; color: var(--eda-text); padding-top: 0.1rem;}}
.eda-sev i {{font-style: normal; color: var(--sev); font-size: 0.8rem;}}
.eda-alert-title {{font-weight: 600; font-size: 0.9rem; color: var(--eda-text);}}
.eda-alert-title code {{font-size: 0.8rem;}}
.eda-alert-detail {{grid-column: 2; font-size: 0.82rem; color: var(--eda-muted);}}
.eda-pill {{display: inline-flex; align-items: center; gap: 0.3rem; font-size: 0.75rem; font-weight: 600;
  padding: 0.1rem 0.55rem; border-radius: 999px; border: 1px solid var(--eda-border); margin-right: 0.35rem;
  color: var(--eda-text); background: var(--eda-card);}}
.eda-pill i {{font-style: normal; color: var(--sev);}}

/* Tarjeta de sección */
.eda-card-title {{font-weight: 600; font-size: 1rem; color: var(--eda-text); margin: 0;}}
.eda-card-sub {{font-size: 0.8rem; color: var(--eda-muted); margin: 0.15rem 0 0.4rem 0; line-height: 1.4;}}

/* Bienvenida */
.eda-hero {{padding: 2.6rem 2.4rem; border: 1px solid var(--eda-border); border-radius: 1rem;
  background: radial-gradient(1200px 300px at 0% 0%, var(--eda-accent-soft), transparent 60%), var(--eda-card);
  margin-bottom: 1.4rem;}}
.eda-hero h1 {{font-size: 2.3rem; font-weight: 750; letter-spacing: -0.03em; line-height: 1.1;
  margin: 0.3rem 0 0.7rem 0; color: var(--eda-text);}}
.eda-hero p {{font-size: 1.05rem; color: var(--eda-muted); max-width: 680px; margin: 0;}}
.eda-grid {{display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 0.9rem;
  margin: 0.4rem 0 1.4rem 0;}}
.eda-feature {{padding: 1.05rem 1.1rem; border: 1px solid var(--eda-border); border-radius: 0.8rem;
  background: var(--eda-card);}}
.eda-feature .ic {{width: 2rem; height: 2rem; border-radius: 0.5rem; display: grid; place-items: center;
  background: var(--eda-accent-soft); color: var(--eda-accent); font-size: 1rem; margin-bottom: 0.6rem;}}
.eda-feature h4 {{margin: 0 0 0.25rem 0; font-size: 0.95rem; font-weight: 600; color: var(--eda-text);}}
.eda-feature .txt {{margin: 0; font-size: 0.83rem; color: var(--eda-muted); line-height: 1.45;}}
.eda-steps {{display: flex; gap: 1.4rem; flex-wrap: wrap; color: var(--eda-muted); font-size: 0.85rem;}}
.eda-steps b {{display: inline-grid; place-items: center; width: 1.35rem; height: 1.35rem; border-radius: 50%;
  background: var(--eda-accent); color: white; font-size: 0.72rem; margin-right: 0.4rem;}}
@media (max-width: 640px) {{
  .eda-hero {{padding: 1.6rem 1.2rem;}} .eda-hero h1 {{font-size: 1.7rem;}}
  .block-container {{padding-left: 1rem; padding-right: 1rem;}}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS.format(**tok()), unsafe_allow_html=True)


def setup_page() -> None:
    st.set_page_config(page_title="Explorador EDA", page_icon=":material/insights:", layout="wide",
                       initial_sidebar_state="expanded")
    pio.templates["eda"] = go.layout.Template(layout=go.Layout(
        separators=",.",                       # 1.234,5 en ejes y tooltips
        barcornerradius=4, bargap=0.18, bargroupgap=0.06,
        hoverlabel=dict(font_size=12),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=11), title_text=""),
    ))
    pio.templates.default = "plotly+eda"


def style(fig, height: int = 380, title: str | None = None, legend: bool | None = None,
          subtitle: str | None = None):
    """Uniform size, margins, recessive grid and legend for any figure."""
    t = tok()
    named = sum(1 for tr in fig.data if tr.name and tr.showlegend is not False)
    shows = fig.layout.showlegend if fig.layout.showlegend is not None else named >= 2
    if legend is not None:
        shows = legend
    # Legend sits top-right, on the title row: it can never collide with the x-axis title.
    top = 56 if subtitle else (40 if title else (34 if shows else 12))
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=top, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11),
                    title_text=""),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor=t["grid"], linecolor=t["axis"], zerolinecolor=t["grid"], ticks="")
    fig.update_yaxes(gridcolor=t["grid"], linecolor=t["axis"], zerolinecolor=t["grid"], ticks="")
    if title:
        text = f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:{t['muted']}'>{subtitle}</span>"
                                    if subtitle else "")
        fig.update_layout(title=dict(text=text, x=0, xanchor="left", y=0.98, yanchor="top",
                                     font=dict(size=14)))
    if legend is not None:
        fig.update_layout(showlegend=legend)
    return fig


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
