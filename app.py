"""
Explorador EDA — análisis exploratorio de datos para cualquier dataset.

Ejecutar:  streamlit run app.py   (o ejecutar.bat / ejecutar.sh)

Estructura:
  eda/    lógica sin interfaz (lectura, tipos, calidad, nulos, outliers, estadística, informe)
  views/  una sección de la interfaz por módulo; cada una expone render(ctx)

Rendimiento:
  - Cada etapa del pipeline (lectura → preparación → SQL → tipos → filtros → muestra) se
    cachea con una *huella* barata del estado, no hasheando el DataFrame en cada clic.
  - Cada sección corre dentro de un st.fragment: tocar un control de la sección solo
    re-ejecuta esa sección.
"""

import hashlib
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

from eda.config import inject_css, setup_page

setup_page()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from eda.coltypes import (NULL_TOKENS, TYPE_OPTIONS, apply_type_overrides,  # noqa: E402
                          prepare, semantic_type, split_columns)
from eda.io import (ENCODINGS, EXCEL_EXT, SEPARATORS, UPLOAD_TYPES, demo_dataset,  # noqa: E402
                    excel_sheets, read_file)
from views import (export, missing, multivariate, outliers, overview, quality,  # noqa: E402
                   relationships, survival, target, timeseries, univariate)
from views._components import chips, feature_grid  # noqa: E402
from views._ui import Ctx, es_int  # noqa: E402

ASSETS = Path(__file__).parent / "assets"
SOURCE_COL = "__origen__"
MAX_ROWS_DEFAULT = 200_000

# (grupo, título, icono, url, render)
SECTIONS = [
    ("Datos", "Resumen", ":material/space_dashboard:", "resumen", overview.render),
    ("Datos", "Calidad de datos", ":material/fact_check:", "calidad", quality.render),
    ("Datos", "Valores faltantes", ":material/grid_off:", "faltantes", missing.render),
    ("Exploración", "Univariante", ":material/bar_chart:", "univariante", univariate.render),
    ("Exploración", "Outliers", ":material/scatter_plot:", "outliers", outliers.render),
    ("Exploración", "Relaciones", ":material/hub:", "relaciones", relationships.render),
    ("Modelado", "Variable objetivo", ":material/target:", "objetivo", target.render),
    ("Modelado", "Multivariante", ":material/bubble_chart:", "multivariante", multivariate.render),
    ("Modelado", "Series temporales", ":material/timeline:", "series", timeseries.render),
    ("Modelado", "Supervivencia", ":material/monitor_heart:", "supervivencia", survival.render),
    ("Salida", "Informe y exportación", ":material/ios_share:", "informe", export.render),
]


def fp(*parts) -> str:
    """Cheap, stable fingerprint of a pipeline state."""
    return hashlib.md5(repr(parts).encode("utf-8", "ignore")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Etapas cacheadas del pipeline (el DataFrame no se hashea: lo identifica `key`).
# cache_resource devuelve el mismo objeto sin copiarlo: las vistas nunca modifican ctx.df.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False, max_entries=4)
def combine(_frames: list, key: str) -> pd.DataFrame:
    return pd.concat([d.assign(**{SOURCE_COL: lbl}) for lbl, d in _frames], ignore_index=True)


@st.cache_resource(show_spinner="Ejecutando SQL…", max_entries=4)
def run_sql(_df: pd.DataFrame, key: str, sql: str) -> pd.DataFrame:
    import duckdb
    # La consulta es texto escrito por el usuario: solo una sentencia SELECT, y sin acceso al disco
    # ni a la red (read_csv('/ruta'), COPY ... TO, ATTACH, extensiones…) — solo ve la tabla `df`.
    stmts = duckdb.extract_statements(sql)
    if len(stmts) != 1 or stmts[0].type != duckdb.StatementType.SELECT:
        raise ValueError("solo se permite una única consulta SELECT.")
    con = duckdb.connect(config={"enable_external_access": False, "autoload_known_extensions": False,
                                 "autoinstall_known_extensions": False, "lock_configuration": True})
    con.register("df", _df)
    return con.sql(sql).df()


@st.cache_resource(show_spinner=False, max_entries=4)
def overrides_stage(_df: pd.DataFrame, key: str, types: dict, overrides: tuple):
    return apply_type_overrides(_df, types, dict(overrides))


@st.cache_resource(show_spinner="Aplicando filtros…", max_entries=4)
def filter_stage(_df: pd.DataFrame, key: str, spec: tuple) -> pd.DataFrame:
    df = _df
    mask = pd.Series(True, index=df.index)
    for col, kind, value, keep_na in spec:
        s = df[col]
        if kind == "fecha":
            lo, hi = (pd.Timestamp(v) for v in value)
            tz = getattr(s.dt, "tz", None)
            if tz is not None:
                lo, hi = lo.tz_localize(tz), hi.tz_localize(tz)
            m = (s >= lo) & (s < hi + pd.Timedelta(days=1))
        elif kind == "num":
            m = s.between(*value)
        else:
            m = s.astype(str).isin(set(value))
        mask &= m | (s.isna() & keep_na)
    out = df[mask].reset_index(drop=True)
    for c in out.select_dtypes(include="category").columns:
        out[c] = out[c].cat.remove_unused_categories()
    return out


@st.cache_resource(show_spinner=False, max_entries=4)
def sample_stage(_df: pd.DataFrame, key: str, n: int) -> pd.DataFrame:
    return _df.sample(n, random_state=0).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Barra lateral: datos
# --------------------------------------------------------------------------- #
try:
    st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")
except Exception:  # noqa: BLE001
    pass
inject_css()

st.sidebar.markdown("#### Datos")
uploaded = st.sidebar.file_uploader("Sube uno o varios archivos", type=UPLOAD_TYPES,
                                    accept_multiple_files=True, label_visibility="collapsed",
                                    help="CSV, TSV, TXT, Excel (.xlsx), Parquet o JSON.")

with st.sidebar.expander("Opciones de lectura", icon=":material/tune:"):
    sep_label = st.selectbox("Separador", list(SEPARATORS), key="opt_sep")
    encoding = st.selectbox("Codificación", ENCODINGS, key="opt_enc")
    decimal = st.radio("Separador decimal al leer", [".", ","], horizontal=True, key="opt_dec",
                       help="Con «.» los números con coma decimal se detectan igualmente en la "
                            "preparación automática.")

frames, load_info, src_parts = [], [], []
if uploaded:
    st.session_state.pop("use_demo", None)
    labels = []
    for f in uploaded:
        fkey = fp(f.name, f.file_id, f.size)
        sheet = None
        if f.name.lower().endswith(EXCEL_EXT):
            try:
                sheets = excel_sheets(f.getvalue(), fkey)
            except Exception as e:  # noqa: BLE001
                st.sidebar.error(f"{f.name}: no se pudo abrir ({e}).")
                continue
            if len(sheets) > 1:
                sheet = st.sidebar.selectbox(f"Hoja de {f.name}", sheets, key=f"sheet_{f.name}")
        rkey = fp(fkey, sep_label, encoding, decimal, sheet)
        try:
            d, info = read_file(f.name, f.getvalue(), rkey, SEPARATORS[sep_label], encoding, decimal, sheet)
        except Exception as e:  # noqa: BLE001
            st.sidebar.error(f"{f.name}: error al leer ({e}).")
            continue
        label = f.name if f.name not in labels else f"{f.name} ({labels.count(f.name) + 1})"
        labels.append(label)
        frames.append((label, d))
        load_info.append(info)
        src_parts.append(rkey)
elif st.session_state.get("use_demo"):
    demo = demo_dataset()
    frames = [("ejemplo_clientes.csv", demo)]
    load_info = [{"archivo": "ejemplo_clientes.csv", "formato": "sintético", "filas": len(demo),
                  "columnas": demo.shape[1]}]
    src_parts = ["demo", len(demo)]


# --------------------------------------------------------------------------- #
# Bienvenida (sin datos)
# --------------------------------------------------------------------------- #
def welcome() -> None:
    st.markdown(
        "<div class='eda-hero'><div class='eda-eyebrow'>Análisis exploratorio de datos</div>"
        "<h1>Entiende cualquier dataset<br>en minutos, no en días.</h1>"
        "<p>Sube un CSV, Excel, Parquet o JSON y obtén al instante un diagnóstico de calidad, "
        "valores faltantes, outliers, distribuciones y relaciones, con los tests estadísticos ya "
        "interpretados.</p></div>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([2, 2.2, 3])
    if c1.button("Probar con datos de ejemplo", type="primary", icon=":material/science:",
                 width="stretch"):
        st.session_state["use_demo"] = True
        st.rerun()
    c2.markdown("<div class='eda-steps' style='padding-top:.45rem'><span><b>1</b>Sube tu archivo en el "
                "panel lateral</span></div>", unsafe_allow_html=True)
    feature_grid([
        ("✓", "Calidad de datos", "Duplicados, nulos disfrazados, códigos −999, categorías mal escritas, "
                                  "tipos mezclados y columnas redundantes, priorizados por gravedad."),
        ("▦", "Valores faltantes", "Patrones, correlación de nulidad, test MCAR de Little, dependencia "
                                   "MAR y simulación de imputaciones."),
        ("◎", "Outliers", "IQR, z robusto, Mahalanobis robusta, Isolation Forest y LOF con consenso "
                          "entre métodos y efecto de tratarlos."),
        ("◩", "Distribuciones", "Histogramas, Q-Q, normalidad, Pareto y comparación entre grupos con "
                                "tamaño del efecto."),
        ("⟷", "Relaciones y objetivo", "Correlaciones, asociación mixta, ranking de variables frente a "
                                       "un target, ROC validada y alertas de fuga."),
        ("⇩", "Informe y exportación", "Informe HTML para compartir y datos preparados en CSV, "
                                       "Parquet o Excel."),
    ])
    st.caption("Los datos se procesan en local, en tu equipo. El ejemplo es sintético e incluye "
               "problemas a propósito para ver qué detecta cada sección.")


if not frames:
    st.navigation([st.Page(welcome, title="Inicio", icon=":material/home:")], position="hidden").run()
    st.stop()

# --------------------------------------------------------------------------- #
# Combinación de archivos
# --------------------------------------------------------------------------- #
source_col = None
if len(frames) == 1:
    raw_df = frames[0][1]
    src_key = fp(*src_parts)
else:
    st.sidebar.caption("Archivos incluidos")
    active = [(lbl, d, k) for (lbl, d), k in zip(frames, src_parts)
              if st.sidebar.checkbox(lbl, True, key=f"inc_{lbl}")]
    if not active:
        st.info("Marca al menos un archivo en el panel lateral.", icon=":material/info:")
        st.stop()
    if len(active) == 1:
        raw_df, src_key = active[0][1], fp(active[0][2])
    else:
        src_key = fp(*[k for _, _, k in active])
        raw_df = combine([(lbl, d) for lbl, d, _ in active], src_key)
        source_col = SOURCE_COL
        cols_sets = [set(d.columns) for _, d, _ in active]
        if any(s != cols_sets[0] for s in cols_sets):
            st.sidebar.warning("Los archivos no tienen las mismas columnas: las que faltan en un archivo "
                               "aparecen como nulos (nulos estructurales).", icon=":material/warning:")

# --------------------------------------------------------------------------- #
# Preparación, SQL, tipos, filtros, muestra
# --------------------------------------------------------------------------- #
with st.sidebar.expander("Preparación", icon=":material/cleaning_services:"):
    auto_convert = st.toggle("Detectar números y fechas en texto", True, key="prep_conv")
    strip_text = st.toggle("Recortar espacios en textos", False, key="prep_strip")
    drop_empty = st.toggle("Quitar filas/columnas vacías", True, key="prep_empty")
    drop_dups = st.toggle("Eliminar filas duplicadas", False, key="prep_dups")
    tokens_txt = st.text_input("Valores a tratar como nulo (separados por ;)", "", key="prep_tokens",
                               placeholder="N/A; -; ?; -999",
                               help="Sin distinguir mayúsculas. Habituales: "
                                    + "; ".join(t for t in NULL_TOKENS if t)[:200])
    null_tokens = tuple(sorted({t.strip() for t in tokens_txt.split(";") if t.strip()}))

prep_key = fp(src_key, null_tokens, strip_text, drop_empty, drop_dups, auto_convert)
df, types, prep_log = prepare(raw_df, prep_key, null_tokens, strip_text, drop_empty, drop_dups,
                              auto_convert)
key = prep_key

with st.sidebar.expander("Consulta SQL", icon=":material/database:"):
    sql = st.text_area("La tabla se llama df", "", height=90, key="sql",
                       placeholder="SELECT * FROM df WHERE edad > 30")
    if sql.strip():
        try:
            key = fp(key, sql.strip())
            df = run_sql(df, key, sql.strip())
            types = {c: types[c] if c in types else semantic_type(df[c], c) for c in df.columns}
            st.caption(f"Resultado: {es_int(len(df))} filas × {df.shape[1]} columnas.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Error SQL: {e}")
            key = prep_key

with st.sidebar.expander("Tipos de columna", icon=":material/category:"):
    st.caption("Corrige el tipo si la detección se ha equivocado. «ignorar» excluye la columna.")
    edited = st.data_editor(
        pd.DataFrame({"columna": list(types), "tipo": list(types.values())}), hide_index=True,
        key=f"types_{fp(tuple(df.columns))}",
        column_config={"columna": st.column_config.TextColumn(disabled=True),
                       "tipo": st.column_config.SelectboxColumn(options=TYPE_OPTIONS, required=True)},
        height=min(36 * (len(types) + 1), 420))
    overrides = tuple(sorted((r.columna, r.tipo) for r in edited.itertuples()
                             if r.tipo != types.get(r.columna)))
if overrides:
    key = fp(key, overrides)
    df, types = overrides_stage(df, key, types, overrides)
cols = split_columns(df, types)
n_loaded = len(df)

with st.sidebar.expander("Filtros", icon=":material/filter_alt:"):
    fcols = st.multiselect("Filtrar por", [c for c in df.columns if types.get(c) != "ignorar"],
                           key="filter_cols", placeholder="Elige columnas")
    spec = []
    for c in fcols:
        s = df[c]
        keep_na = st.checkbox(f"Incluir nulos de {c}", True, key=f"fna_{c}") if s.isna().any() else True
        if c in cols["datetime"]:
            lo, hi = s.min(), s.max()
            if pd.notna(lo) and lo < hi:
                sel = st.date_input(c, (lo.date(), hi.date()), key=f"fdt_{c}", format="DD/MM/YYYY")
                if isinstance(sel, tuple) and len(sel) == 2:
                    spec.append((c, "fecha", (str(sel[0]), str(sel[1])), keep_na))
        elif pd.api.types.is_numeric_dtype(s) and types.get(c) == "numérica":
            lo, hi = float(s.min()), float(s.max())
            if lo < hi:
                sel = st.slider(c, lo, hi, (lo, hi), key=f"fnum_{c}")
                if sel != (lo, hi) or not keep_na:
                    spec.append((c, "num", tuple(sel), keep_na))
        else:
            vals = sorted(s.dropna().astype(str).unique().tolist())
            if 1 < len(vals) <= 500:
                sel = st.multiselect(c, vals, default=vals, key=f"fcat_{c}")
                if len(sel) < len(vals) or not keep_na:
                    spec.append((c, "cat", tuple(sel), keep_na))
            else:
                st.caption(f"{c}: {es_int(len(vals))} valores distintos, demasiados para filtrar aquí "
                           "(usa SQL).")
if spec:
    key = fp(key, tuple(spec))
    df = filter_stage(df, key, tuple(spec))
n_filtered = len(df)

sampled = False
if len(df) > MAX_ROWS_DEFAULT:
    with st.sidebar.expander("Rendimiento", icon=":material/speed:", expanded=True):
        max_rows = st.number_input("Máx. filas a analizar", 10_000, len(df), MAX_ROWS_DEFAULT, 10_000,
                                   key="max_rows", help="Con archivos grandes se analiza una muestra "
                                                        "aleatoria para que todo vaya fluido.")
    if max_rows < len(df):
        key = fp(key, "sample", int(max_rows))
        df = sample_stage(df, key, int(max_rows))
        sampled = True

# --------------------------------------------------------------------------- #
# Navegación superior + barra de contexto + sección activa (en un fragment)
# --------------------------------------------------------------------------- #
CTX = Ctx(df=df, key=key, types=types, numeric=cols["numeric"], categorical=cols["categorical"],
          datetime=cols["datetime"], text=cols["text"], ids=cols["id"], source_col=source_col,
          prep_log=prep_log, load_info=load_info, n_loaded=n_loaded, n_filtered=n_filtered,
          sampled=sampled)


@st.fragment
def run_section(render, ctx: Ctx) -> None:
    render(ctx)


def make_page(render):
    def page() -> None:
        if CTX.df.empty:
            st.warning("No quedan filas con los filtros actuales.", icon=":material/filter_alt_off:")
            return
        run_section(render, CTX)
    return page


start = st.session_state.get("start_page", "Resumen")
groups: dict[str, list] = {}
for group, title, icon, url, render in SECTIONS:
    groups.setdefault(group, []).append(
        st.Page(make_page(render), title=title, icon=icon, url_path=url, default=(title == start)))
pg = st.navigation(groups, position="top")

names = ", ".join(i["archivo"] for i in load_info)
bar = [(names, ""), ("filas", es_int(len(df))), ("columnas", str(df.shape[1]))]
if n_filtered < n_loaded:
    bar.append(("filtrado", f"{es_int(n_filtered)} de {es_int(n_loaded)}"))
if sampled:
    bar.append(("muestra aleatoria", es_int(len(df))))
if prep_log:
    bar.append(("ajustes de preparación", str(len(prep_log))))
chips(bar)
pg.run()
