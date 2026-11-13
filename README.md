# Explorador EDA

Aplicación para hacer **análisis exploratorio de datos (EDA) de cualquier dataset**. Subes un archivo y la
herramienta detecta los tipos, revisa la calidad y te guía por nulos, outliers, distribuciones y relaciones,
con tests estadísticos ya interpretados.

## Cómo ejecutarlo
1. Necesitas **Python 3.10 o superior**.
2. Abre la carpeta y ejecuta:
   - **Windows:** doble clic en `ejecutar.bat`
   - **Linux / Mac:** `./ejecutar.sh`
3. Se abre el navegador (si no, entra en http://localhost:8501).
4. Sube tu archivo en la barra lateral, o pulsa **«Probar con un dataset de ejemplo»**.

La primera vez tarda unos minutos porque instala las dependencias. Si cambia `requirements.txt`,
el lanzador las actualiza solo. Para cerrar, pulsa `Ctrl + C` en la terminal.

## Con Docker (para compartirlo sin instalar nada)
Solo hace falta tener Docker. Desde esta carpeta:

```bash
docker compose up --build        # construye y arranca → http://localhost:8501
```

O sin compose:

```bash
docker build -t explorador-eda .
docker run --rm -p 8501:8501 explorador-eda
```

- **Otro puerto:** `EDA_PORT=9000 docker compose up` (Windows PowerShell: `$env:EDA_PORT=9000; docker compose up`).
- **Más memoria** para datasets grandes: `EDA_MEMORY=8g docker compose up`.
- **Pasarlo a otra persona:** `docker save explorador-eda | gzip > explorador-eda.tar.gz` y, en el otro equipo,
  `docker load < explorador-eda.tar.gz` (o súbelo a un registro con `docker push`).
- Los archivos se suben desde el navegador; los datos no salen de tu equipo.
- La imagen corre con un usuario sin privilegios y tiene un *healthcheck* en `/_stcore/health`.

## Formatos admitidos
CSV / TSV / TXT (separador y codificación automáticos), Excel `.xlsx` (con selección de hoja),
Parquet y JSON. Puedes subir **varios archivos**: se apilan y se añade la columna `__origen__`.

## Barra lateral
| Bloque | Qué hace |
|---|---|
| Opciones de lectura | Fuerza separador, codificación o decimal si la detección automática falla. |
| Sección | Navegación. Solo se calcula la sección activa, así que la app va fluida. |
| Preparación de datos | Convierte a número o fecha los valores guardados como texto (`1.234,5`, `12%`, `31/12/2024`), recorta espacios, quita filas o columnas vacías y duplicados, y trata como nulo valores como `N/A` o `-999`. |
| Consulta SQL | Filtra o transforma con SQL (DuckDB). La tabla se llama `df`. |
| Tipos de columna | Corrige el tipo detectado (numérica, categórica, fecha, identificador, ignorar…). |
| Filtros | Por rango, categorías o fechas. Se aplican a todas las secciones. |

## Secciones
- **Resumen**: tamaño, tipos, principales problemas, perfil de cada columna con mini-histogramas y vista de los datos.
- **Calidad de datos**: alertas priorizadas por gravedad (duplicados, nulos disfrazados, códigos centinela, categorías mal escritas, tipos mezclados, columnas constantes o redundantes, fechas imposibles…), búsqueda de duplicados por clave y consistencia de texto.
- **Valores faltantes**: % por columna y por fila, matriz y patrones de nulos, correlación de nulidad, **test MCAR de Little**, de qué variables depende la ausencia (MCAR/MAR) y **simulación de imputación** (media, mediana, moda, interpolación, KNN…) con su efecto sobre la distribución.
- **Univariante**: histograma, boxplot, Q-Q, ECDF, estadísticos completos, tests de normalidad, comparación entre grupos con tamaño del efecto, frecuencias y Pareto para categóricas, y rejillas de todas las variables.
- **Outliers**: resumen de todas las variables (IQR, z-score, z robusto MAD), detalle de cada variable con el impacto de eliminar o recortar, y detección **multivariante** (Mahalanobis robusta MCD, Isolation Forest, LOF) con consenso entre métodos y descarga de las filas marcadas.
- **Relaciones**: matriz de correlación (Pearson, Spearman o Kendall), asociación mixta (η, V de Cramér), análisis de cualquier par de variables con su test, pairplot y correlación estratificada (variables de confusión).
- **Variable objetivo**: ranking de las variables más relacionadas con un target (clasificación o regresión), información mutua, gráficos variable vs. objetivo, curvas ROC con un modelo combinado validado y aviso de fuga de información.
- **Multivariante**: PCA (scree plot, biplot, cargas) y clustering K-means con selección de k por silueta.
- **Series temporales**: evolución con frecuencia y agregación a elegir, media móvil, huecos sin datos y estacionalidad.
- **Supervivencia**: Kaplan-Meier con intervalos de confianza y test log-rank.
- **Informe y exportación**: informe HTML descargable y datos preparados en CSV, Parquet o Excel.

## Rendimiento
Cada etapa (lectura → preparación → SQL → tipos → filtros → muestra) se cachea con una huella barata del
estado, y cada sección corre en un `st.fragment`: mover un control solo re-ejecuta esa sección. Los textos se
procesan sobre sus valores únicos, y histogramas y boxplots se agregan en el servidor (viaja un resumen, no
las filas). Con 500.000 filas, un clic tarda ~0,1–0,3 s; la primera carga, unos segundos.

## Estructura del código
```
app.py            barra lateral, carga, preparación, filtros y navegación superior
Dockerfile        imagen autocontenida (+ docker-compose.yml y .dockerignore)
.streamlit/       tema claro/oscuro, tipografía y paletas de gráficos
assets/           logo
eda/              lógica sin interfaz (reutilizable y testeable)
  charts.py       histogramas y boxplots pre-agregados
  io.py           lectura de archivos y dataset de ejemplo
  coltypes.py     preparación y detección de tipos
  profiling.py    perfil de columnas y alertas de calidad
  missing.py      nulos: patrones, test de Little, dependencia, imputación
  outliers.py     detección univariante y multivariante
  stats_utils.py  descriptivos, tests, tamaños de efecto, asociación
  report.py       informe HTML
  config.py       estilo visual compartido
views/            una sección de la interfaz por archivo (render(ctx))
_version_anterior/  copia de la versión previa del dashboard
```
