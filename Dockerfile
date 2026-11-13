# syntax=docker/dockerfile:1
# Explorador EDA — imagen autocontenida (Streamlit + librerías científicas).
#   docker build -t explorador-eda .
#   docker run --rm -p 8501:8501 explorador-eda     ->  http://localhost:8501

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencias primero: esta capa solo se reconstruye si cambia requirements.txt.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Usuario sin privilegios.
RUN useradd --create-home --uid 1000 eda
COPY --chown=eda:eda . .
USER eda

EXPOSE 8501

# Streamlit expone un endpoint de salud; sin curl, usamos Python.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

# headless + 0.0.0.0 son necesarios dentro de un contenedor.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
