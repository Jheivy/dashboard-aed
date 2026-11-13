#!/usr/bin/env bash
# Linux / Mac: instala lo necesario (la primera vez o si cambia requirements.txt) y abre el dashboard.
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Primera vez: creando entorno e instalando dependencias, puede tardar unos minutos..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --quiet --upgrade pip
fi

if ! cmp -s requirements.txt .venv/requirements.instalados; then
    echo "Instalando/actualizando dependencias..."
    .venv/bin/python -m pip install --quiet -r requirements.txt
    cp requirements.txt .venv/requirements.instalados
fi

.venv/bin/python -m streamlit run app.py
