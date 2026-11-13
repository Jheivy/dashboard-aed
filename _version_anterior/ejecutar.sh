#!/usr/bin/env bash
# Linux / Mac: instala lo necesario la primera vez y abre el dashboard.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Primera vez: instalando dependencias (puede tardar unos minutos)..."
    python3 -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -r requirements.txt
fi

exec .venv/bin/streamlit run app.py "$@"
