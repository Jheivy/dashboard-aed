@echo off
REM Windows: instala lo necesario la primera vez y abre el dashboard.
cd /d "%~dp0"

if not exist ".venv" (
    echo Primera vez: instalando dependencias, puede tardar unos minutos...
    python -m venv .venv
    .venv\Scripts\python -m pip install --quiet --upgrade pip
    .venv\Scripts\python -m pip install --quiet -r requirements.txt
)

.venv\Scripts\python -m streamlit run app.py
pause
