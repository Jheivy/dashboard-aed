@echo off
REM Windows: instala lo necesario (la primera vez o si cambia requirements.txt) y abre el dashboard.
cd /d "%~dp0"

if not exist ".venv" (
    echo Primera vez: creando entorno e instalando dependencias, puede tardar unos minutos...
    python -m venv .venv
    .venv\Scripts\python -m pip install --quiet --upgrade pip
)

fc /b requirements.txt .venv\requirements.instalados >nul 2>&1
if errorlevel 1 (
    echo Instalando/actualizando dependencias...
    .venv\Scripts\python -m pip install --quiet -r requirements.txt && copy /y requirements.txt .venv\requirements.instalados >nul
)

.venv\Scripts\python -m streamlit run app.py
pause
