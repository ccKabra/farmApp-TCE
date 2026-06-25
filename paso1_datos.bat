@echo off
REM ============================================================================
REM  PASO 1 — Preparar datos (CORRER UNA SOLA VEZ)
REM  Combina los trimestres de data/raw y samplea SAMPLE_SIZE casos.
REM  Tarda unos minutos (carga ~1.5 GB). No usa GPU.
REM  Resultado: data/processed/dataset.csv
REM ============================================================================
setlocal
cd /d "%~dp0"
set PY=venv\Scripts\python.exe

echo [PASO 1/3] Preparando datos FAERS...
"%PY%" src\preprocess.py
if errorlevel 1 (
    echo.
    echo ###  FALLO el preprocesamiento. Revisa el mensaje de arriba.
    pause
    exit /b 1
)
echo.
echo  LISTO. dataset.csv generado. Ahora podes correr paso2_entrenar.bat
pause
