@echo off
REM ============================================================================
REM  PASO 2 - Entrenar el clasificador con el ALGORITMO GENETICO
REM
REM  Optimiza los pesos de una red de 1 capa oculta con un AG (sin GPU, solo
REM  NumPy). Corre de una sola vez (~1-2 min en CPU). Para ajustar poblacion,
REM  generaciones u operadores, edita la seccion GA de src/config.py.
REM
REM  Resultado: models/ga_model/ (weights.npy, thresholds.npy, config.json, ...)
REM             y la curva de convergencia en outputs/figures/ga_convergence.png
REM ============================================================================
setlocal
cd /d "%~dp0"
set PY=venv\Scripts\python.exe

if not exist data\processed\dataset.csv (
    echo No existe data\processed\dataset.csv
    echo Corre primero paso1_datos.bat
    pause
    exit /b 1
)

echo [PASO 2/3] Entrenando el clasificador con Algoritmo Genetico...
"%PY%" src\train_ga.py
if errorlevel 1 (
    echo.
    echo ###  FALLO el entrenamiento. Revisa el mensaje de arriba.
    pause
    exit /b 1
)
echo.
echo  LISTO. Modelo evolucionado guardado en models\ga_model.
echo  Ahora corre paso3_evaluar.bat para regenerar la tabla de test.
pause
