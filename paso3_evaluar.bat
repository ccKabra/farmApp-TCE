@echo off
REM ============================================================================
REM  PASO 3 — Evaluar (correr cuando el entrenamiento este completo)
REM  Corre el modelo sobre el 30% de test y regenera outputs/test_cases.csv,
REM  la tabla que se ve en la pestania "Casos reales de test" de la app.
REM  Se puede correr tambien entre tandas para ver como va mejorando.
REM ============================================================================
setlocal
cd /d "%~dp0"
set PY=venv\Scripts\python.exe

if not exist models\ga_model\weights.npy (
    echo No hay modelo entrenado todavia. Corre paso2_entrenar.bat primero.
    pause
    exit /b 1
)

echo [PASO 3/3] Generando tabla de casos de test...
"%PY%" src\eval_test_cases.py
if errorlevel 1 (
    echo.
    echo ###  FALLO la evaluacion. Revisa el mensaje de arriba.
    pause
    exit /b 1
)
echo.
echo  LISTO. Proba la app:  venv\Scripts\streamlit run app.py
pause
