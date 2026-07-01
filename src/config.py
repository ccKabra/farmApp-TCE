from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# FAERS: se cargan TODOS los trimestres presentes en data/raw (2025Q1..2026Q1).
# preprocess.py descubre los archivos por patron, no hace falta listarlos.
def faers_files(prefix):
    """Todos los archivos de un tipo (DEMO/DRUG/REAC/INDI) ordenados por trimestre."""
    return sorted(DATA_RAW.glob(f"{prefix}*.txt"))

# Sampling
SAMPLE_SIZE = 55000     # casos FAERS a usar. Subir = mas datos = mejor (mas lento)
RANDOM_SEED = 42
TEST_SIZE = 0.30        # 70% train / 30% test

# ── Perillas de entrenamiento ─────────────────────────────────────────────────
# El entrenamiento es RESUMABLE y POR TANDAS (ver src/train.py):
#   - TOTAL_EPOCHS    = cuantas epocas en total quiere entrenar el modelo.
#   - EPOCHS_PER_RUN  = cuantas hace CADA vez que corres paso2_entrenar.bat.
# Ej: TOTAL=15, PER_RUN=3 -> corres el .bat 5 veces, 3 epocas cada una.
# Podes parar entre tandas (o interrumpir a mitad: guarda tras cada epoca).
TOTAL_EPOCHS    = 15    # objetivo total de epocas
EPOCHS_PER_RUN  = 3     # epocas por corrida del .bat (mas bajo = sesiones mas cortas)

# Carga sobre la PC: bajar BATCH_TRAIN si se queda sin memoria de GPU o para
# que la maquina quede mas libre mientras entrena.
BATCH_TRAIN     = 32
POS_WEIGHT_CAP  = 10.0  # tope del peso de clases (mas alto = mas recall, mas FP)

# Frecuencia minima para que una reaccion sea etiqueta del modelo.
# Con 55k casos, 300 deja ~98 etiquetas bien representadas (>=300 casos c/u).
# Subir = menos etiquetas pero mejor aprendidas; bajar = mas etiquetas raras.
MIN_REACTION_FREQ = 300

# (heredados; el flujo viejo de 2 scripts los usa, el nuevo train.py no)
BASE_EPOCHS     = 5
EXTENDED_EPOCHS = 10

# Model
BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.2"
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# ══════════════════════════════════════════════════════════════════════════════
#  COMPUTACION EVOLUTIVA — Algoritmo Genetico (proyecto farmApp-TCE)
# ══════════════════════════════════════════════════════════════════════════════
# En esta version el clasificador NO se entrena por retropropagacion (como BioBERT)
# sino que sus pesos se OPTIMIZAN con un Algoritmo Genetico (AG). El fenotipo es
# una red neuronal de una capa oculta; el genotipo es el vector real de sus pesos
# (representacion NO binaria, Modulo 3 de la materia). Ver src/train_ga.py.

GA_MODELS_DIR = MODELS_DIR / "ga_model"   # donde se guarda el modelo evolucionado

# ── Featurizacion (entrada a la red) ──────────────────────────────────────────
# TF-IDF sobre el MISMO texto canonico del paciente (patient_text.row_to_text) que
# usa BioBERT, + features estructurados (edad, sexo, peso). Se mantiene chico para
# que el cromosoma sea manejable por el AG.
TFIDF_MAX_FEATURES = 150   # nro de terminos TF-IDF (dimension principal de entrada)
TFIDF_MIN_DF       = 5      # frecuencia documental minima de un termino

# ── Arquitectura del fenotipo (red de 1 capa oculta) ──────────────────────────
GA_HIDDEN = 24             # neuronas en la capa oculta (ReLU); salida = sigmoide

# ── Hiperparametros del Algoritmo Genetico ────────────────────────────────────
GA_POP_SIZE      = 80      # tamanio de poblacion (PopSize)
GA_GENERATIONS   = 300     # nro de generaciones (criterio de parada fijo)
GA_TOURNAMENT_K  = 3       # tamanio del torneo (presion de seleccion)
GA_P_CROSSOVER   = 0.9     # probabilidad de cruce (pxover)
GA_CROSSOVER     = "uniform"   # "uniform" | "one_point" | "arithmetic"
GA_P_MUTATION    = 0.04    # probabilidad de mutacion POR GEN (pmut)
GA_MUT_SIGMA     = 0.25    # desvio inicial de la mutacion gaussiana (sigma_0)
GA_MUT_SIGMA_END = 0.02    # desvio final (control deterministico: sigma decrece, Modulo 5)
GA_ELITISM       = 2       # nro de elites que pasan intactos (Modulo 2: elitismo)
GA_INIT_SCALE    = 1.0     # escala de la inicializacion gaussiana de pesos
# ── Seleccion (Modulo 4) ──────────────────────────────────────────────────────
GA_SELECTION      = "rank"   # "tournament" | "rank" (rank = presion constante, cf. ga_metrics)
GA_RANK_S         = 1.7      # presion selectiva para rank lineal (1.0=neutra, 2.0=max)

# ── Control ADAPTATIVO (Modulo 5) ─────────────────────────────────────────────
# Ahora la reinyeccion de inmigrantes se dispara por diversidad GENOTIPICA
# (std promedio de los cromosomas), no por diversidad de fitness -> es una
# senal directa del colapso de la poblacion, no una consecuencia.
GA_DIV_LOW           = 0.090   # umbral bajo de diversidad -> inyectar diversidad
GA_DIV_HIGH          = 0.115   # umbral alto de diversidad -> reducir mutacion
GA_IMMIGRANTS        = 8       # nro de inmigrantes aleatorios cuando diversidad<LOW
GA_IMMIGRANT_PERIOD  = 15      # cada cuantas gens se pueden inyectar inmigrantes
GA_SIGMA_BOOST       = 3.0     # multiplicador de sigma cuando diversidad<LOW
GA_SIGMA_DAMP        = 0.5     # multiplicador de sigma cuando diversidad>HIGH

# Conjunto de fitness: subconjunto de TRAIN para acelerar (None = usar todo train).
GA_FIT_SUBSAMPLE = 5000
# Grilla de umbral GLOBAL usada DENTRO del fitness (suaviza la senal de F1-macro).
GA_FITNESS_GRID  = (0.30, 0.40, 0.50, 0.60)
# Fraccion del train reservada como validacion (calibrar umbrales por etiqueta).
GA_VAL_FRACTION  = 0.10
