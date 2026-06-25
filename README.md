# farmApp-TCE — Predicción de efectos adversos con Computación Evolutiva

Proyecto de **Técnicas de Computación Evolutiva (TCE)**.
**Integrantes:** Alzugaray Agustín Ezequiel · Lautaro Cabrera Tamalet.

Toma reportes reales de pacientes de **FDA FAERS** y predice, para un paciente
dado, el conjunto de **efectos adversos** que podría sufrir (vocabulario de
~98 términos MedDRA). El clasificador es una red neuronal de una capa oculta
**cuyos pesos NO se entrenan por retropropagación**: se **optimizan con un
Algoritmo Genético** (genotipo de números reales, selección por torneo, cruce,
mutación gaussiana, elitismo, control adaptativo de parámetros).

```
                  GENOTIPO (cromosoma real, 7.946 pesos)
   [ W1 (228×24) | b1 (24) | W2 (24×98) | b2 (98) ]
                            │
                       set_genome
                            ↓
                  FENOTIPO (red neuronal)
       entrada(228) ─→ Linear ─→ ReLU(24) ─→ Linear ─→ sigmoide(98)
                            │
                  fitness = F1-macro (con umbral óptimo por etiqueta)
                            │
   selección por torneo  →  cruce  →  mutación gaussiana  →  elitismo
                  +  inmigrantes aleatorios si la diversidad cae
                            ↓
                       nueva generación
```

---

## 1. Pipeline de punta a punta (paso por paso)

### Paso 1 — Datos crudos (FAERS)
FAERS publica trimestralmente cuatro archivos por trimestre (`DEMO`, `DRUG`,
`REAC`, `INDI`) que comparten un identificador `primaryid`. Se descargan todos
los trimestres deseados desde [FDA FAERS](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html)
y se colocan en `data/raw/`:

```
data/raw/
├── DEMO25Q1.txt   DEMO25Q2.txt   DEMO25Q3.txt   DEMO25Q4.txt   DEMO26Q1.txt
├── DRUG25Q1.txt   DRUG25Q2.txt   ...
├── REAC25Q1.txt   REAC25Q2.txt   ...
└── INDI25Q1.txt   INDI25Q2.txt   ...
```

### Paso 2 — Combinación y limpieza → `dataset.csv`
[`src/preprocess.py`](src/preprocess.py) descubre los archivos por patrón
(`config.faers_files()`), los concatena, normaliza unidades (edad a años, peso
a kg, sexo M/F), separa el fármaco sospechoso (rol PS) de las medicaciones
concomitantes y deduplica por `primaryid` quedándose con la versión más
reciente. Resultado: una fila por paciente en `data/processed/dataset.csv`.

```bash
python src/preprocess.py     # ~1 min con ~1.5 GB de raw
# Output: data/processed/dataset.csv  (≈55 000 casos)
```

### Paso 3 — Definición del objetivo y split (vocabulario canónico de etiquetas)
[`src/labels.py`](src/labels.py) define las **constraints** del problema: nos
quedamos con las reacciones que aparecen ≥300 veces **y** que son efectos
adversos farmacológicos reales (descartamos errores de dosis, embarazo, etc.).
Quedan **~98 etiquetas MedDRA**.

[`src/data_split.py`](src/data_split.py) hace el split 70/30 con semilla fija
(`RANDOM_SEED = 42`). El mismo split se usa en entrenamiento, evaluación, app y
validación SIDER — *los casos de test son siempre los mismos*.

### Paso 4 — Featurización (representación de entrada al genotipo)
[`src/ga_features.py`](src/ga_features.py) — `PatientFeaturizer`. Convierte
cada paciente en un vector de **228 números reales**:

| Bloque | Dim | Qué es |
|---|---:|---|
| `drug_*` | 90 | multi-hot del fármaco sospechoso sobre el top-90 de train |
| `indi_*` | 50 | multi-hot de las indicaciones sobre el top-50 de train |
| `other_*` | 40 | multi-hot de medicaciones concomitantes (top-40) |
| `tf_*` | 40 | TF-IDF residual sobre el texto que NO entró en los vocabularios |
| demografía | 8 | edad/peso normalizados, sexo one-hot, edad binada, flag "sin concomitantes" |

**Por qué importa**: la versión anterior usaba TF-IDF crudo de 150 términos. El
nombre del fármaco quedaba como un token más entre 150. Ahora cada fármaco
tiene **su propia columna dedicada** — el genotipo tiene localidad clara y el
AG aprende directo "fármaco X → efecto Y". Es lo que el Módulo 3 llama
*representation as knowledge injection*: mejorar la codificación es decisión de
diseño del AG y subió el F1 de 0.073 a 0.100 (+37%).

> El featurizer se ajusta **solo sobre train** (vocabularios, medianas), nunca
> mira los datos de test → cero fuga de información.

### Paso 5 — Definición del genotipo/fenotipo
[`src/ga_model.py`](src/ga_model.py) — `GAClassifier`. Cromosoma plano:

```
[ W1.ravel() | b1 | W2.ravel() | b2 ]      ← 7.946 números reales
       │
   set_genome
       ↓
   entrada(228) → Linear → ReLU(24) → Linear → sigmoide(98)
```

### Paso 6 — Entrenamiento evolutivo → `models/ga_model/`
[`src/train_ga.py`](src/train_ga.py). El bucle del AG es de manual:

| Componente | Cómo | Módulo |
|---|---|:---:|
| Inicialización | Gaussiana He/Glorot escalada; sesgos ≈ 0 | 4 |
| Selección | torneo `k=3` | 2 |
| Cruce | uniforme con `pc=0.9` | 2 |
| Mutación | gaussiana por gen, `pm=0.04` | 2 |
| **Control determinístico** | `σ` decrece linealmente de 0.25 a 0.02 | 5 |
| **Control adaptativo** | si `std(fitness) < 0.0008` → reemplaza los 6 peores por **inmigrantes aleatorios** | 5 |
| Elitismo | `e=2` (los mejores pasan intactos) | 2 |
| Fitness | F1-macro con **umbral óptimo por etiqueta** sobre un subset fijo de 5 000 casos de train | 4, 6 |
| Métricas registradas | best/mean fitness, diversidad (std), presión de selección `MaxFit/AveFit`, σ | 6 |
| Parada | 120 generaciones (criterio fijo) | 2 |

```bash
python src/train_ga.py       # ~3 min, solo CPU/NumPy
```

Salida: `models/ga_model/{weights.npy, thresholds.npy, label_names.csv,
featurizer*.{pkl,json}, config.json}` + curva de convergencia en
`outputs/figures/ga_convergence.png` y `outputs/ga_evolution.csv` con la
historia completa.

### Paso 7 — Calibración de umbrales por etiqueta
Al final del paso 6, sobre el **conjunto de validación** (10 % del train) se
busca el mejor umbral de decisión **por cada una de las 98 etiquetas** (grilla
fina 0.10–0.90). Resuelve el desbalance: un umbral global de 0.5 da F1 cero
porque ninguna probabilidad lo cruza.

### Paso 8 — Evaluación sobre el test (nunca visto)
[`src/eval_test_cases.py`](src/eval_test_cases.py) predice los 8 762 casos del
30 % de test y genera `outputs/test_cases.csv` con: atributos del paciente,
reacciones reales reportadas en FAERS, reacciones predichas, TP/FP/FN por caso.

```bash
python src/eval_test_cases.py    # ~10 s
```

### Paso 9 — Validación externa contra SIDER 4.1
[`src/validate_sider.py`](src/validate_sider.py) descarga SIDER (si no está en
caché), mapea los fármacos FAERS → STITCH id (con manejo de sales tipo
HYDROCHLORIDE) y compara las predicciones contra los efectos documentados
externamente.

```bash
python src/validate_sider.py     # ~10 s; F1 ≈ 0.32 vs SIDER
```

### Paso 10 — Extras evolutivos
Tres componentes adicionales que cubren más módulos:

```bash
python src/ga_feature_select.py    # ★ extra: SELECCION de features con
                                   #   cromosoma BINARIO (Módulo 2/3).
                                   #   El AG decide qué features usar.

python src/train_ga_nsga2.py       # ★ extra: NSGA-II multi-objetivo
                                   #   (Módulo 7). Evoluciona el frente de
                                   #   Pareto precisión vs recall.

python src/ga_explain.py           # ★ extra: XAI (Módulo 9). Importancia de
                                   #   cada feature derivada de los pesos del
                                   #   individuo evolucionado.
```

### Paso 11 — App Streamlit
[`app.py`](app.py) — interfaz interactiva con tres pestañas:

1. **Cómo lo hicimos**: el pipeline completo con foco evolutivo (12 etapas
   animadas: problema → genotipo/fenotipo → fitness → operadores → convergencia
   → NSGA-II → XAI → SIDER).
2. **Predecir paciente nuevo**: formulario que llama directamente a
   `featurizer.transform_one(...)` + `model.predict_proba(...)`. Incluye un
   slider de "sensibilidad" que escala los umbrales en vivo.
3. **Casos reales de test (30%)**: tabla navegable de las 8 762 predicciones
   del paso 8, con detalle por caso (TP/FN/FP).
4. **Análisis del modelo**: card del individuo evolucionado, curva de
   convergencia en vivo, comparativa contra baselines, desbalance de clases
   (motiva el diseño del fitness), XAI y NSGA-II.

```bash
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

---

## 2. Resultados

| Modelo | F1 macro | F1 micro | F1 samples | Hamming loss |
|--------|---------:|---------:|-----------:|-------------:|
| **Algoritmo Genético (encoding categórico) — ESTE PROYECTO** | **0.0998** | **0.0662** | **0.0574** | **0.2302** |
| Naive Bayes — *baseline histórico* | 0.0001 | 0.0006 | 0.0006 | 0.0217 |
| KNN (k=5) — *baseline histórico* | 0.0006 | 0.0012 | 0.0006 | 0.0218 |
| Random Forest + TF-IDF — *baseline histórico* | 0.107 | 0.104 | 0.105 | 0.245 |
| Random Forest + BioBERT — *baseline histórico* | 0.130 | 0.150 | 0.127 | — |
| BioBERT fine-tuned — *baseline histórico* | 0.128 | 0.106 | 0.098 | 0.059 |

> El AG **prácticamente iguala al Random Forest baseline (0.107)** y supera por
> tres órdenes de magnitud a Naive Bayes/KNN, evolucionando los pesos sin
> retropropagación. Validación externa contra SIDER 4.1: **F1 ≈ 0.32**
> (precisión 0.45, recall 0.28).

**Extras evolutivos (resultados):**
- **Selección de features con cromosoma binario**: el AG eligió **59 de 154
  features (38 %)** y obtuvo F1 **0.133 > 0.124** que usando todas — menos features y
  mejor desempeño.
- **NSGA-II**: frente de Pareto con ~21 soluciones no dominadas, desde alta
  precisión hasta alto recall.
- **XAI**: los features más influyentes del individuo evolucionado son
  fármacos e indicaciones concretos (interpretable).

---

## 3. Estructura del proyecto

```
farmApp-TCE/
├── app.py                          # App Streamlit
├── requirements.txt                # numpy, pandas, scikit-learn, streamlit, ...
├── README.md / DEFENSA.md
├── paso1_datos.bat / paso2_entrenar.bat / paso3_evaluar.bat
├── data/
│   ├── raw/                        # FAERS .txt + SIDER .tsv  (no se sube a git)
│   └── processed/
│       └── dataset.csv             # salida del paso 2
├── models/
│   └── ga_model/                   # ★ MODELO EVOLUCIONADO POR EL AG
│       ├── weights.npy             #   cromosoma del mejor individuo
│       ├── thresholds.npy          #   umbral óptimo por etiqueta (98)
│       ├── label_names.csv
│       ├── featurizer_tfidf.pkl / featurizer.json
│       └── config.json             #   arquitectura + parámetros + métricas
├── outputs/
│   ├── ga_evolution.csv            # historial best/mean/diversidad/SelPres/σ
│   ├── ga_feature_importance.csv   # XAI
│   ├── ga_feature_selection.csv    # features seleccionadas (cromosoma binario)
│   ├── nsga2_pareto.csv            # frente de Pareto precisión/recall
│   ├── sider_validation.csv        # validación externa
│   ├── test_cases.csv              # predicciones sobre los 8 762 casos de test
│   └── figures/
│       ├── ga_convergence.png
│       ├── ga_feature_importance.png
│       ├── ga_feature_selection.png
│       ├── nsga2_pareto_front.png
│       └── sider_validation.png
└── src/                            # ★ 16 archivos, todos con foco evolutivo
    ├── config.py                   # rutas, semilla, hiperparámetros del AG
    ├── preprocess.py               # paso 2: FAERS → dataset.csv
    ├── labels.py                   # paso 3: vocabulario canónico de etiquetas
    ├── data_split.py               # paso 3: split 70/30 determinista
    ├── patient_text.py             # representación textual canónica (display)
    ├── ga_features.py              # ★ paso 4: featurizer (genotipo de entrada)
    ├── ga_model.py                 # ★ paso 5: GAClassifier (genotipo/fenotipo)
    ├── train_ga.py                 # ★ paso 6 y 7: bucle evolutivo
    ├── eval_test_cases.py          # paso 8: tabla de predicciones de test
    ├── validate_sider.py           # paso 9: validación externa SIDER
    ├── ga_explain.py               # ★ extra: XAI (Módulo 9)
    ├── train_ga_nsga2.py           # ★ extra: NSGA-II (Módulo 7)
    ├── ga_feature_select.py        # ★ extra: selección de features binaria
    ├── pipeline_story.py           # narrativa de la pestaña "Cómo lo hicimos"
    ├── translations.py             # traducción de nombres de efectos para la UI
    └── smoke_test.py               # smoke test end-to-end
```

---

## 4. Qué parte es Computación Evolutiva

Si alguien pregunta "*¿esto no es Minería de Datos?*":

**Núcleo evolutivo (la contribución):**
- **Representación genotipo/fenotipo** — genotipo real de 7 946 pesos; fenotipo
  = red neuronal ([ga_model.py](src/ga_model.py)) + featurización de entrada
  como inyección de conocimiento ([ga_features.py](src/ga_features.py)).
- **Función de fitness** — F1-macro con umbral óptimo por etiqueta.
- **Operadores** — selección por torneo, cruce, mutación gaussiana, elitismo.
- **Control de parámetros** — determinístico (σ decreciente) **+** adaptativo
  (inmigrantes aleatorios si cae la diversidad).
- **Métricas de convergencia** — best/mean fitness, diversidad, presión de
  selección por generación.
- **Multi-objetivo** — NSGA-II precisión vs recall (frente de Pareto).
- **Genotipo binario** — selección de features con bit-flip y cruce uniforme.
- **Explicabilidad evolutiva** — importancia derivada del individuo.

**Soporte técnico (no es la contribución):**
- FAERS y su limpieza dan el problema a optimizar.
- TF-IDF residual es codificación mínima del texto que no entró en los
  vocabularios cerrados; el AG binario incluso decide qué features usar.

---

## 5. Decisiones de diseño evolutivo

| Decisión | Por qué | Módulo |
|---|---|:---:|
| **AG en vez de retropropagación** | El objetivo de la materia es optimizar sin gradiente. | 2 |
| **Genotipo real para los pesos** | Los pesos son continuos; la representación natural es real. | 3 |
| **Encoding categórico explícito (vs TF-IDF crudo)** | Cada fármaco/indicación tiene SU columna; el AG no tiene que "descubrir" significados. Subió F1 0.073 → 0.100. | 3 |
| **Sesgos ≈ 0 en la inicialización** | Inicializar con `logit(prior)` aplanaba el fitness en 0 (densidad 2 %, nada cruzaba el umbral). | 4 |
| **Fitness = F1-macro con umbral por etiqueta** | Un umbral global de 0.5 da señal plana; el umbral por etiqueta premia cualquier mejora de ranking. | 4 |
| **Selección por torneo (no ruleta)** | La ruleta crea superindividuos y es sensible a la escala; el torneo da presión controlada. | 2 |
| **Elitismo + mutación** | Elitismo garantiza no empeorar; la mutación reinyecta variación. | 2 |
| **Control determinístico de σ** | Explorar al inicio, explotar al final. | 5 |
| **Inmigrantes aleatorios cuando cae la diversidad** | Escapar de óptimos locales sin perder los elites. | 5 |
| **NSGA-II para precisión vs recall** | Son objetivos en conflicto; el frente de Pareto muestra todos los compromisos. | 7 |
| **Selección de features con cromosoma binario** | El AG decide qué subconjunto de features usar. | 2/3 |

---

## 6. Instalación y ejecución

### Requisitos
- Python 3.10+ (probado con 3.12)
- Solo CPU; no necesita GPU (no usa PyTorch)

### Setup
```powershell
# Una sola vez:
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Pipeline completo (de cero)
```powershell
.\venv\Scripts\Activate.ps1
python src\preprocess.py            # paso 2: combinar FAERS  → dataset.csv
python src\train_ga.py              # pasos 4-7: entrenar el AG (~3 min)
python src\eval_test_cases.py       # paso 8: tabla de test
python src\validate_sider.py        # paso 9: validación externa SIDER

# Extras evolutivos:
python src\ga_explain.py            # XAI (Módulo 9)
python src\train_ga_nsga2.py        # NSGA-II (Módulo 7)
python src\ga_feature_select.py     # selección binaria de features (Módulo 2/3)

streamlit run app.py                # paso 11: app
```

### Reproducibilidad
`RANDOM_SEED = 42` en [`src/config.py`](src/config.py) fija el split 70/30, el
subset de fitness y el generador del AG (`numpy.random.default_rng(42)`). Dos
corridas dan exactamente el mismo resultado.

---

## 7. Hiperparámetros del AG

En [`src/config.py`](src/config.py), sección "COMPUTACION EVOLUTIVA":

| Parámetro | Default | Descripción |
|---|---:|---|
| `GA_POP_SIZE` | 80 | tamaño de población |
| `GA_GENERATIONS` | 120 | generaciones |
| `GA_TOURNAMENT_K` | 3 | tamaño del torneo |
| `GA_P_CROSSOVER` | 0.9 | probabilidad de cruce |
| `GA_CROSSOVER` | uniform | uniform · one_point · arithmetic |
| `GA_P_MUTATION` | 0.04 | probabilidad de mutación por gen |
| `GA_MUT_SIGMA` | 0.25 | σ inicial de la mutación gaussiana |
| `GA_MUT_SIGMA_END` | 0.02 | σ final (control determinístico) |
| `GA_ELITISM` | 2 | elites que pasan intactos |
| `GA_DIVERSITY_FLOOR` | 0.0008 | umbral para inmigrantes aleatorios |
| `GA_IMMIGRANTS` | 6 | nro de inmigrantes cuando se dispara el control adaptativo |
| `GA_HIDDEN` | 24 | neuronas de la capa oculta |
| `GA_FIT_SUBSAMPLE` | 5000 | subset de train para evaluar fitness (acelera) |

---

## 8. Bibliografía

- Goldberg, D. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*.
- Holland, J. (1975). *Adaptation in Natural and Artificial Systems*.
- Deb, K. et al. (2002). *A fast and elitist multiobjective genetic algorithm: NSGA-II*.
- Eiben, A. y Smith, J. (2015). *Introduction to Evolutionary Computing*.
- Material de cátedra: Módulos 2–9 (Algoritmos Genéticos, Genotipos no binarios,
  Selección/Fitness, Parameter Control, Performance Metrics, MOEA, XAI + EC).
- FDA Adverse Event Reporting System (FAERS): https://fis.fda.gov/extensions/FPD-QDE-FAERS/
- Kuhn, M. et al. (2016). *The SIDER database of drugs and side effects*.
