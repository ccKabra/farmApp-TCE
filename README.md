# Predicción de Efectos Adversos de Fármacos — versión **Computación Evolutiva**

Proyecto de **Técnicas de Computación Evolutiva (TCE)**.

**Integrantes:** Alzugaray Agustín Ezequiel · Lautaro Cabrera Tamalet

Toma reportes reales de pacientes de **FDA FAERS** y predice, para un paciente
dado, el conjunto de **efectos adversos** (vocabulario de ~98 términos MedDRA) que
podría sufrir. Es la **misma tarea, los mismos datos y la misma app** que el
proyecto original `farmApp`, pero el **clasificador ya no se entrena por
retropropagación** (Random Forest / BioBERT): sus pesos se **optimizan con un
Algoritmo Genético (AG)**.

> Este repo es la adaptación de `farmApp` a un enfoque de **Computación Evolutiva**.
> El preprocesamiento, el split train/test, la evaluación, la validación contra
> SIDER y la app Streamlit se conservan; lo que cambia es **cómo se entrena el
> modelo**.

---

## La idea en una frase

El clasificador es una **red neuronal de una capa oculta**. En vez de ajustar sus
pesos con descenso por gradiente, los tratamos como un **cromosoma de números
reales** y dejamos que un **Algoritmo Genético** los evolucione maximizando el
**F1-macro**.

```
                 GENOTIPO (cromosoma real)                    FENOTIPO (red)
   [ W1 | b1 | W2 | b2 ]  ── set_genome ──▶   entrada → Linear → ReLU → Linear → Sigmoide → 98 probabilidades
       6.170 pesos reales                          (154 → 24 → 98)
                                       │
                                  fitness = F1-macro
                                       │
        selección por torneo → cruce → mutación gaussiana → elitismo → nueva generación
```

---

## ¿Por qué Algoritmos Genéticos? (mapa a los módulos de la materia)

| Componente del proyecto | Concepto de la materia |
|---|---|
| Cromosoma = vector real de pesos | **Módulo 3** — genotipos no binarios; *representación = inyección de conocimiento*; localidad |
| Selección por torneo · fitness F1-macro | **Módulo 2 / 4** — selección y diseño de función de fitness |
| Cruce (uniforme/1-punto/aritmético) + mutación gaussiana + elitismo | **Módulo 2** — operadores y elitismo (única garantía teórica de convergencia) |
| σ de mutación decreciente por generación | **Módulo 5** — control determinístico de parámetros |
| Curva de convergencia, fitness best/mean, presión de selección, diversidad | **Módulo 6** — métricas de EAs mono-objetivo |
| NSGA-II precisión vs recall *(extra)* | **Módulo 7** — EAs multi-objetivo / frente de Pareto |
| Importancia de features desde los pesos *(extra)* | **Módulo 9** — XAI + Computación Evolutiva |

---

## Estructura del proyecto

```
farmApp-TCE/
├── app.py                          # App Streamlit (carga el modelo AG)
├── requirements.txt                # El núcleo AG NO necesita PyTorch
├── README.md / DEFENSA.md
├── paso1_datos.bat / paso2_entrenar.bat / paso3_evaluar.bat
├── data/
│   ├── raw/                        # FAERS .txt + SIDER .tsv (no se sube a git)
│   └── processed/                  # dataset.csv, X.csv, Y.csv, label_names.txt, ...
├── models/
│   └── ga_model/                   # ← MODELO EVOLUCIONADO POR EL AG
│       ├── weights.npy             #   cromosoma (mejor individuo)
│       ├── thresholds.npy          #   umbral óptimo por etiqueta
│       ├── label_names.csv
│       ├── featurizer_tfidf.pkl / featurizer.json
│       └── config.json             #   arquitectura + parámetros del AG + métricas
├── outputs/
│   ├── ga_evolution.csv            # historial de la evolución (best/mean/diversidad/...)
│   └── figures/ga_convergence.png  # curva de convergencia
└── src/
    ├── config.py                   # rutas, semilla y sección de parámetros del AG
    ├── patient_text.py             # texto canónico del paciente (única definición)
    ├── ga_features.py        # ★ featurizer: paciente → vector (única definición)
    ├── ga_model.py           # ★ GAClassifier (red NumPy) + load_ga_model()
    ├── train_ga.py           # ★ EL ALGORITMO GENÉTICO (entrenamiento)
    ├── ga_explain.py         # ★ extra XAI (importancia de features)  [Módulo 9]
    ├── train_ga_nsga2.py     # ★ extra multi-objetivo NSGA-II         [Módulo 7]
    ├── labels.py / data_split.py   # vocabulario de etiquetas + split 70/30 determinista
    ├── preprocess.py / explore.py / prepare_features.py
    ├── eval_test_cases.py          # predicción vs realidad sobre el 30% de test (usa AG)
    ├── validate_sider.py           # validación contra SIDER 4.1 (usa AG)
    ├── visualize.py
    ├── model.py / train.py / train_biobert_finetune.py / ...   # baselines heredados (referencia)
    └── smoke_test.py               # test rápido end-to-end (usa AG)
```

★ = nuevo o reescrito para la versión evolutiva.

---

## Instalación

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # solo si vas a correr el NER (extra)
```

> El modelo AG **no requiere PyTorch ni GPU**: corre solo con NumPy/scikit-learn.
> PyTorch/transformers quedan comentados en `requirements.txt` (solo los usan los
> baselines de comparación heredados).

---

## Ejecutar el pipeline

```bash
# Etapa 1: preprocesar FAERS -> dataset.csv  (requiere data/raw, ~55k casos)
python src/preprocess.py

# Etapa 2 (opcional): features TF-IDF para los baselines  -> X.csv / Y.csv
python src/prepare_features.py

# Etapa 3: ENTRENAR el clasificador con el Algoritmo Genético  (~1-2 min, CPU)
python src/train_ga.py

# Etapa 4: tabla de predicción vs realidad sobre el 30% de test
python src/eval_test_cases.py

# Etapa 5: validación contra SIDER 4.1
python src/validate_sider.py

# Extras (opcionales, alineados a los módulos):
python src/ga_explain.py         # importancia de features (XAI, Módulo 9)
python src/train_ga_nsga2.py     # multi-objetivo precisión vs recall (NSGA-II, Módulo 7)
```

En Windows también: `paso1_datos.bat` → `paso2_entrenar.bat` → `paso3_evaluar.bat`.

`train_ga.py` toma `data/processed/dataset.csv` (ya provisto) → construye el
featurizer y el split → corre el AG → guarda `models/ga_model/` y la curva de
convergencia. **No necesita los datos crudos** si `dataset.csv` ya existe.

---

## Correr la app

```bash
streamlit run app.py
```

La interfaz es idéntica a la original: predicción para un paciente nuevo, casos
reales de test (30% nunca visto), validación contra SIDER y la pestaña de análisis
(que ahora incluye la **curva de convergencia del AG**).

---

## Cómo funciona el Algoritmo Genético (`src/train_ga.py`)

1. **Representación (genotipo).** Cada individuo es un vector real
   `[W1 | b1 | W2 | b2]` con los **6.170 pesos** de la red `154 → 24 → 98`.
   Representación **no binaria** (Módulo 3): los operadores trabajan sobre reales.
2. **Inicialización.** Pesos `~ N(0, σ)` con escala tipo He/Glorot; sesgos ≈ 0,
   de modo que las salidas sigmoide arrancan repartidas (el clasificador trivial
   "predecir todo" ya da un F1-macro no nulo → el AG tiene pendiente que escalar).
3. **Fitness.** F1-macro de la red sobre un subconjunto de *train*, calibrando el
   **umbral óptimo por etiqueta** (recompensa cualquier etiqueta cuyo ranking
   mejore → paisaje suave). Análogo a minimizar la pérdida de entrenamiento.
4. **Selección.** Por **torneo** (k=3): presión de selección controlada.
5. **Cruce.** Uniforme por gen (configurable: 1-punto o aritmético/BLX).
6. **Mutación.** **Gaussiana**: cada gen, con probabilidad `pm`, recibe ruido
   `N(0, σ)`. σ **decrece** de 0.20 a 0.02 a lo largo de las generaciones
   (control determinístico, Módulo 5: más exploración al inicio, más explotación
   al final).
7. **Reemplazo + elitismo.** Generacional con **elitismo** (los 2 mejores pasan
   intactos): garantiza que el mejor nunca empeora.
8. **Parada.** Número fijo de generaciones (80). Se registra best/mean fitness,
   diversidad y presión de selección por generación.

Al terminar: se toma el mejor individuo, se **calibran los umbrales por etiqueta
sobre validación** (10% del train) y se reportan las métricas sobre **test** (30%
nunca visto). Todo con el **mismo split determinista (semilla 42)** que el resto
del proyecto, así los casos de test coinciden con los de la app.

---

## Métricas obtenidas

| Modelo | F1 macro | F1 micro | F1 samples | Hamming loss |
|--------|---------:|---------:|-----------:|-------------:|
| Naive Bayes (MultinomialNB) | 0.0001 | 0.0006 | 0.0006 | 0.0217 |
| KNN (k=5, coseno) | 0.0006 | 0.0012 | 0.0006 | 0.0218 |
| **Algoritmo Genético (red 1 capa)** | **0.0732** | **0.0594** | **0.0530** | **0.2832** |
| Random Forest + TF-IDF | 0.107 | 0.104 | 0.105 | 0.245 |
| Random Forest + BioBERT | 0.130 | 0.150 | 0.127 | — |
| BioBERT fine-tuned | 0.128 | 0.106 | 0.098 | 0.059 |

> El AG (con features TF-IDF simples y una red chica) **supera ampliamente** a
> Naive Bayes y KNN y se acerca a Random Forest, demostrando que un AG puede
> entrenar un clasificador multi-label de texto biomédico. **Validación externa
> contra SIDER 4.1: F1 medio ≈ 0.32** (precisión 0.45, recall 0.28) sobre los
> fármacos mapeables. El objetivo no es ganarle a BioBERT sino mostrar la
> aplicabilidad de la Computación Evolutiva (F1 > 0.05 holgado).

El problema es inherentemente difícil (98 etiquetas, densidad ~2%, texto clínico
ruidoso): el F1 absoluto bajo es esperable y común a todos los modelos.

---

## Configuración (`src/config.py`, sección AG)

| Parámetro | Default | Descripción |
|-----------|--------:|-------------|
| `GA_POP_SIZE` | 60 | Tamaño de población |
| `GA_GENERATIONS` | 80 | Generaciones (criterio de parada) |
| `GA_TOURNAMENT_K` | 3 | Tamaño del torneo (presión de selección) |
| `GA_P_CROSSOVER` | 0.9 | Probabilidad de cruce |
| `GA_CROSSOVER` | `uniform` | `uniform` / `one_point` / `arithmetic` |
| `GA_P_MUTATION` | 0.05 | Probabilidad de mutación por gen |
| `GA_MUT_SIGMA` → `GA_MUT_SIGMA_END` | 0.20 → 0.02 | σ gaussiana (decae) |
| `GA_ELITISM` | 2 | Élites que pasan intactos |
| `GA_HIDDEN` | 24 | Neuronas de la capa oculta |
| `TFIDF_MAX_FEATURES` | 150 | Términos TF-IDF (dimensión de entrada) |
| `GA_FIT_SUBSAMPLE` | 5000 | Casos de train para evaluar fitness (acelera) |

La convergencia sigue subiendo al llegar a 80 generaciones: subir
`GA_GENERATIONS` mejora el F1 a costa de tiempo.

---

## Bibliografía

- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
- Deb, K. et al. (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*. IEEE TEC, 6(2), 182–197.
- Eiben, A. E. & Smith, J. E. (2015). *Introduction to Evolutionary Computing* (2nd ed.). Springer.
- Ponzoni, I. *Evolutionary Computing* — apuntes de la materia (Módulos 2–9).
- Kuhn, M. et al. (2016). *The SIDER database of drugs and side effects*. Nucleic Acids Research, 44(D1).
- FDA Adverse Event Reporting System (FAERS). https://www.fda.gov/drugs/questions-research-reporting/fda-adverse-event-reporting-system-faers
