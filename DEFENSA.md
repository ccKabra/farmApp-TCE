# Ayuda memoria para la defensa — versión Computación Evolutiva

Cheat-sheet de decisiones de diseño del **Algoritmo Genético** y respuestas a
preguntas probables de tribunal. Para uso/estructura ver `README.md`. Acá va el
**por qué** de cada cosa.

---

## 1. El problema y el enfoque en una frase

Clasificación **multi-label**: dado un paciente (edad, sexo, peso, fármaco,
concomitantes, indicación) predecir **qué efectos adversos** puede sufrir (~98
etiquetas MedDRA, densidad ~2%, datos reales de FDA FAERS). En esta versión el
clasificador es una red de una capa oculta cuyos pesos **se optimizan con un
Algoritmo Genético** en lugar de retropropagación.

---

## 2. Genotipo, fenotipo y fitness (lo más importante)

- **Genotipo:** vector real `[W1 | b1 | W2 | b2]` con los 6.170 pesos de la red
  `154 → 24 → 98`. Es una representación **no binaria** (Módulo 3): los operadores
  trabajan sobre números reales, no bits.
- **Fenotipo:** la red clasificadora (NumPy): `entrada → Linear → ReLU → Linear →
  Sigmoide`. `ga_model.GAClassifier.set_genome` desempaqueta el cromosoma en las
  matrices.
- **Fitness:** F1-macro de la red, calibrando el umbral óptimo por etiqueta sobre
  un subconjunto de *train*. Es la función que el AG maximiza.

---

## 3. Decisiones que te pueden preguntar (y la respuesta)

**¿Por qué un AG y no backprop?**
Es el objetivo del trabajo (Técnicas de Computación Evolutiva): demostrar que un
AG puede *entrenar* (optimizar los pesos de) un clasificador, sin gradiente. El
AG explora el espacio de pesos con selección + cruce + mutación.

**¿Por qué el genotipo son los pesos y no, por ejemplo, hiperparámetros?**
Optimizar los pesos *es* el aprendizaje (análogo a entrenar la red). Optimizar
solo hiperparámetros o seleccionar features (Opciones B/C del enunciado) no
reemplazaría el entrenamiento. Elegimos la Opción A: el AG aprende los pesos.

**¿Por qué features TF-IDF + demográficos y no los embeddings BioBERT (768d)?**
Tamaño del cromosoma. Con 768 dimensiones de entrada el genoma tendría ~100k
pesos: un AG sobre 100k reales converge pésimo. Con TF-IDF (150) + edad/sexo/peso
el genoma es de 6.170 pesos, manejable. Además TF-IDF se calcula sobre el **mismo
texto canónico** del paciente (`patient_text.row_to_text`) que usaba BioBERT, así
no cambiamos la representación del paciente.

**¿Por qué la función de fitness usa F1-macro con umbral por etiqueta y no a 0.5
fijo?**
Probamos umbral global fijo: con densidad ~2% el F1-macro queda **plano** (casi
todo da 0) y el AG no tiene señal. El umbral óptimo por etiqueta hace que el
fitness **premie cualquier etiqueta cuyo ranking mejore** → paisaje suave y
escalable. Es además coherente con cómo se evalúa el modelo final.

**¿Por qué inicializan los sesgos de salida en ~0 y no en logit(frecuencia)?**
Lo intentamos (inyección de conocimiento, Módulo 3) pero con densidad ~2% las
salidas colapsan en ~0.02, **ninguna cruza el umbral** y el fitness arranca en 0
(plano). Con sesgos ≈ 0 las salidas arrancan repartidas alrededor de 0.5: el
clasificador trivial "predecir todo" ya da F1-macro ≈ densidad, dándole al AG una
pendiente para escalar.

**¿Por qué selección por torneo y no ruleta?**
La ruleta (Módulo 2) sufre con **superindividuos** (un individuo muy fit acapara
la selección → convergencia prematura) y es sensible a la escala/!signo del
fitness. El torneo (k=3) da **presión de selección controlada** y es invariante a
la escala. Monitoreamos `SelPres = MaxFit/AveFit` (Whitley recomienda ≈1.5).

**¿Cómo evitan la convergencia prematura / pérdida de diversidad? (Módulo 2)**
Presión de selección baja (torneo k=3), **mutación gaussiana** que reinyecta
variación, y **elitismo** moderado (solo 2). Registramos la **diversidad** (desvío
del fitness) y la presión de selección por generación en `outputs/ga_evolution.csv`.

**¿Qué operadores de cruce/mutación y por qué? (Módulo 3)**
Cruce **uniforme** por gen por defecto (también disponibles 1-punto y aritmético/
BLX, que mezcla pesos por promedio ponderado — bueno para reales). Mutación
**gaussiana** (el análogo real del bit-flip): suma ruido `N(0,σ)` a genes elegidos.

**¿Controlan parámetros durante la corrida? (Módulo 5)**
Sí: **control determinístico** de la mutación, σ decrece de 0.20 a 0.02 a lo largo
de las generaciones (más exploración al principio, más explotación al final).

**¿Qué métricas de convergencia muestran? (Módulo 6)**
Best y mean fitness por generación (curva de convergencia), diversidad y presión
de selección. La curva (`outputs/figures/ga_convergence.png`) muestra que el mejor
y la media suben juntos y que la convergencia sigue activa a la gen 80.

**¿Y el multi-objetivo? (Módulo 7)**
Precisión y recall **están en conflicto** en este problema. `train_ga_nsga2.py`
implementa **NSGA-II** (fast-non-dominated-sort + crowding distance) para evolucionar
el **frente de Pareto** precisión-vs-recall, en vez de colapsarlo en un solo F1.

**¿Explicabilidad? (Módulo 9)**
`ga_explain.py` deriva la **importancia de cada feature** desde los pesos
evolucionados (norma de las filas de W1, propagada por W2): qué términos TF-IDF y
qué variables demográficas pesan más en las predicciones del modelo.

**¿Cómo evitan el data leakage y garantizan consistencia con la app?**
El featurizer (TF-IDF, medianas) se fitea **solo sobre train**. Usamos el **mismo
split determinista (semilla 42, `data_split.load_split`)** que la app/eval/SIDER, y
el **mismo vocabulario canónico de 98 etiquetas** (frecuencia ≥300 + filtro de
no-ADR). *No* usamos `X.csv`/`Y.csv` (que `prepare_features.py` arma con frecuencia
≥50 y otro filtrado): tienen otro conjunto de etiquetas/filas y romperían la
compatibilidad con la app y la evaluación.

**¿Qué garantiza que la app prediga igual que como se entrenó?**
`app.py` importa `load_ga_model` y el `PatientFeaturizer` del mismo `src/` — no
reimplementa la featurización. El featurizer es la **única definición** de
paciente→vector (el análogo de `patient_text.py` para el AG).

---

## 4. Limitaciones honestas (mejor decirlas vos)

- **F1 absoluto bajo:** inherente al problema (98 clases, densidad ~2%). Lo
  relevante es la mejora relativa (>>NB/KNN, cerca de RF) y la metodología.
- **Sobre-predicción (recall alto, precisión baja):** el umbral óptimo por
  etiqueta, con precisión intrínsecamente baja, tiende a predecir de más. El
  slider de sensibilidad de la app permite endurecer los umbrales. Es la misma
  forma que tienen RF y BioBERT en este dataset.
- **Convergencia más lenta que backprop:** un AG sobre 6.170 reales explora, no
  sigue el gradiente; por eso fijamos features chicos y red chica. Subir
  generaciones/poblacion mejora, a costa de tiempo.

---

## 5. Reproducibilidad

`RANDOM_SEED = 42` en `config.py` fija: el split 70/30, el subconjunto de fitness y
el generador del AG (`numpy.random.default_rng(42)`). Dos corridas dan el mismo
resultado.
