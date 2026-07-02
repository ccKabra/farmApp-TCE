# Guión de la defensa — farmApp-TCE

Duración objetivo: **20 minutos**. Las slides ahora son **visuales** (gráficos grandes + palabras clave). Todo lo explicativo está acá.

---

## Slide 1 — Portada (30 seg)

"Buenos días. Presento **farmApp-TCE**: predicción de efectos adversos de fármacos entrenada con **Algoritmo Genético** en vez de retropropagación. Cubre 7 de los 9 módulos del curso."

---

## Slide 2 — Descripción del problema (1 min)

**Pregunta central:** dado un paciente que toma un fármaco, ¿qué efectos adversos va a sufrir?

**Por qué es multi-label:** un paciente puede tener varios efectos a la vez (Náusea, Ansiedad, Cefalea…). Trabajamos con 98 efectos del vocabulario MedDRA.

**El problema del desbalance:** solo ~2% de cada efecto es positivo. Un modelo que predice "no" a todo tiene 98% de accuracy pero F1 ≈ 0. **Por eso usamos F1-macro** — promedia el F1 de cada etiqueta, penaliza que ignores clases raras.

**Por qué importa:** farmacovigilancia post-mercado. Metformina, Adalimumab, Ozempic tuvieron efectos descubiertos post-aprobación. Predecir antes ayuda a la seguridad del paciente.

---

## Slide 3 — Datasets (1 min)

**FAERS (FDA):** base pública de reportes reales. 55.000 casos, 5 trimestres (2025Q1 → 2026Q1). Cada caso viene en 4 archivos: DEMO + DRUG + REAC + INDI.

**SIDER 4.1:** base curada por farmacólogos. Para cada fármaco, sus efectos adversos conocidos. **No la usamos para entrenar** — es validación externa post-hoc.

**Cómo armamos X e Y:** de FAERS extraemos el perfil del paciente (X: fármaco, indicación, edad, sexo, peso, otras medicaciones) y el vector binario de 98 efectos (Y).

---

## Slide 4 — Arquitectura (1.5 min)

**Pipeline de 6 pasos** (izquierda):
1. Cargar FAERS crudo.
2. Featurizer → vector X de 228 números.
3. Red neuronal 228 → 24 → 98 predice Ŷ.
4. F1-macro compara Ŷ con Y → **fitness**.
5. **AG** ajusta los 7.946 pesos.
6. Validación externa con SIDER.

**Genotipo (derecha, rojo):** vector real de 7.946 pesos = W1 (228×24) + b1 (24) + W2 (24×98) + b2 (98).

**Fenotipo (derecha, azul):** red 1 capa oculta, ReLU + sigmoide.

**Frase clave:** "El cromosoma son los pesos de la red. El AG no usa gradiente — combina y muta pesos hasta que la red predice bien."

---

## Slide 5 — Representación (Módulo 3) (1.5 min)

**Decisión más importante del AG: cómo codificar el problema.**

**Por qué representación REAL y no binaria:**
- Los pesos de una red son continuos. Codificar en bits pierde precisión.
- Los operadores naturales para reales están cubiertos en Módulo 3: cruce aritmético y mutación gaussiana.

**Propiedad clave: LOCALIDAD.** Cada gen del cromosoma es un peso individual. Una mutación chica cambia un peso poco → cambia el fenotipo poco. Es lo que el material llama "perfect representation".

**Conexión genotipo → fenotipo:** `set_genome(cromosoma)` desempaqueta los 7.946 números en W1, b1, W2, b2. El forward pass es `Ŷ = sigmoide(ReLU(X·W1 + b1) · W2 + b2)`.

---

## Slide 6 — Operadores evolutivos (Módulos 2 y 4) (2 min)

**Selección por RANK lineal (Módulo 4, s=1.7):** cada individuo recibe probabilidad según su posición en el ranking, NO según fitness absoluto. Presión selectiva **constante = 1.7**, independiente de la escala del fitness y del colapso de la población. Módulo 4 destaca que rank "*controls selection pressure independently of fitness scale*" — clave cuando el fitness converge a un rango chico.

**Cruce uniforme (Módulo 2, p=0.9):** para cada gen, hijo hereda del padre A o B con prob 0.5. Operador **primario** — recombina building blocks de individuos buenos.

**Mutación gaussiana (Módulo 2, p=0.04):** cada gen con 4 % de probabilidad recibe ruido N(0, σ). Operador **secundario** — reinyecta variación, evita perder alelos.

**Elitismo + control adaptativo (Módulos 2 y 5):**
- Los 2 mejores pasan intactos → garantiza monotonía del mejor.
- Diversidad medida como **std promedio del genotipo** (no del fitness).
- Si diversidad < 0.09: inyectamos 8 inmigrantes aleatorios + sigma × 3 (boost).
- Si diversidad > 0.115: sigma × 0.5 (damp).

---

## Slide 7 — Cómo evolucionó el fitness (2.5 min)

**Este gráfico es central. Muestra 3 métricas independientes.**

**Panel izquierdo — Convergencia:**
- Mejor fitness sube de **0.05 a 0.14** en 300 gens.
- Fitness medio (azul) sube en paralelo → toda la población mejora, no solo el elite.
- **Líneas violetas verticales = 5 inyecciones de inmigrantes**. Cada una hace un bache en el fitness medio, pero el mejor no baja (elitismo).

**Panel central — Diversidad genotípica:**
- Ya NO es `std(fitness)` (que colapsa a 0). Es std promedio del **cromosoma**.
- Arranca en 0.13, baja gradualmente, toca el umbral bajo 0.09 → **dispara inmigrantes** → rebota a 0.14.
- **El ciclo cae → inyecta → recupera se ve 5 veces**. Módulo 5 en acción.

**Panel derecho — Parámetros de control:**
- **Línea violeta constante en 1.7:** presión selectiva **teórica** de rank lineal. Constante por diseño (Módulo 4: rank controla la presión independientemente de la escala del fitness).
- **Línea verde (sigma):** combina control **determinístico** (Módulo 5: decrece linealmente 0.25 → 0.02) con control **adaptativo** (Módulo 5: × 3 cuando la diversidad cae, × 0.5 cuando sube). Por eso ven picos, no una recta pura — los dos controles conviven.

**Mensaje:** antes teníamos gráficos planos porque `std(fitness)` y `max/mean` colapsan cuando la población converge. Cambiamos a métricas independientes — cuentan cosas distintas y pueden moverse por separado.

---

## Slide 8 — Ablación (1.5 min)

**Le sacamos al AG un componente a la vez.** Mismo presupuesto: 2.040 evaluaciones de fitness en todas.

- **Random search** (sin AG): F1 test 0.051 — el peor.
- **Solo mutación:** 0.059.
- **Solo cruce:** 0.066 — mejor que solo mutación → **cruce es primario** (Goldberg, Módulo 2).
- **Sin elitismo:** 0.075 — sube pero permite regresiones.
- **AG completo:** 0.065.

**Random search → AG completo: F1 +27%.** Con el mismo presupuesto de evaluaciones, la evolución hace algo más que muestrear al azar — la presión de selección + los operadores guían la búsqueda.

**Cruce vs mutación:** solo cruce > solo mutación. Coincide con el SGA de Goldberg del Módulo 2.

---

## Slide 9a — Experimento p_mutation (1 min)

Barrido de p_mut = 0.01, 0.03, 0.05, 0.10, 0.20, 0.30.

- **0.01:** no explora → 0.092 fitness.
- **0.05:** 0.100.
- **0.10 (mejor):** 0.103.
- **0.30:** 0.092 — destruye soluciones más rápido de lo que las construye.

**Patrón en U invertida** — exactamente lo que predice la teoría. La configuración final elegida es p_mut = 0.04 (dentro del plateau óptimo 0.03–0.10, priorizando estabilidad).

---

## Slide 9b — Experimento p_crossover (1 min)

Barrido de p_cx = 0.30, 0.60, 0.90, 1.00.

- **0.30:** 0.086.
- **0.60:** 0.095.
- **0.90:** 0.100.
- **1.00 (mejor):** 0.110.

**Monótono creciente.** Cuanta más recombinación, mejor. Confirma el rol del cruce como primario en alta dimensionalidad (7.946 dims). Elegimos 0.90 para mantener 10% de padres puros como fuente de diversidad.

---

## Slide 9c — Operador de cruce (1 min)

Comparamos:
- **uniform (elegido):** F1 test 0.065, fitness 0.100.
- **one_point:** 0.060, 0.086.
- **arithmetic:** 0.064, 0.086.

**Uniforme gana en alta dimensión.** Con 7.946 dims, one_point rompe building blocks dispersos y arithmetic queda "entre los padres" → poca exploración.

**Configuración final:** rank(s=1.7), p_mut=0.04, p_cx=0.9, cruce=uniforme, elitismo=2, inmigrantes+sigma adaptativo.

---

## Slide 10 — NSGA-II (Módulo 7) (1.5 min)

**Precisión y recall están en conflicto** — un solo F1 esconde el compromiso.

**Qué cambia respecto al AG normal:**
- Antes (mono-objetivo): un solo número (F1) por individuo.
- Ahora (NSGA-II): cada individuo tiene **2 fitness** (precisión y recall).
- Selección por **non-dominated sort** (Módulo 7).
- Diversidad mantenida por **crowding distance** (Módulo 7).

**Resultado:** ~21 soluciones **no dominadas** — el frente de Pareto. El usuario elige el punto según prefiera precisión o cobertura. Alta precisión = alertas confiables; alto recall = no perderse efectos.

Demuestra que la evolución también explora trade-offs, no solo optimiza un objetivo.

---

## Slide 11 — XAI (Módulo 9) (1.5 min)

**Como la red es chica, sus pesos son inspeccionables** — derivamos importancia por feature.

**Cómo se calcula:** propagamos las magnitudes: **| W1 | · | W2 |**. Cada feature (columna de entrada) recibe un score según cuánto pesa en las decisiones finales.

**Qué pesa más:** fármacos concretos e indicaciones aparecen primero — el modelo aprendió relaciones interpretables. Ejemplos: para "Nausea", fármacos gastrointestinales y edad avanzada. Coincide con conocimiento médico → validación cualitativa.

**Mensaje central:** el individuo evolucionado **NO es una caja negra**. La representación elegida (real, con localidad) lo hace interpretable.

---

## Slide 12 — Resultados finales (1.5 min)

**4 KPIs principales** (los 4 boxes arriba):
- **F1-macro test: 0.106** — modelo evolucionado sobre 8.762 casos no vistos.
- **Precision macro: 0.111** — del individuo ganador.
- **F1 vs SIDER 4.1: 0.31** (0.306) — validación externa contra base curada; precisión vs SIDER = 0.50.
- **Tiempo: 7.15 min** en CPU, 300 gens.

**Gráfico SIDER (abajo):** el modelo recupera efectos que farmacólogos documentaron independientemente. **El AG aprendió relaciones reales**, no solo patrones estadísticos de FAERS.

**Perspectiva:** es un problema muy difícil (98 etiquetas, 2% densidad, texto ruidoso). El AG evolutivo pelea de igual a igual con backprop, con la ventaja de no requerir gradientes y ser interpretable.

---

## Slide 13 — Mapeo a módulos (30 seg)

**7 de 9 módulos cubiertos:**
- **Módulo 2:** SGA + elitismo.
- **Módulo 3:** genotipo real con localidad.
- **Módulo 4:** rank lineal (s=1.7) con presión constante, F1-macro con umbral por etiqueta.
- **Módulo 5:** σ determinístico + inmigrantes adaptativos + sweep de hiperparámetros.
- **Módulo 6:** métricas independientes (diversidad genotípica, presión teórica).
- **Módulo 7:** NSGA-II con crowding.
- **Módulo 9:** XAI por magnitudes de pesos.

**Cierre:** demostramos que un AG puede entrenar un clasificador real multi-label sin gradiente, validado contra base externa, con resultados reproducibles.

"Gracias."

---

**Total estimado:** ~19-20 minutos. Ajustá bajando las slides 9a/b/c a 30 seg cada una si te apura el tiempo.
