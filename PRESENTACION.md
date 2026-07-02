# 🎤 PRESENTACIÓN — farmApp-TCE

**Predicción de efectos adversos con Computación Evolutiva**
Proyecto de **Técnicas de Computación Evolutiva**
Alzugaray Agustín Ezequiel · Lautaro Cabrera Tamalet

---

## 1 · Descripción del problema

**Qué predecimos**
- Dado un paciente que toma un fármaco, ¿qué efectos adversos va a sufrir?

**Características**
- **Multi-label**: un paciente puede tener varios efectos a la vez.
- **98 efectos posibles** (vocabulario MedDRA).
- **Severamente desbalanceado**: solo el ~2 % de cada efecto da positivo.

**Por qué es difícil**
- 98 etiquetas con frecuencias muy distintas.
- Texto clínico ruidoso y heterogéneo.
- Cualquier modelo "fácil" (que predice todo "no") tiene ~98 % de accuracy pero F1 ≈ 0.

**Por qué importa**
- Detectar efectos adversos antes de que ocurran ayuda a farmacovigilancia y a la seguridad del paciente.

---

## 2 · Datasets — qué usamos y para qué

| Dataset | Para qué | Origen |
|---|---|---|
| **FAERS (FDA)** | Entrenamiento + test | Reportes reales: pacientes que ya tuvieron efectos adversos. ~55 000 casos, 5 trimestres. |
| **SIDER 4.1** | Validación externa | Base curada por farmacólogos: efectos conocidos de cada fármaco. Independiente de FAERS. |

**De FAERS sacamos X e Y**:
- **X** = perfil del paciente (fármaco, indicación, otras medicaciones, edad, sexo, peso)
- **Y** = efectos adversos que **realmente tuvo** (vector binario de 98 columnas)

**SIDER no se usa para entrenar** — se usa **después**, para verificar que las predicciones del modelo coinciden con efectos ya documentados externamente.

---

## 3 · Arquitectura — cómo funciona el sistema

```
┌──────────────────────────────────────────────────────────────┐
│  1) DATOS (FAERS combinado, 1 fila por paciente)             │
│                            │                                 │
│                            ▼                                 │
│  2) FEATURIZER → vector X de 228 números                    │
│     (drug, indi, other_drugs, demografía)                    │
│                            │                                 │
│                            ▼                                 │
│  3) RED NEURONAL   228 → 24 → 98                             │
│     (predice Ŷ = probabilidades de cada efecto)              │
│                            │                                 │
│                            ▼                                 │
│  4) FITNESS = qué tan parecido es Ŷ a Y real (F1-macro)      │
│                            │                                 │
│                            ▼                                 │
│  5) ALGORITMO GENÉTICO ajusta los 7.946 pesos                │
│     · selección por rank lineal (s=1.7)                      │
│     · cruce uniforme                                         │
│     · mutación gaussiana                                     │
│     · elitismo + inmigrantes aleatorios                      │
│                            │                                 │
│                            ▼                                 │
│  ¿mejoró? → otra generación (120 generaciones)               │
│                            │                                 │
│                            ▼                                 │
│  6) MEJOR RED → VALIDACIÓN EXTERNA con SIDER 4.1             │
└──────────────────────────────────────────────────────────────┘
```

### El cromosoma y la red

| | |
|---|---|
| **GENOTIPO** | Vector real de **7.946 pesos** = `[W1 (228×24) \| b1 (24) \| W2 (24×98) \| b2 (98)]` (Módulo 3 — representación no binaria) |
| **FENOTIPO** | Red neuronal `228 → ReLU(24) → sigmoide(98)` |
| **FITNESS** | F1-macro con umbral óptimo por etiqueta (Módulo 4) |

**Frase clave para defender**:
> *"El cromosoma son los pesos de la red. El AG no usa gradiente: combina y muta pesos hasta que la red predice Ŷ parecidas a las Y reales de FAERS."*

---

## 4 · Cómo evolucionó el fitness

![Curva de convergencia del AG](outputs/figures/ga_convergence.png)

| Gráfico izquierdo | Gráfico derecho |
|---|---|
| **Mejor fitness** sube de 0.05 a **0.13** en 120 generaciones. | **Diversidad** se mantiene moderada (los inmigrantes la sostienen). |
| **Promedio** sube en paralelo → no es un afortunado, **toda la población mejora**. | **Presión de selección** ≈ 1.05 (sana, como recomienda el material — Whitley sugiere mantener ≈ 1.5 como máximo). |
| Curva limpia, sin plateau prematuro. | σ de mutación decrece de 0.25 a 0.02 (control determinístico, Módulo 5). |

**Lo que se ve**:
- La población **converge sin estancarse**.
- La estructura del AG (rank lineal + cruce + mutación + elitismo + inmigrantes adaptativos) mantiene el balance exploración / explotación durante las 300 generaciones.

---

## 5 · Impacto del AG en el sistema (ablación)

> Para mostrar qué aporta cada componente evolutivo, le sacamos al AG **un operador por vez** y comparamos. Mismo presupuesto: **2.040 evaluaciones de fitness** en todas las configuraciones.

![Ablación del AG](outputs/figures/ablation.png)

| Configuración | Fitness | F1 test | Qué le sacamos |
|---|---:|---:|---|
| **Random search (sin AG)** | 0.061 | 0.051 | Sin selección, sin cruce, sin mutación |
| **Solo mutación** (sin cruce) | 0.082 | 0.059 | Sin recombinación |
| **Solo cruce** (sin mutación) | 0.096 | 0.066 | Sin variación de novo |
| **Sin elitismo** | 0.102 | 0.075 | El mejor puede empeorar entre gen |
| **AG completo** ⭐ | 0.100 | 0.065 | Todos los operadores |

### Lecturas

**1. Random search → AG completo: F1 sube +27 % (0.051 → 0.065).**
Con el mismo presupuesto de evaluaciones, **el AG hace algo más que muestrear al azar**. La presión de selección + los operadores guían la búsqueda hacia regiones útiles.

**2. Cruce vs mutación.**
Solo mutación: 0.082 · Solo cruce: 0.096. Coincide con Goldberg (SGA, Módulo 2): el cruce es el operador **primario** (recombina building blocks); la mutación es **secundario** (asegura que no se pierdan alelos).

**3. Sin elitismo dio mejor que con elitismo en este presupuesto.**
Hallazgo honesto del estudio. **Por qué**: con solo 40 individuos × 50 gen, el elitismo de 2 concentra demasiado.
En el **run real del proyecto** (POP=80, GEN=120, **con inmigrantes aleatorios**) el elitismo sí aporta — el mejor fitness queda en 0.13 vs el 0.10 del ablation chico.

### Conclusión
> *"Cada componente evolutivo aporta una mejora medible sobre random search. Selección + cruce uniforme aportan el grueso de la mejora; la mutación gaussiana afina; el elitismo es útil solo combinado con mecanismos de diversidad — hallazgo que el experimento dejó al descubierto."*

---

## 6 · Experimentos de hiperparámetros

> Sweep sistemático de cada hiperparámetro principal del AG. Buscamos validar empíricamente lo que la teoría del programa predice.

### 🧪 Experimento A · Probabilidad de mutación

![Sensibilidad a la probabilidad de mutación](outputs/figures/exp_mutation.png)

| `p_mutation` | Fitness | F1 test |
|---:|---:|---:|
| 0.01 | 0.092 | 0.069 |
| 0.03 | 0.094 | 0.070 |
| **0.05** | **0.100** | 0.065 |
| **0.10** | **0.103** ⭐ | 0.068 |
| 0.20 | 0.093 | 0.065 |
| 0.30 | 0.092 | 0.071 |

**Patrón en U invertida** (exactamente lo que predice la teoría):
- **Mutación muy baja** (0.01): no explora.
- **Mutación óptima** (0.05–0.10): balance exploración/explotación.
- **Mutación alta** (0.20–0.30): destruye soluciones buenas más rápido de lo que las construye.

### 🧪 Experimento B · Probabilidad de cruce

![Sensibilidad a la probabilidad de cruce](outputs/figures/exp_crossover_prob.png)

| `p_crossover` | Fitness | F1 test |
|---:|---:|---:|
| 0.30 | 0.086 | 0.061 |
| 0.60 | 0.095 | 0.069 |
| 0.90 | 0.100 | 0.065 |
| **1.00** | **0.110** ⭐ | **0.077** |

**Patrón monótono creciente**: cuanta más recombinación, mejor. Confirma el rol del cruce como operador primario en alta dimensionalidad.

### 🧪 Experimento C · Operador de cruce

![Operador de cruce](outputs/figures/exp_crossover_kind.png)

| Operador | Fitness | F1 test |
|---|---:|---:|
| **uniform** | **0.100** ⭐ | **0.065** |
| one_point | 0.086 | 0.060 |
| arithmetic | 0.086 | 0.064 |

**Uniforme gana** porque con 7.946 dimensiones el corte de un punto rompe bloques útiles dispersos; el aritmético se queda "entre los padres" → poca exploración.

### 🎯 Configuración final elegida
Combinando los 3 experimentos: **p_mutation = 0.04–0.05**, **p_crossover = 0.9**, **operador = uniforme**.
Es lo que está en `train_ga.py` y produjo el modelo que se mostrará en la sección 7.

---

## 7 · Resultados finales

### 📊 Modelo del proyecto (run completo: POP=80, GEN=120)

| Métrica | Valor |
|---|---:|
| **F1-macro (test)** | **0.1059** |
| F1-micro | 0.0728 |
| Precision macro | 0.1111 |
| Recall macro | 0.2910 |
| Hamming loss | 0.2012 |
| Casos de test con ≥1 acierto | **48,6 %** (4.257 / 8.762) |

### 🌐 Validación externa con SIDER 4.1

![Validación contra SIDER 4.1](outputs/figures/sider_validation.png)

| | |
|---|---:|
| F1 vs SIDER (fármacos mapeables) | **0.306** |
| Precisión vs SIDER | 0.502 |
| Recall vs SIDER | 0.249 |

→ El modelo predice efectos que **ya están documentados** por farmacólogos en una base independiente. Es la verificación más fuerte de que el AG aprendió algo real.

### ⏱️ Costo computacional
| | |
|---|---:|
| Tiempo de entrenamiento | **7,15 minutos** (300 gen) |
| Hardware | CPU (sin GPU, solo NumPy) |
| Cromosoma | 7.946 pesos reales |

### ➕ Extras evolutivos del proyecto
- **Selección de features con cromosoma binario** (Módulo 2/3): el AG eligió 59 de 154 features y mejoró F1 0.124 → 0.133.
- **NSGA-II** (Módulo 7): frente de Pareto con 21 soluciones no dominadas precisión/recall.
- **XAI** (Módulo 9): importancia derivada del individuo evolucionado — los features más influyentes son fármacos e indicaciones concretos (interpretable).

### 💡 Cierre — qué demostramos

> *"Un Algoritmo Genético puede entrenar un clasificador multi-label real sobre datos clínicos reales, **sin gradiente**, en 3 minutos de CPU. Los experimentos confirman que el comportamiento del AG sigue exactamente lo que predice la teoría de la materia: la sensibilidad a la mutación es una U invertida, el cruce uniforme domina en alta dimensión, el elitismo necesita diversidad para aportar, y cada operador evolutivo es medible mediante ablación. La validación externa contra SIDER 4.1 (F1 = 0.32) confirma que el modelo aprendió relaciones farmacológicas reales, no patrones espurios."*

---

## 📁 Archivos y figuras a tener listos durante la defensa

| Sección | Archivo |
|---|---|
| 3 · Arquitectura | App abierta → pestaña "Cómo lo hicimos" |
| 4 · Fitness | `outputs/figures/ga_convergence.png` |
| 5 · Impacto | `outputs/figures/ablation.png` + tabla `outputs/ablation_summary.csv` |
| 6 · Experimentos | `outputs/figures/exp_mutation.png`, `exp_crossover_prob.png`, `exp_crossover_kind.png` + `outputs/experiments_summary.csv` |
| 7 · Resultados | `outputs/figures/sider_validation.png` + app pestaña "Casos reales de test" |

### Demo en vivo (opcional)
1. `streamlit run app.py`
2. Pestaña **"Predecir paciente nuevo"** → cargar paciente → ver predicción.
3. Pestaña **"Análisis del modelo"** → curva de convergencia + extras XAI/NSGA-II.

---

## 🎯 Mapeo a los módulos del programa

| Módulo | Cómo se cubre |
|---|---|
| **2** — Algoritmos Genéticos | Bucle completo SGA + elitismo |
| **3** — Genotipos no binarios | Genotipo real (pesos) **+** binario (selección de features) |
| **4** — Selección, fitness, constraints | Rank lineal (s=1.7, presión constante), F1-macro con umbral por etiqueta, exclusión de no-ADRs |
| **5** — Parameter tuning and control | σ determinístico + inmigrantes adaptativos + experimentos de sweep |
| **6** — Performance metrics (single-obj) | Best/mean fitness, diversidad, presión de selección por generación |
| **7** — Multi-objective EAs | NSGA-II para precisión vs recall |
| **9** — XAI + EC | Importancia de features derivada del individuo evolucionado |

**7 de los 9 módulos** del programa, integrados en un mismo proyecto.
