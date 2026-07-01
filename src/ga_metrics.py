"""
Metricas del AG INDEPENDIENTES entre si (Modulo 6).

Antes mediamos:
  - diversidad = std(fitness)           <- consecuencia del fitness
  - presion   = max(fitness)/mean(fitness) <- tambien consecuencia
Ambas colapsaban juntas cuando la poblacion converge -> curvas planas y correlacionadas.

Ahora:
  - diversidad = std promedio de los GENOTIPOS (dimension por dimension)
                 -> mide dispersion del cromosoma en el espacio, NO del fitness
  - presion   = valor TEORICO del operador de seleccion
                 -> depende SOLO del operador, no de la poblacion
Ambas son independientes: cuentan cosas distintas y pueden moverse por separado.
"""

import numpy as np


def genotypic_diversity(pop: np.ndarray) -> float:
    """Desvio estandar promedio dimension a dimension.

    Con genotipo real de N pesos y POP individuos:
      D = (1/N) * sum_i std(pop[:, i])
    Interpretacion:
      - Alto  -> la poblacion cubre un volumen amplio del espacio de pesos
      - Bajo  -> los individuos son casi iguales (colapso genotipico)
    Es independiente del fitness: puede haber diversidad genotipica alta con
    fitness plano (paisaje neutro) o baja con fitness diverso (varios optimos).
    """
    return float(pop.std(axis=0).mean())


def theoretical_selection_pressure(operator: str, **params) -> float:
    """Presion selectiva teorica del operador (NO depende de la poblacion).

    Devuelve la "tasa de reproduccion esperada del mejor individuo" segun la
    formula del operador. Como es analitica, la curva a lo largo de las
    generaciones es horizontal salvo que cambies el operador o sus parametros.

    Formulas estandar (Blickle & Thiele 1996, cubierto en Modulo 4):
      - torneo de tamano k:   presion ~ log(k) + 1     (~2.1 para k=3)
      - rank lineal con s:    presion = s               (1 <= s <= 2)
      - proporcional (Roulette): presion depende de la varianza del fitness
        (no se puede dar valor teorico -> devolvemos NaN)
    """
    if operator == "tournament":
        k = params.get("k", 3)
        return float(np.log(k) + 1.0)
    if operator == "rank":
        s = params.get("s", 1.5)
        return float(s)
    return float("nan")


def rank_selection_probs(fitness: np.ndarray, s: float = 1.5) -> np.ndarray:
    """Probabilidades de seleccion por RANK lineal (Modulo 4).

    Cada individuo recibe una probabilidad segun su posicion en el ranking,
    no segun su fitness absoluto. Ventaja: presion selectiva CONSTANTE e
    independiente de la escala del fitness (no se dispara ni se apaga aunque
    el fitness varie mucho o poco).

      p(i) = (1/N) * (2 - s + 2*(s - 1)*rank(i) / (N - 1))

    con rank(i) = 0 para el peor, N-1 para el mejor.  s en [1.0, 2.0]:
      - s = 1.0 -> sin presion (todas las prob. iguales, deriva aleatoria)
      - s = 2.0 -> presion maxima (el peor recibe prob=0)
    """
    n = len(fitness)
    order = np.argsort(fitness)                     # peor -> mejor
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n)                     # rank del individuo i
    p = (2.0 - s + 2.0 * (s - 1.0) * ranks / max(1, n - 1)) / n
    return p / p.sum()                              # normalizacion por seguridad
