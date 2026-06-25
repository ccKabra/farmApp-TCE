"""
Extra — EA MULTI-OBJETIVO con NSGA-II  (Modulo 7: EAs multi-objetivo).

En este problema la PRECISION y el RECALL estan en CONFLICTO: predecir mas
efectos sube el recall pero baja la precision, y viceversa. En vez de colapsar
ambos en un solo F1, aca evolucionamos el **frente de Pareto** precision-vs-recall
con NSGA-II (Deb et al., 2002):

  * fast non-dominated sorting  -> ranking por dominancia de Pareto
  * crowding distance           -> diversidad a lo largo del frente
  * crowded-comparison operator -> seleccion por (rango, luego crowding)
  * reemplazo elitista (padres + hijos, se queda con los N mejores)

Cada individuo es el MISMO cromosoma real de pesos que en train_ga.py (red
154->24->98). Los dos objetivos se miden a umbral global fijo (0.5), asi distintos
vectores de pesos dan distintos balances precision/recall.

Salida: outputs/nsga2_pareto.csv + outputs/figures/nsga2_pareto_front.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (OUTPUTS_DIR, GA_HIDDEN, GA_TOURNAMENT_K, GA_P_CROSSOVER,
                    GA_CROSSOVER, GA_P_MUTATION, GA_MUT_SIGMA, GA_MUT_SIGMA_END,
                    GA_INIT_SCALE, GA_FIT_SUBSAMPLE, TFIDF_MAX_FEATURES, TFIDF_MIN_DF)
from data_split import load_split
from ga_features import PatientFeaturizer
from ga_model import GAClassifier
from train_ga import build_Y, init_population, crossover, mutate, rng

# Parametros propios del NSGA-II (mas chicos: 2 objetivos, frente)
NSGA_POP = 60
NSGA_GEN = 40
THRESH = 0.5   # umbral global fijo para medir precision/recall


def objectives(net, genome, X, Yb, t=THRESH):
    """Devuelve (precision_macro, recall_macro) — ambos a MAXIMIZAR."""
    net.set_genome(genome)
    pred = net.forward(X) >= t
    tp = np.logical_and(pred, Yb).sum(0).astype(np.float64)
    pp = pred.sum(0).astype(np.float64)
    ap = Yb.sum(0).astype(np.float64)
    prec = np.divide(tp, pp, out=np.zeros_like(tp), where=pp > 0).mean()
    rec = np.divide(tp, ap, out=np.zeros_like(tp), where=ap > 0).mean()
    return float(prec), float(rec)


def dominates(p, q):
    """p domina a q (maximizacion): mejor o igual en todo y estrictamente mejor en algo."""
    return (p[0] >= q[0] and p[1] >= q[1]) and (p[0] > q[0] or p[1] > q[1])


def fast_nondominated_sort(objs):
    n = len(objs)
    S = [[] for _ in range(n)]
    dom_count = [0] * n
    rank = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    rank[q] = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts, rank


def crowding_distance(objs, front):
    dist = {i: 0.0 for i in front}
    l = len(front)
    if l == 0:
        return dist
    for m in range(2):
        order = sorted(front, key=lambda i: objs[i][m])
        dist[order[0]] = dist[order[-1]] = float("inf")
        lo, hi = objs[order[0]][m], objs[order[-1]][m]
        span = hi - lo
        if span <= 0:
            continue
        for k in range(1, l - 1):
            dist[order[k]] += (objs[order[k + 1]][m] - objs[order[k - 1]][m]) / span
    return dist


def crowded_tournament(rank, dist, k=GA_TOURNAMENT_K):
    """Gana el de menor rango; a igual rango, el de mayor crowding distance."""
    asp = rng.integers(0, len(rank), size=k)
    best = asp[0]
    for c in asp[1:]:
        if rank[c] < rank[best] or (rank[c] == rank[best] and dist[c] > dist[best]):
            best = c
    return best


def main():
    print("=" * 70)
    print("  NSGA-II — frente de Pareto PRECISION vs RECALL  (Modulo 7)")
    print("=" * 70)

    df, label_names, train_idx, test_idx = load_split()
    val_n = int(len(train_idx) * 0.10)
    tr_idx = train_idx[val_n:]
    df_tr = df.iloc[tr_idx]

    feat = PatientFeaturizer(TFIDF_MAX_FEATURES, TFIDF_MIN_DF).fit(df_tr)
    Xtr = feat.transform(df_tr)
    Ytr = build_Y(df_tr, label_names)
    if GA_FIT_SUBSAMPLE and GA_FIT_SUBSAMPLE < len(Xtr):
        sub = rng.choice(len(Xtr), size=GA_FIT_SUBSAMPLE, replace=False)
        Xfit, Yfit = Xtr[sub], Ytr[sub]
    else:
        Xfit, Yfit = Xtr, Ytr

    net = GAClassifier(feat.dim, GA_HIDDEN, len(label_names))
    pop = list(init_population(net, NSGA_POP, GA_INIT_SCALE))
    objs = [objectives(net, ind, Xfit, Yfit) for ind in pop]

    print(f"Poblacion {NSGA_POP} | {NSGA_GEN} generaciones | genoma {net.n_params:,}\n")
    print(f"{'Gen':>4} | {'#frente':>7} | {'maxPrec':>7} | {'maxRec':>7}")
    print("-" * 36)

    for gen in range(1, NSGA_GEN + 1):
        fronts, rank = fast_nondominated_sort(objs)
        dist = {}
        for fr in fronts:
            dist.update(crowding_distance(objs, fr))

        # Generar hijos por seleccion + cruce + mutacion
        frac = (gen - 1) / max(1, NSGA_GEN - 1)
        sigma = GA_MUT_SIGMA + frac * (GA_MUT_SIGMA_END - GA_MUT_SIGMA)
        offspring = []
        while len(offspring) < NSGA_POP:
            p1 = pop[crowded_tournament(rank, dist)]
            p2 = pop[crowded_tournament(rank, dist)]
            if rng.random() < GA_P_CROSSOVER:
                c1, c2 = crossover(p1, p2, GA_CROSSOVER)
            else:
                c1, c2 = p1.copy(), p2.copy()
            offspring.append(mutate(c1, GA_P_MUTATION, sigma))
            if len(offspring) < NSGA_POP:
                offspring.append(mutate(c2, GA_P_MUTATION, sigma))

        # Union padres+hijos, ordenar por frentes y crowding (elitismo NSGA-II)
        combined = pop + offspring
        comb_objs = objs + [objectives(net, ind, Xfit, Yfit) for ind in offspring]
        c_fronts, _ = fast_nondominated_sort(comb_objs)
        new_idx = []
        fi = 0
        while fi < len(c_fronts) and len(new_idx) + len(c_fronts[fi]) <= NSGA_POP:
            new_idx.extend(c_fronts[fi])
            fi += 1
        if len(new_idx) < NSGA_POP and fi < len(c_fronts):
            d = crowding_distance(comb_objs, c_fronts[fi])
            last = sorted(c_fronts[fi], key=lambda x: -d[x])
            new_idx.extend(last[:NSGA_POP - len(new_idx)])

        pop = [combined[i] for i in new_idx]
        objs = [comb_objs[i] for i in new_idx]

        if gen == 1 or gen % 5 == 0 or gen == NSGA_GEN:
            f0 = fast_nondominated_sort(objs)[0][0]
            print(f"{gen:>4} | {len(f0):>7} | "
                  f"{max(o[0] for o in objs):7.4f} | {max(o[1] for o in objs):7.4f}")

    # ── Frente de Pareto final ────────────────────────────────────────────────
    fronts, _ = fast_nondominated_sort(objs)
    pf = sorted([objs[i] for i in fronts[0]], key=lambda o: o[0])
    pf_df = pd.DataFrame(pf, columns=["precision_macro", "recall_macro"])
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pf_df.to_csv(OUTPUTS_DIR / "nsga2_pareto.csv", index=False)

    print(f"\nFrente de Pareto: {len(pf)} soluciones no dominadas")
    print(pf_df.to_string(index=False))

    plt.figure(figsize=(8, 6))
    all_p = [o[0] for o in objs]
    all_r = [o[1] for o in objs]
    plt.scatter(all_r, all_p, c="#bdc3c7", s=25, label="Poblacion final")
    plt.plot([o[1] for o in pf], [o[0] for o in pf], "o-", color="#e74c3c",
             label="Frente de Pareto")
    plt.xlabel("Recall (macro)")
    plt.ylabel("Precision (macro)")
    plt.title("NSGA-II — frente de Pareto Precision vs Recall (modelo AG)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = OUTPUTS_DIR / "figures" / "nsga2_pareto_front.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\nGuardado: {OUTPUTS_DIR / 'nsga2_pareto.csv'}")
    print(f"Figura:   {out}")


if __name__ == "__main__":
    main()
