"""
Experimentos de sensibilidad del AG: como impactan los hiperparametros
(probabilidad de mutacion, probabilidad de cruce, operador de cruce) sobre el
fitness final, el F1 de test y la velocidad de convergencia.

NO modifica models/ga_model/ — solo reporta y grafica.

Salida: outputs/experiments_summary.csv + outputs/figures/exp_mutation.png,
exp_crossover_prob.png, exp_crossover_kind.png
"""

import time
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (RANDOM_SEED, OUTPUTS_DIR, TFIDF_MAX_FEATURES, TFIDF_MIN_DF,
                    GA_VAL_FRACTION, GA_HIDDEN, GA_INIT_SCALE, GA_TOURNAMENT_K,
                    GA_ELITISM)
from data_split import load_split
from ga_features import PatientFeaturizer
from ga_model import GAClassifier
from train_ga import (build_Y, best_thresholds_and_f1, init_population,
                      tournament_select, crossover, mutate, FIT_GRID,
                      PERLABEL_GRID)
import train_ga as TG   # para sobreescribir su rng

# Presupuesto reducido para que cada experimento sea rapido
EXP_POP = 40
EXP_GEN = 50
EXP_SUB = 2500


def run_ga(pmut, pcross, ckind, sigma0=0.20, sigma_end=0.02):
    """Corre un AG con la config dada y devuelve metricas finales + historia."""
    # Reset determinista del rng global usado por init_population / mutate / ...
    TG.rng = np.random.default_rng(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    df, label_names, train_idx, test_idx = load_split()
    val_n = int(len(train_idx) * GA_VAL_FRACTION)
    val_idx, tr_idx = train_idx[:val_n], train_idx[val_n:]

    feat = PatientFeaturizer(TFIDF_MAX_FEATURES, TFIDF_MIN_DF).fit(df.iloc[tr_idx])
    Xtr_full = feat.transform(df.iloc[tr_idx])
    Xval = feat.transform(df.iloc[val_idx])
    Xte = feat.transform(df.iloc[test_idx])
    Ytr = build_Y(df.iloc[tr_idx], label_names)
    Yval = build_Y(df.iloc[val_idx], label_names)
    Yte = build_Y(df.iloc[test_idx], label_names)

    sub = rng.choice(len(Xtr_full), size=EXP_SUB, replace=False)
    Xfit, Yfit = Xtr_full[sub], Ytr[sub]

    net = GAClassifier(feat.dim, GA_HIDDEN, len(label_names))

    def fit_of(g):
        net.set_genome(g)
        _, f = best_thresholds_and_f1(net.forward(Xfit), Yfit, FIT_GRID)
        return float(f.mean())

    pop = init_population(net, EXP_POP, GA_INIT_SCALE)
    fit = np.array([fit_of(ind) for ind in pop])
    best_genome = pop[np.argmax(fit)].copy()
    best_fit = float(fit.max())
    history_best = [best_fit]
    history_mean = [float(fit.mean())]

    for gen in range(1, EXP_GEN + 1):
        frac = (gen - 1) / max(1, EXP_GEN - 1)
        sigma = sigma0 + frac * (sigma_end - sigma0)

        order = np.argsort(fit)[::-1]
        new_pop = [pop[order[e]].copy() for e in range(GA_ELITISM)]
        while len(new_pop) < EXP_POP:
            p1 = pop[tournament_select(fit, GA_TOURNAMENT_K)]
            p2 = pop[tournament_select(fit, GA_TOURNAMENT_K)]
            if TG.rng.random() < pcross:
                c1, c2 = crossover(p1, p2, ckind)
            else:
                c1, c2 = p1.copy(), p2.copy()
            new_pop.append(mutate(c1, pmut, sigma))
            if len(new_pop) < EXP_POP:
                new_pop.append(mutate(c2, pmut, sigma))
        pop = np.array(new_pop)
        fit = np.array([fit_of(ind) for ind in pop])
        if float(fit.max()) > best_fit:
            best_fit = float(fit.max())
            best_genome = pop[np.argmax(fit)].copy()
        history_best.append(best_fit)
        history_mean.append(float(fit.mean()))

    # Test final con umbrales calibrados en validacion
    net.set_genome(best_genome)
    thresholds, _ = best_thresholds_and_f1(net.forward(Xval), Yval, PERLABEL_GRID)
    te_probs = net.forward(Xte)
    te_pred = (te_probs >= thresholds).astype(int)
    Yte_i = Yte.astype(int)
    tp = (te_pred & Yte_i).sum(0)
    pp = te_pred.sum(0); ap = Yte_i.sum(0)
    prec = np.divide(tp, pp, out=np.zeros_like(tp, dtype=float), where=pp > 0)
    rec  = np.divide(tp, ap, out=np.zeros_like(tp, dtype=float), where=ap > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp, dtype=float),
                   where=(prec + rec) > 0)
    test_f1_macro = float(f1.mean())

    return {
        "best_fit": best_fit,
        "test_f1_macro": test_f1_macro,
        "history_best": history_best,
        "history_mean": history_mean,
    }


def main():
    t0 = time.time()
    results = []
    print("=" * 70)
    print(f"  EXPERIMENTOS DE SENSIBILIDAD DEL AG  "
          f"(pop={EXP_POP}, gen={EXP_GEN}, subsample={EXP_SUB})")
    print("=" * 70)

    # ── Sweep 1: probabilidad de MUTACION ─────────────────────────────────────
    print("\n[1/3] Probabilidad de MUTACION (cruce=uniform pc=0.9 fijo)")
    mut_runs = {}
    for pmut in [0.01, 0.03, 0.05, 0.10, 0.20, 0.30]:
        tt = time.time()
        r = run_ga(pmut=pmut, pcross=0.9, ckind="uniform")
        dt = time.time() - tt
        mut_runs[pmut] = r
        results.append({"sweep": "mutation", "param": "p_mutation",
                        "value": pmut, "best_fit": r["best_fit"],
                        "test_f1": r["test_f1_macro"], "seconds": round(dt, 1)})
        print(f"  pmut={pmut:.2f}  ->  fit={r['best_fit']:.4f}  "
              f"test_F1={r['test_f1_macro']:.4f}  ({dt:.0f}s)")

    # ── Sweep 2: probabilidad de CRUCE ────────────────────────────────────────
    print("\n[2/3] Probabilidad de CRUCE (mut=0.05, uniform)")
    cross_p_runs = {}
    for pc in [0.3, 0.6, 0.9, 1.0]:
        tt = time.time()
        r = run_ga(pmut=0.05, pcross=pc, ckind="uniform")
        dt = time.time() - tt
        cross_p_runs[pc] = r
        results.append({"sweep": "crossover_prob", "param": "p_crossover",
                        "value": pc, "best_fit": r["best_fit"],
                        "test_f1": r["test_f1_macro"], "seconds": round(dt, 1)})
        print(f"  pc={pc:.2f}  ->  fit={r['best_fit']:.4f}  "
              f"test_F1={r['test_f1_macro']:.4f}  ({dt:.0f}s)")

    # ── Sweep 3: OPERADOR de cruce ────────────────────────────────────────────
    print("\n[3/3] OPERADOR de cruce (mut=0.05, pc=0.9)")
    kind_runs = {}
    for kind in ["uniform", "one_point", "arithmetic"]:
        tt = time.time()
        r = run_ga(pmut=0.05, pcross=0.9, ckind=kind)
        dt = time.time() - tt
        kind_runs[kind] = r
        results.append({"sweep": "crossover_kind", "param": "crossover",
                        "value": kind, "best_fit": r["best_fit"],
                        "test_f1": r["test_f1_macro"], "seconds": round(dt, 1)})
        print(f"  {kind:10s} ->  fit={r['best_fit']:.4f}  "
              f"test_F1={r['test_f1_macro']:.4f}  ({dt:.0f}s)")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUTS_DIR / "experiments_summary.csv", index=False)

    figdir = OUTPUTS_DIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # Figura 1: efecto de pmut (best fitness vs generacion)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for pmut, r in mut_runs.items():
        axes[0].plot(r["history_best"], label=f"pmut={pmut:.2f}")
    axes[0].set_title("Convergencia segun probabilidad de mutacion")
    axes[0].set_xlabel("Generacion"); axes[0].set_ylabel("Mejor fitness")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    pmuts = list(mut_runs.keys())
    axes[1].bar([f"{p:.2f}" for p in pmuts],
                [mut_runs[p]["test_f1_macro"] for p in pmuts],
                color="#e74c3c")
    axes[1].set_title("F1-macro de test segun probabilidad de mutacion")
    axes[1].set_xlabel("p_mutation"); axes[1].set_ylabel("F1-macro (test)")
    axes[1].grid(alpha=0.3, axis="y")
    for i, p in enumerate(pmuts):
        axes[1].text(i, mut_runs[p]["test_f1_macro"] + 0.001,
                     f"{mut_runs[p]['test_f1_macro']:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(figdir / "exp_mutation.png", dpi=150); plt.close(fig)

    # Figura 2: efecto de pc
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for pc, r in cross_p_runs.items():
        axes[0].plot(r["history_best"], label=f"pc={pc:.2f}")
    axes[0].set_title("Convergencia segun probabilidad de cruce")
    axes[0].set_xlabel("Generacion"); axes[0].set_ylabel("Mejor fitness")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    pcs = list(cross_p_runs.keys())
    axes[1].bar([f"{p:.2f}" for p in pcs],
                [cross_p_runs[p]["test_f1_macro"] for p in pcs],
                color="#3498db")
    axes[1].set_title("F1-macro de test segun probabilidad de cruce")
    axes[1].set_xlabel("p_crossover"); axes[1].set_ylabel("F1-macro (test)")
    axes[1].grid(alpha=0.3, axis="y")
    for i, p in enumerate(pcs):
        axes[1].text(i, cross_p_runs[p]["test_f1_macro"] + 0.001,
                     f"{cross_p_runs[p]['test_f1_macro']:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(figdir / "exp_crossover_prob.png", dpi=150); plt.close(fig)

    # Figura 3: efecto del operador
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for k, r in kind_runs.items():
        axes[0].plot(r["history_best"], label=k)
    axes[0].set_title("Convergencia segun operador de cruce")
    axes[0].set_xlabel("Generacion"); axes[0].set_ylabel("Mejor fitness")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    kinds = list(kind_runs.keys())
    axes[1].bar(kinds, [kind_runs[k]["test_f1_macro"] for k in kinds],
                color=["#2ecc71", "#f39c12", "#9b59b6"])
    axes[1].set_title("F1-macro de test segun operador de cruce")
    axes[1].set_ylabel("F1-macro (test)"); axes[1].grid(alpha=0.3, axis="y")
    for i, k in enumerate(kinds):
        axes[1].text(i, kind_runs[k]["test_f1_macro"] + 0.001,
                     f"{kind_runs[k]['test_f1_macro']:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(figdir / "exp_crossover_kind.png", dpi=150); plt.close(fig)

    print(f"\nTiempo total: {time.time() - t0:.0f}s")
    print(f"Resumen:  {OUTPUTS_DIR / 'experiments_summary.csv'}")
    print(f"Figuras:  exp_mutation.png  exp_crossover_prob.png  exp_crossover_kind.png")


if __name__ == "__main__":
    main()
