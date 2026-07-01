"""
Ablacion del AG — muestra el impacto de CADA componente evolutivo aislando uno a la vez.

Toda comparacion es DENTRO de Computacion Evolutiva (no contra modelos de
Mineria de Datos). Mismo presupuesto de evaluaciones de fitness en todos los
casos: 40 individuos x 51 = 2.040 evaluaciones por experimento.

Configuraciones comparadas:
  1. Random search       - sin AG: solo muestrear N cromosomas aleatorios.
  2. Solo mutacion       - AG sin cruce (pc=0).
  3. Solo cruce          - AG sin mutacion (pm=0).
  4. Sin elitismo        - AG completo pero elitism=0 (puede empeorar).
  5. AG COMPLETO         - selecion + cruce + mutacion + elitismo.

Salida: outputs/ablation_summary.csv + outputs/figures/ablation.png
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (RANDOM_SEED, OUTPUTS_DIR, TFIDF_MAX_FEATURES, TFIDF_MIN_DF,
                    GA_VAL_FRACTION, GA_HIDDEN, GA_INIT_SCALE, GA_TOURNAMENT_K)
from data_split import load_split
from ga_features import PatientFeaturizer
from ga_model import GAClassifier
from train_ga import (build_Y, best_thresholds_and_f1, init_population,
                      tournament_select, crossover, mutate, FIT_GRID,
                      PERLABEL_GRID)
import train_ga as TG

POP = 40
GEN = 50
SUB = 2500
N_EVAL_BUDGET = POP * (GEN + 1)     # 2.040 evaluaciones para todos


def _setup():
    """Carga datos, featurizer, etc. — comun a todos los experimentos."""
    TG.rng = np.random.default_rng(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    df, label_names, train_idx, test_idx = load_split()
    val_n = int(len(train_idx) * GA_VAL_FRACTION)
    val_idx, tr_idx = train_idx[:val_n], train_idx[val_n:]

    feat = PatientFeaturizer(TFIDF_MAX_FEATURES, TFIDF_MIN_DF).fit(df.iloc[tr_idx])
    Xtr_full = feat.transform(df.iloc[tr_idx])
    Xval = feat.transform(df.iloc[val_idx])
    Xte  = feat.transform(df.iloc[test_idx])
    Ytr  = build_Y(df.iloc[tr_idx], label_names)
    Yval = build_Y(df.iloc[val_idx], label_names)
    Yte  = build_Y(df.iloc[test_idx], label_names)

    sub = rng.choice(len(Xtr_full), size=SUB, replace=False)
    Xfit, Yfit = Xtr_full[sub], Ytr[sub]

    net = GAClassifier(feat.dim, GA_HIDDEN, len(label_names))
    return net, Xfit, Yfit, Xval, Yval, Xte, Yte


def _test_f1(net, genome, Xval, Yval, Xte, Yte):
    """F1-macro en test calibrando umbrales sobre validacion."""
    net.set_genome(genome)
    thresholds, _ = best_thresholds_and_f1(net.forward(Xval), Yval, PERLABEL_GRID)
    pred = (net.forward(Xte) >= thresholds).astype(int)
    Y = Yte.astype(int)
    tp = (pred & Y).sum(0)
    pp = pred.sum(0); ap = Y.sum(0)
    prec = np.divide(tp, pp, out=np.zeros_like(tp, dtype=float), where=pp > 0)
    rec  = np.divide(tp, ap, out=np.zeros_like(tp, dtype=float), where=ap > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp, dtype=float),
                   where=(prec + rec) > 0)
    return float(f1.mean())


# ── Estrategia 1: Random search (sin AG) ─────────────────────────────────────
def random_search(net, Xfit, Yfit, Xval, Yval, Xte, Yte, n_budget):
    TG.rng = np.random.default_rng(RANDOM_SEED)
    history_best = []
    best_g, best_f = None, -1.0
    for k in range(n_budget):
        g = init_population(net, 1, GA_INIT_SCALE)[0]
        net.set_genome(g)
        _, f1 = best_thresholds_and_f1(net.forward(Xfit), Yfit, FIT_GRID)
        f = float(f1.mean())
        if f > best_f:
            best_f, best_g = f, g.copy()
        if (k + 1) % POP == 0:
            history_best.append(best_f)
    return {"history": history_best, "best_fit": best_f,
            "test_f1": _test_f1(net, best_g, Xval, Yval, Xte, Yte)}


# ── Bucle generico del AG con flags para activar/desactivar componentes ──────
def ga_run(net, Xfit, Yfit, Xval, Yval, Xte, Yte,
           pcross=0.9, pmut=0.05, elitism=2, sigma0=0.20, sigma_end=0.02):
    TG.rng = np.random.default_rng(RANDOM_SEED)
    pop = init_population(net, POP, GA_INIT_SCALE)

    def fit_of(g):
        net.set_genome(g)
        _, f = best_thresholds_and_f1(net.forward(Xfit), Yfit, FIT_GRID)
        return float(f.mean())

    fitness = np.array([fit_of(ind) for ind in pop])
    best_g = pop[np.argmax(fitness)].copy()
    best_f = float(fitness.max())
    history = [best_f]

    for gen in range(1, GEN + 1):
        frac = (gen - 1) / max(1, GEN - 1)
        sigma = sigma0 + frac * (sigma_end - sigma0)

        order = np.argsort(fitness)[::-1]
        new_pop = [pop[order[e]].copy() for e in range(elitism)]
        while len(new_pop) < POP:
            p1 = pop[tournament_select(fitness, GA_TOURNAMENT_K)]
            p2 = pop[tournament_select(fitness, GA_TOURNAMENT_K)]
            if pcross > 0 and TG.rng.random() < pcross:
                c1, c2 = crossover(p1, p2, "uniform")
            else:
                c1, c2 = p1.copy(), p2.copy()
            new_pop.append(mutate(c1, pmut, sigma) if pmut > 0 else c1)
            if len(new_pop) < POP:
                new_pop.append(mutate(c2, pmut, sigma) if pmut > 0 else c2)
        pop = np.array(new_pop)
        fitness = np.array([fit_of(ind) for ind in pop])
        if float(fitness.max()) > best_f:
            best_f = float(fitness.max())
            best_g = pop[np.argmax(fitness)].copy()
        history.append(best_f)

    return {"history": history, "best_fit": best_f,
            "test_f1": _test_f1(net, best_g, Xval, Yval, Xte, Yte)}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  ABLACION DEL AG — impacto de cada componente evolutivo")
    print(f"  Presupuesto fijo: {N_EVAL_BUDGET} evaluaciones de fitness por config")
    print("=" * 70)

    net, Xfit, Yfit, Xval, Yval, Xte, Yte = _setup()

    runs = {}
    configs = [
        ("Random search (sin AG)",
         lambda: random_search(net, Xfit, Yfit, Xval, Yval, Xte, Yte, N_EVAL_BUDGET)),
        ("Solo mutacion (sin cruce)",
         lambda: ga_run(net, Xfit, Yfit, Xval, Yval, Xte, Yte, pcross=0.0, pmut=0.05, elitism=2)),
        ("Solo cruce (sin mutacion)",
         lambda: ga_run(net, Xfit, Yfit, Xval, Yval, Xte, Yte, pcross=0.9, pmut=0.0, elitism=2)),
        ("Sin elitismo",
         lambda: ga_run(net, Xfit, Yfit, Xval, Yval, Xte, Yte, pcross=0.9, pmut=0.05, elitism=0)),
        ("AG COMPLETO (este proyecto)",
         lambda: ga_run(net, Xfit, Yfit, Xval, Yval, Xte, Yte, pcross=0.9, pmut=0.05, elitism=2)),
    ]

    for name, fn in configs:
        tt = time.time()
        r = fn()
        dt = time.time() - tt
        runs[name] = r
        print(f"  {name:35s} fit={r['best_fit']:.4f}  test_F1={r['test_f1']:.4f}  ({dt:.0f}s)")

    df_res = pd.DataFrame([
        {"config": k, "best_fit": v["best_fit"], "test_f1": v["test_f1"]}
        for k, v in runs.items()
    ])
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(OUTPUTS_DIR / "ablation_summary.csv", index=False)

    # Figura: convergencia + barras test F1
    figdir = OUTPUTS_DIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ["#95a5a6", "#f39c12", "#9b59b6", "#3498db", "#e74c3c"]
    for (name, r), c in zip(runs.items(), colors):
        axes[0].plot(r["history"], label=name, color=c, linewidth=2)
    axes[0].set_title("Convergencia: mejor fitness por generacion")
    axes[0].set_xlabel("Generacion (= POP evaluaciones)")
    axes[0].set_ylabel("Mejor fitness")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    names = list(runs.keys())
    f1s = [runs[n]["test_f1"] for n in names]
    axes[1].barh(range(len(names)), f1s, color=colors)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=10)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("F1-macro (test)")
    axes[1].set_title("Impacto de cada componente en el F1 de test")
    for i, v in enumerate(f1s):
        axes[1].text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=10)
    axes[1].grid(alpha=0.3, axis="x")

    fig.suptitle("Ablacion del AG — el rol de cada operador evolutivo", fontsize=13)
    fig.tight_layout()
    fig.savefig(figdir / "ablation.png", dpi=150)
    print(f"\nTiempo total: {time.time() - t0:.0f}s")
    print(f"Resumen: {OUTPUTS_DIR / 'ablation_summary.csv'}")
    print(f"Figura:  {figdir / 'ablation.png'}")


if __name__ == "__main__":
    main()
