"""
Extra evolutivo — SELECCION DE FEATURES con cromosoma BINARIO (Modulo 2 y 3).

Mientras train_ga.py usa un genotipo REAL (los pesos de la red), este script
muestra el OTRO tipo de representacion clasico de los AG: un genotipo BINARIO.

Pregunta que responde: "el TF-IDF mete 150 terminos, pero ¿cuales hacen falta
de verdad?". En vez de quedarnos con todos, dejamos que un Algoritmo Genetico
DECIDA el subconjunto de features. Asi el TF-IDF deja de ser un aporte fijo y
pasa a ser materia prima que la evolucion selecciona.

  * Genotipo: mascara binaria m in {0,1}^dim (1 = usar ese feature). [Modulo 3]
  * Fenotipo: el clasificador que solo ve los features encendidos.
  * Operadores binarios CANONICOS: cruce uniforme + mutacion bit-flip. [Modulo 2]
  * Fitness: F1-macro de un regresor lineal de forma cerrada (ridge) entrenado
    SOLO con los features elegidos, MENOS una penalidad de parsimonia (preferir
    pocas features). La penalidad es una restriccion blanda (constraint, Modulo 4).
    El regresor lineal es solo el ORACULO de evaluacion, barato; el aporte es la
    seleccion evolutiva.

Salida: outputs/ga_feature_selection.csv (features elegidas) +
        outputs/figures/ga_feature_selection.png (evolucion).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (OUTPUTS_DIR, RANDOM_SEED, TFIDF_MAX_FEATURES, TFIDF_MIN_DF,
                    GA_VAL_FRACTION)
from data_split import load_split
from ga_features import PatientFeaturizer
from train_ga import build_Y, best_thresholds_and_f1, FIT_GRID

rng = np.random.default_rng(RANDOM_SEED)

# Hiperparametros del AG binario
POP = 40
GENS = 30
TOURNEY_K = 3
P_CROSS = 0.9
P_MUT = 0.03          # prob. de flip por bit
ELITES = 2
PARSIMONY = 0.04      # peso de la penalidad por usar muchas features
SUBSAMPLE = 2500
RIDGE_ALPHA = 5.0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def ridge_scores(Xtr, Ytr, Xval, alpha=RIDGE_ALPHA):
    """Regresor lineal multisalida de forma cerrada (ridge). Oraculo de fitness.
    W = (X^T X + alpha I)^-1 X^T Y ; scores_val = Xval @ W."""
    k = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * np.eye(k)
    b = Xtr.T @ Ytr.astype(np.float64)
    W = np.linalg.solve(A, b)
    return Xval @ W


def evaluate(mask, Xtr, Ytr, Xval, Yval):
    """F1-macro (con umbral por etiqueta) usando solo las features de la mascara,
    menos la penalidad de parsimonia."""
    if mask.sum() == 0:
        return 0.0, 0.0
    cols = np.where(mask)[0]
    scores = ridge_scores(Xtr[:, cols], Ytr, Xval[:, cols])
    # llevar a [0,1] por etiqueta (z-score + sigmoide) para usar la grilla de umbral
    z = (scores - scores.mean(0)) / (scores.std(0) + 1e-9)
    probs = _sigmoid(z)
    _, f1_per = best_thresholds_and_f1(probs, Yval, FIT_GRID)
    f1 = float(f1_per.mean())
    frac = mask.sum() / len(mask)
    return f1 - PARSIMONY * frac, f1


def tournament(fitness):
    asp = rng.integers(0, len(fitness), size=TOURNEY_K)
    return asp[np.argmax(fitness[asp])]


def uniform_crossover(a, b):
    swap = rng.random(a.shape) < 0.5
    c1 = np.where(swap, a, b)
    c2 = np.where(swap, b, a)
    return c1, c2


def bitflip(mask):
    flip = rng.random(mask.shape) < P_MUT
    out = mask.copy()
    out[flip] = ~out[flip]
    return out


def main():
    print("=" * 70)
    print("  SELECCION EVOLUTIVA DE FEATURES — genotipo BINARIO (Modulo 2/3)")
    print("=" * 70)

    df, label_names, train_idx, test_idx = load_split()
    val_n = int(len(train_idx) * GA_VAL_FRACTION)
    val_idx, tr_idx = train_idx[:val_n], train_idx[val_n:]

    feat = PatientFeaturizer(TFIDF_MAX_FEATURES, TFIDF_MIN_DF).fit(df.iloc[tr_idx])
    Xtr_full = feat.transform(df.iloc[tr_idx])
    Xval = feat.transform(df.iloc[val_idx])
    Ytr = build_Y(df.iloc[tr_idx], label_names)
    Yval = build_Y(df.iloc[val_idx], label_names)
    names = np.array(feat.feature_names)
    dim = feat.dim

    if SUBSAMPLE and SUBSAMPLE < len(Xtr_full):
        sub = rng.choice(len(Xtr_full), size=SUBSAMPLE, replace=False)
        Xtr, Ytr_s = Xtr_full[sub], Ytr[sub]
    else:
        Xtr, Ytr_s = Xtr_full, Ytr

    # Linea base: usar TODAS las features
    _, base_f1 = evaluate(np.ones(dim, dtype=bool), Xtr, Ytr_s, Xval, Yval)
    print(f"Features totales (TF-IDF + demograficos): {dim}")
    print(f"F1 base usando TODAS las features: {base_f1:.4f}\n")

    # Poblacion inicial: mascaras con ~50% de features encendidas
    pop = rng.random((POP, dim)) < 0.5
    fit = np.array([evaluate(m, Xtr, Ytr_s, Xval, Yval)[0] for m in pop])

    history = []
    best_mask = pop[np.argmax(fit)].copy()
    best_fit = fit.max()

    print(f"{'Gen':>4} | {'fitness':>7} | {'F1':>7} | {'#feats':>6}")
    print("-" * 34)
    for gen in range(1, GENS + 1):
        order = np.argsort(fit)[::-1]
        new = [pop[order[e]].copy() for e in range(ELITES)]
        while len(new) < POP:
            p1 = pop[tournament(fit)]
            p2 = pop[tournament(fit)]
            if rng.random() < P_CROSS:
                c1, c2 = uniform_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            new.append(bitflip(c1))
            if len(new) < POP:
                new.append(bitflip(c2))
        pop = np.array(new)
        fit = np.array([evaluate(m, Xtr, Ytr_s, Xval, Yval)[0] for m in pop])

        gi = int(np.argmax(fit))
        if fit[gi] > best_fit:
            best_fit = fit[gi]
            best_mask = pop[gi].copy()
        _, best_real_f1 = evaluate(best_mask, Xtr, Ytr_s, Xval, Yval)
        history.append({"gen": gen, "fitness": float(fit.max()),
                        "f1": best_real_f1, "n_feats": int(best_mask.sum())})
        if gen == 1 or gen % 5 == 0 or gen == GENS:
            print(f"{gen:>4} | {fit.max():7.4f} | {best_real_f1:7.4f} | {best_mask.sum():>6}")

    # ── Resultado ─────────────────────────────────────────────────────────────
    _, final_f1 = evaluate(best_mask, Xtr, Ytr_s, Xval, Yval)
    sel = names[best_mask]
    print("-" * 34)
    print(f"\nSeleccionadas {best_mask.sum()} de {dim} features "
          f"({best_mask.sum()/dim:.0%})")
    print(f"F1 con subconjunto evolucionado: {final_f1:.4f}  (base con todas: {base_f1:.4f})")
    print(f"\nFeatures elegidas (primeras 30):\n{', '.join(sel[:30])}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": sel}).to_csv(
        OUTPUTS_DIR / "ga_feature_selection.csv", index=False)

    hist = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(hist["gen"], hist["f1"], "-o", color="#e74c3c", label="F1 (mejor)")
    ax1.axhline(base_f1, color="#7f8c8d", ls="--", label=f"F1 con TODAS ({base_f1:.3f})")
    ax1.set_xlabel("Generacion"); ax1.set_ylabel("F1-macro (val)"); ax1.legend(loc="lower right")
    ax2 = ax1.twinx()
    ax2.plot(hist["gen"], hist["n_feats"], "-s", color="#2980b9", alpha=0.6,
             label="# features")
    ax2.set_ylabel("# features seleccionadas")
    ax2.legend(loc="upper right")
    plt.title("Seleccion evolutiva de features (genotipo binario)")
    plt.tight_layout()
    out = OUTPUTS_DIR / "figures" / "ga_feature_selection.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\nGuardado: {OUTPUTS_DIR / 'ga_feature_selection.csv'}")
    print(f"Figura:   {out}")


if __name__ == "__main__":
    main()
