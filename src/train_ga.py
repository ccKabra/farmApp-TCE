"""
Entrenamiento del clasificador por ALGORITMO GENETICO (proyecto farmApp-TCE).

Reemplaza la retropropagacion de BioBERT (src/train.py) por un AG que optimiza
los pesos de una red de una capa oculta (ga_model.GAClassifier). Es la etapa 5
de la version evolutiva del pipeline.

Mapa a los modulos de la materia
--------------------------------
  * Representacion NO binaria (Modulo 3): el genotipo es el vector real de los
    pesos de la red; el fenotipo es la red clasificadora. Inicializamos el sesgo
    de salida en logit(prior) de cada etiqueta -> "inyeccion de conocimiento"
    via la representacion (arranca cerca de la frecuencia base de cada clase).
  * Seleccion / fitness (Modulo 2 y 4): seleccion por TORNEO; fitness = F1-macro.
  * Operadores: cruce (uniforme / un punto / aritmetico) + mutacion GAUSSIANA.
  * Elitismo (Modulo 2): los mejores individuos pasan intactos -> garantiza que
    el mejor nunca empeora (unica garantia teorica de convergencia).
  * Control de parametros (Modulo 5): control DETERMINISTICO de la mutacion,
    sigma decrece linealmente de GA_MUT_SIGMA a GA_MUT_SIGMA_END (mas
    exploracion al principio, mas explotacion al final).
  * Metricas (Modulo 6): se registra best/mean fitness, diversidad (desvio del
    fitness) y presion de seleccion SelPres = MaxFit/AveFit por generacion;
    curva de convergencia en outputs/figures/ga_convergence.png.

Metodologia (por que asi)
-------------------------
  * El AG ajusta los pesos maximizando F1-macro sobre un subconjunto de TRAIN
    (analogo a minimizar la perdida de entrenamiento en backprop).
  * El conjunto de VALIDACION (10% del train) se usa para calibrar el umbral
    optimo POR ETIQUETA del mejor individuo (igual criterio que train.py).
  * El conjunto de TEST (30%, nunca visto) da la metrica final no sesgada.
  * Mismo split determinista (semilla 42, data_split.load_split) que el resto
    del proyecto -> los casos de test coinciden con los de la app y eval.

Salida: models/ga_model/{weights.npy, thresholds.npy, label_names.csv,
config.json, featurizer*.{pkl,json}} y outputs/ga_evolution.csv + figura.
"""

import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             hamming_loss)

from config import (GA_MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED, GA_HIDDEN,
                    GA_POP_SIZE, GA_GENERATIONS, GA_TOURNAMENT_K, GA_P_CROSSOVER,
                    GA_CROSSOVER, GA_P_MUTATION, GA_MUT_SIGMA, GA_MUT_SIGMA_END,
                    GA_ELITISM, GA_INIT_SCALE, GA_FIT_SUBSAMPLE,
                    GA_VAL_FRACTION, TFIDF_MAX_FEATURES, TFIDF_MIN_DF,
                    GA_SELECTION, GA_RANK_S, GA_DIV_LOW, GA_DIV_HIGH,
                    GA_IMMIGRANTS, GA_IMMIGRANT_PERIOD, GA_SIGMA_BOOST,
                    GA_SIGMA_DAMP)
from data_split import load_split
from ga_features import PatientFeaturizer
from ga_model import GAClassifier
from ga_metrics import (genotypic_diversity, theoretical_selection_pressure,
                        rank_selection_probs)

rng = np.random.default_rng(RANDOM_SEED)
FIT_GRID = np.arange(0.10, 0.90, 0.10)        # grilla (gruesa) para el fitness
PERLABEL_GRID = np.arange(0.10, 0.90, 0.05)   # grilla (fina) para calibrar umbrales


# ──────────────────────────────────────────────────────────────────────────────
#  Datos y metricas
# ──────────────────────────────────────────────────────────────────────────────
def build_Y(df, label_names):
    """Matriz binaria [n, n_labels] a partir de la columna reaction_list."""
    lab2i = {l: i for i, l in enumerate(label_names)}
    Y = np.zeros((len(df), len(label_names)), dtype=bool)
    for i, reacs in enumerate(df["reaction_list"]):
        for r in reacs:
            j = lab2i.get(r)
            if j is not None:
                Y[i, j] = True
    return Y


def _f1_per_label(probs, Yb, t):
    """F1 de CADA etiqueta a un umbral t (vectorizado, sin sklearn)."""
    pred = probs >= t
    tp = np.logical_and(pred, Yb).sum(0).astype(np.float64)
    pp = pred.sum(0).astype(np.float64)
    ap = Yb.sum(0).astype(np.float64)
    prec = np.divide(tp, pp, out=np.zeros_like(tp), where=pp > 0)
    rec = np.divide(tp, ap, out=np.zeros_like(tp), where=ap > 0)
    return np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp),
                     where=(prec + rec) > 0)


def best_thresholds_and_f1(probs, Yb, grid):
    """Umbral optimo POR ETIQUETA sobre una grilla. Devuelve (umbrales, f1_por_etiqueta).
    El F1-macro alcanzable con esos umbrales es f1.mean(): esa es la senal de
    fitness (recompensa cualquier etiqueta cuyo ranking mejore, no solo si un
    unico umbral global la captura -> paisaje mucho mas suave)."""
    n_lab = probs.shape[1]
    best_t = np.full(n_lab, 0.5)
    best_f = np.zeros(n_lab)
    for t in grid:
        f1 = _f1_per_label(probs, Yb, t)
        upd = f1 > best_f
        best_f[upd] = f1[upd]
        best_t[upd] = t
    return best_t, best_f


# ──────────────────────────────────────────────────────────────────────────────
#  Operadores del AG (sobre cromosomas reales)
# ──────────────────────────────────────────────────────────────────────────────
def init_population(net, pop_size, scale):
    """Poblacion inicial de cromosomas reales. Pesos ~ N(0, sigma) con escala
    tipo He/Glorot; sesgos ~ 0. Asi las salidas sigmoide arrancan repartidas
    alrededor de 0.5 (el clasificador trivial "predecir todo" ya da un F1-macro
    no nulo ~densidad), dando al AG un paisaje de fitness con pendiente.

    NOTA: probamos inicializar el sesgo de salida en logit(prior) de cada
    etiqueta ("inyeccion de conocimiento", Modulo 3) pero con densidad ~2% las
    salidas colapsan en ~0.02 y NINGUNA cruza el umbral -> fitness plano en 0.
    Por eso arrancamos los sesgos en ~0 y dejamos que el AG baje las salidas de
    las etiquetas que no deben dispararse."""
    d, h, o = net.input_dim, net.hidden_dim, net.output_dim
    pop = np.empty((pop_size, net.n_params), dtype=np.float64)
    for k in range(pop_size):
        W1 = rng.normal(0, scale * np.sqrt(2.0 / d), size=net.n_W1)
        b1 = rng.normal(0, 0.01, size=net.n_b1)
        W2 = rng.normal(0, scale * np.sqrt(1.0 / h), size=net.n_W2)
        b2 = rng.normal(0, 0.05, size=net.n_b2)
        pop[k] = np.concatenate([W1, b1, W2, b2])
    return pop


def tournament_select(fitness, k):
    """Devuelve el indice ganador de un torneo de k individuos (presion de
    seleccion controlada por k)."""
    aspirants = rng.integers(0, len(fitness), size=k)
    return aspirants[np.argmax(fitness[aspirants])]


def crossover(p1, p2, kind):
    """Recombinacion de dos cromosomas reales."""
    if kind == "uniform":
        mask = rng.random(p1.shape) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2
    if kind == "one_point":
        cut = rng.integers(1, p1.shape[0])
        c1 = np.concatenate([p1[:cut], p2[cut:]])
        c2 = np.concatenate([p2[:cut], p1[cut:]])
        return c1, c2
    if kind == "arithmetic":   # whole arithmetic / BLX
        a = rng.random()
        c1 = a * p1 + (1 - a) * p2
        c2 = (1 - a) * p1 + a * p2
        return c1, c2
    raise ValueError(f"Cruce desconocido: {kind}")


def mutate(genome, pmut, sigma):
    """Mutacion gaussiana para genotipo real: cada gen, con probabilidad pmut,
    recibe un ruido N(0, sigma). Sigma se atenua a lo largo de las generaciones
    (self-adaptation basica del Modulo 5)."""
    mask = rng.random(genome.shape) < pmut
    genome = genome.copy()
    genome[mask] += rng.normal(0, sigma, size=int(mask.sum()))
    return genome


# ──────────────────────────────────────────────────────────────────────────────
#  Bucle evolutivo
# ──────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 70)
    print("  ENTRENAMIENTO POR ALGORITMO GENETICO  (farmApp-TCE)")
    print("=" * 70)

    # ── Datos (mismo split determinista que todo el proyecto) ─────────────────
    df, label_names, train_idx, test_idx = load_split()
    num_labels = len(label_names)
    val_n = int(len(train_idx) * GA_VAL_FRACTION)
    val_idx, tr_idx = train_idx[:val_n], train_idx[val_n:]

    df_tr, df_val, df_te = df.iloc[tr_idx], df.iloc[val_idx], df.iloc[test_idx]

    # ── Featurizacion (fit solo en train) ─────────────────────────────────────
    feat = PatientFeaturizer(TFIDF_MAX_FEATURES, TFIDF_MIN_DF).fit(df_tr)
    Xtr, Xval, Xte = feat.transform(df_tr), feat.transform(df_val), feat.transform(df_te)
    Ytr, Yval, Yte = (build_Y(df_tr, label_names), build_Y(df_val, label_names),
                      build_Y(df_te, label_names))

    print(f"Casos: {len(df):,} | Etiquetas: {num_labels} | dim entrada: {feat.dim}")
    print(f"Train: {len(df_tr):,}  Val: {len(df_val):,}  Test: {len(df_te):,}")

    # Subconjunto fijo de train para evaluar fitness (acelera el AG)
    if GA_FIT_SUBSAMPLE and GA_FIT_SUBSAMPLE < len(Xtr):
        sub = rng.choice(len(Xtr), size=GA_FIT_SUBSAMPLE, replace=False)
        Xfit, Yfit = Xtr[sub], Ytr[sub]
    else:
        Xfit, Yfit = Xtr, Ytr
    print(f"Conjunto de fitness: {len(Xfit):,} casos")

    # ── Modelo / poblacion inicial ────────────────────────────────────────────
    net = GAClassifier(feat.dim, GA_HIDDEN, num_labels)
    print(f"Fenotipo: red {feat.dim}->{GA_HIDDEN}->{num_labels} | "
          f"genoma = {net.n_params:,} pesos reales")

    pop = init_population(net, GA_POP_SIZE, GA_INIT_SCALE)

    def fitness_of(genome):
        net.set_genome(genome)
        _, f1 = best_thresholds_and_f1(net.forward(Xfit), Yfit, FIT_GRID)
        return float(f1.mean())

    fitness = np.array([fitness_of(ind) for ind in pop])

    # Presion selectiva TEORICA del operador (constante mientras no cambie)
    if GA_SELECTION == "rank":
        selpres_theo = theoretical_selection_pressure("rank", s=GA_RANK_S)
    else:
        selpres_theo = theoretical_selection_pressure("tournament", k=GA_TOURNAMENT_K)

    def select_parent(fitness_arr, rank_p=None):
        """Devuelve el indice de un padre segun GA_SELECTION."""
        if GA_SELECTION == "rank":
            return int(rng.choice(len(fitness_arr), p=rank_p))
        return tournament_select(fitness_arr, GA_TOURNAMENT_K)

    print(f"\n{'Gen':>4} | {'best':>7} | {'mean':>7} | {'divG':>6} | "
          f"{'SelPres':>7} | {'sigma':>6} | {'evt':>6}")
    print("-" * 62)

    history = []
    best_genome = pop[np.argmax(fitness)].copy()
    best_fit = float(fitness.max())
    last_immigrant_gen = -GA_IMMIGRANT_PERIOD

    for gen in range(1, GA_GENERATIONS + 1):
        # ── Metricas de la generacion (Modulo 6) ──────────────────────────────
        ave = float(fitness.mean())
        mx = float(fitness.max())
        div_geno = genotypic_diversity(pop)     # <-- diversidad del CROMOSOMA

        # ── Control ADAPTATIVO de sigma (Modulo 5) ────────────────────────────
        # Base deterministica: sigma decrece linealmente
        frac = (gen - 1) / max(1, GA_GENERATIONS - 1)
        sigma_base = GA_MUT_SIGMA + frac * (GA_MUT_SIGMA_END - GA_MUT_SIGMA)
        # Modulacion reactiva por diversidad genotipica:
        if div_geno < GA_DIV_LOW:
            sigma = sigma_base * GA_SIGMA_BOOST     # poca diversidad -> explora
            event = "boost"
        elif div_geno > GA_DIV_HIGH:
            sigma = sigma_base * GA_SIGMA_DAMP      # mucha diversidad -> explota
            event = "damp"
        else:
            sigma = sigma_base
            event = "-"

        # ── Inmigrantes aleatorios (Modulo 5) ─────────────────────────────────
        # Se activan si la diversidad GENOTIPICA cae Y ya paso el periodo minimo
        # desde la ultima inyeccion (evita reinyectar todas las gens seguidas).
        do_immigrants = (div_geno < GA_DIV_LOW and GA_IMMIGRANTS > 0
                         and gen - last_immigrant_gen >= GA_IMMIGRANT_PERIOD)

        history.append({"gen": gen, "best": mx, "mean": ave,
                        "div_geno": div_geno, "selpres": selpres_theo,
                        "sigma": sigma, "event": event,
                        "immigrants": int(do_immigrants)})
        if gen == 1 or gen % 10 == 0 or gen == GA_GENERATIONS:
            print(f"{gen:>4} | {mx:7.4f} | {ave:7.4f} | {div_geno:6.4f} | "
                  f"{selpres_theo:7.3f} | {sigma:6.3f} | {event:>6}")

        # ── Nueva generacion ──────────────────────────────────────────────────
        order = np.argsort(fitness)[::-1]
        new_pop = [pop[order[e]].copy() for e in range(GA_ELITISM)]   # elitismo

        # Precomputo probabilidades de rank (una sola vez por gen)
        rank_p = rank_selection_probs(fitness, GA_RANK_S) if GA_SELECTION == "rank" else None

        while len(new_pop) < GA_POP_SIZE:
            p1 = pop[select_parent(fitness, rank_p)]
            p2 = pop[select_parent(fitness, rank_p)]
            if rng.random() < GA_P_CROSSOVER:
                c1, c2 = crossover(p1, p2, GA_CROSSOVER)
            else:
                c1, c2 = p1.copy(), p2.copy()
            new_pop.append(mutate(c1, GA_P_MUTATION, sigma))
            if len(new_pop) < GA_POP_SIZE:
                new_pop.append(mutate(c2, GA_P_MUTATION, sigma))

        pop = np.array(new_pop)

        if do_immigrants:
            immigrants = init_population(net, GA_IMMIGRANTS, GA_INIT_SCALE)
            pop[-GA_IMMIGRANTS:] = immigrants
            last_immigrant_gen = gen

        fitness = np.array([fitness_of(ind) for ind in pop])

        gen_best = float(fitness.max())
        if gen_best > best_fit:
            best_fit = gen_best
            best_genome = pop[np.argmax(fitness)].copy()

    print("-" * 52)
    print(f"Mejor fitness (F1-macro en train-sub): {best_fit:.4f}")

    # ── Calibrar umbral por etiqueta sobre validacion ─────────────────────────
    net.set_genome(best_genome)
    val_probs = net.forward(Xval)
    thresholds, _ = best_thresholds_and_f1(val_probs, Yval, PERLABEL_GRID)

    # ── Metricas finales sobre test (no visto) ────────────────────────────────
    te_probs = net.forward(Xte)
    te_pred = (te_probs >= thresholds).astype(int)
    Yte_i = Yte.astype(int)
    metrics = {
        "f1_macro": f1_score(Yte_i, te_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(Yte_i, te_pred, average="micro", zero_division=0),
        "f1_samples": f1_score(Yte_i, te_pred, average="samples", zero_division=0),
        "precision_macro": precision_score(Yte_i, te_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(Yte_i, te_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(Yte_i, te_pred),
    }
    print("\n=== METRICAS EN TEST (30% no visto) ===")
    for k, v in metrics.items():
        print(f"  {k:16s}: {v:.4f}")

    # ── Guardar modelo ────────────────────────────────────────────────────────
    GA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(GA_MODELS_DIR / "weights.npy", best_genome)
    np.save(GA_MODELS_DIR / "thresholds.npy", thresholds)
    pd.DataFrame({"label": label_names}).to_csv(
        GA_MODELS_DIR / "label_names.csv", index=False)
    feat.save(GA_MODELS_DIR)
    config = {
        "input_dim": feat.dim, "hidden_dim": GA_HIDDEN, "output_dim": num_labels,
        "n_params": net.n_params,
        "ga": {"pop_size": GA_POP_SIZE, "generations": GA_GENERATIONS,
               "tournament_k": GA_TOURNAMENT_K, "p_crossover": GA_P_CROSSOVER,
               "crossover": GA_CROSSOVER, "p_mutation": GA_P_MUTATION,
               "sigma0": GA_MUT_SIGMA, "sigma_end": GA_MUT_SIGMA_END,
               "elitism": GA_ELITISM, "init_scale": GA_INIT_SCALE,
               "fit_subsample": GA_FIT_SUBSAMPLE},
        "best_fitness_train": best_fit,
        "test_metrics": metrics,
        "seconds": round(time.time() - t0, 1),
    }
    with open(GA_MODELS_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ── Registro de evolucion + curva de convergencia (Modulo 6) ──────────────
    hist_df = pd.DataFrame(history)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    hist_df.to_csv(OUTPUTS_DIR / "ga_evolution.csv", index=False)

    imm_gens = hist_df.loc[hist_df["immigrants"] == 1, "gen"].tolist()
    sigma_smooth = hist_df["sigma"].rolling(window=10, min_periods=1, center=True).mean()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Convergencia + marcas de inyeccion de inmigrantes
    ax1 = axes[0]
    for g in imm_gens:
        ax1.axvline(g, color="#8e44ad", alpha=0.20, lw=1)
    ax1.plot(hist_df["gen"], hist_df["best"], label="Mejor fitness", color="#2ecc71", lw=2)
    ax1.plot(hist_df["gen"], hist_df["mean"], label="Fitness medio", color="#3498db", lw=1.5)
    ax1.set_xlabel("Generacion"); ax1.set_ylabel("F1-macro (train-sub)")
    ax1.set_title(f"Curva de convergencia  (lineas violetas = {len(imm_gens)} inyecciones)")
    ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)

    # 2) Diversidad GENOTIPICA + umbrales + eventos
    ax2 = axes[1]
    for g in imm_gens:
        ax2.axvline(g, color="#8e44ad", alpha=0.25, lw=1)
    ax2.plot(hist_df["gen"], hist_df["div_geno"],
             label="Diversidad genotipica", color="#e67e22", lw=2)
    ax2.axhline(GA_DIV_LOW, color="#c0392b", ls="--", alpha=0.6,
                label=f"umbral bajo ({GA_DIV_LOW})  -> inyecta")
    ax2.axhline(GA_DIV_HIGH, color="#27ae60", ls="--", alpha=0.6,
                label=f"umbral alto ({GA_DIV_HIGH})  -> dampea")
    ax2.set_xlabel("Generacion"); ax2.set_ylabel("std promedio del cromosoma")
    ax2.set_title(f"Diversidad genotipica  ({len(imm_gens)} inyecciones)")
    ax2.legend(loc="upper right", fontsize=8); ax2.grid(alpha=0.3)

    # 3) Presion selectiva TEORICA + sigma adaptativo SUAVIZADO
    ax3 = axes[2]
    ax3.plot(hist_df["gen"], hist_df["selpres"],
             label=f"Presion selectiva ({GA_SELECTION}, s={GA_RANK_S})",
             color="#9b59b6", lw=2.5)
    ax3.set_xlabel("Generacion"); ax3.set_ylabel("Presion selectiva (teorica)", color="#9b59b6")
    ax3.tick_params(axis="y", labelcolor="#9b59b6")
    ax3.set_ylim(0, 3.0)
    ax3.grid(alpha=0.3)
    ax3b = ax3.twinx()
    ax3b.plot(hist_df["gen"], sigma_smooth,
              label="sigma adaptativo (media movil 10)", color="#16a085", lw=2)
    ax3b.set_ylabel("sigma de mutacion", color="#16a085")
    ax3b.tick_params(axis="y", labelcolor="#16a085")
    ax3.set_title("Presion selectiva vs sigma adaptativo")
    ax3.legend(loc="upper left", fontsize=8)
    ax3b.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig_path = OUTPUTS_DIR / "figures" / "ga_convergence.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)

    print(f"\nModelo guardado en: {GA_MODELS_DIR}")
    print(f"Evolucion: {OUTPUTS_DIR / 'ga_evolution.csv'}")
    print(f"Figura:    {fig_path}")
    print(f"Tiempo total: {config['seconds']}s")
    print("\nAhora corre:  python src/eval_test_cases.py   (tabla de test para la app)")


if __name__ == "__main__":
    main()
