"""
Etapa 3b: Validacion cruzada k-fold sobre el baseline Random Forest.
Cubre la Unidad 3 (aprendizaje no supervisado / validacion cruzada).
Incluye curvas de aprendizaje para diagnosticar overfitting/underfitting.

Output: outputs/figures/cross_validation.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.metrics import f1_score, make_scorer
from config import DATA_PROCESSED, OUTPUTS_DIR, RANDOM_SEED

# ── Cargar datos ──────────────────────────────────────────────────────────────
X = pd.read_csv(DATA_PROCESSED / "X.csv").values
Y = pd.read_csv(DATA_PROCESSED / "Y.csv").values
print(f"X: {X.shape}  |  Y: {Y.shape}")

rf = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_leaf=5,
    n_jobs=-1, random_state=RANDOM_SEED, class_weight="balanced"
)
model = MultiOutputClassifier(rf, n_jobs=-1)

# ── K-Fold 5 pliegues ─────────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("\nValidacion cruzada 5-fold sobre Random Forest multilabel...")
f1_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr, X_val = X[train_idx], X[val_idx]
    Y_tr, Y_val = Y[train_idx], Y[val_idx]
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_val)
    f1_mac = f1_score(Y_val, Y_pred, average="macro", zero_division=0)
    f1_mic = f1_score(Y_val, Y_pred, average="micro", zero_division=0)
    f1_scores.append(f1_mac)
    print(f"  Fold {fold}: F1_macro={f1_mac:.4f}  F1_micro={f1_mic:.4f}")

print(f"\nF1 macro promedio : {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
print(f"Intervalo 95%     : [{np.mean(f1_scores) - 2*np.std(f1_scores):.4f}, "
      f"{np.mean(f1_scores) + 2*np.std(f1_scores):.4f}]")

# ── Curvas de aprendizaje ─────────────────────────────────────────────────────
print("\nCalculando curvas de aprendizaje (puede tardar unos minutos)...")

def f1_macro_multilabel(estimator, X, y):
    return f1_score(y, estimator.predict(X), average="macro", zero_division=0)

train_sizes = np.linspace(0.1, 1.0, 8)
train_sizes_abs, train_scores, val_scores = learning_curve(
    model, X, Y,
    train_sizes=train_sizes,
    cv=3,
    scoring=f1_macro_multilabel,
    n_jobs=-1,
    verbose=1
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

# ── Graficos ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Validacion cruzada y Curva de aprendizaje — Random Forest", fontsize=13)

# Grafico 1: F1 por fold
axes[0].bar(range(1, 6), f1_scores, color="steelblue", edgecolor="white")
axes[0].axhline(np.mean(f1_scores), color="red", linestyle="--",
                label=f"Media={np.mean(f1_scores):.3f}")
axes[0].fill_between(
    [-0.5, 5.5],
    np.mean(f1_scores) - np.std(f1_scores),
    np.mean(f1_scores) + np.std(f1_scores),
    alpha=0.15, color="red", label=f"+/- 1 std ({np.std(f1_scores):.3f})"
)
axes[0].set_xticks(range(1, 6))
axes[0].set_xticklabels([f"Fold {i}" for i in range(1, 6)])
axes[0].set_ylabel("F1 macro"); axes[0].set_ylim(0, 0.5)
axes[0].set_title("F1 macro por pliegue (5-fold CV)")
axes[0].legend()

# Grafico 2: Curva de aprendizaje
axes[1].plot(train_sizes_abs, train_mean, "o-", color="steelblue", label="Train")
axes[1].fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                     alpha=0.15, color="steelblue")
axes[1].plot(train_sizes_abs, val_mean, "o-", color="darkorange", label="Validacion")
axes[1].fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                     alpha=0.15, color="darkorange")
axes[1].set_xlabel("Tamano del conjunto de entrenamiento")
axes[1].set_ylabel("F1 macro")
axes[1].set_title("Curva de aprendizaje")
axes[1].legend()

# Diagnostico automatico
gap = train_mean[-1] - val_mean[-1]
if gap > 0.15:
    diag = "Overfitting detectado (gap train-val > 0.15). Reducir profundidad del arbol."
elif val_mean[-1] < 0.1:
    diag = "Underfitting: F1 de validacion bajo. Aumentar datos o complejidad del modelo."
else:
    diag = "Modelo balanceado. Agregar mas datos puede mejorar la validacion."
axes[1].set_title(f"Curva de aprendizaje\n({diag})", fontsize=9)

plt.tight_layout()
out = OUTPUTS_DIR / "figures" / "cross_validation.png"
plt.savefig(out, dpi=150)
print(f"\nGrafico guardado: {out}")
