"""
Etapa 3: Modelo baseline multi-label con Random Forest.
Split 70/30, métricas por etiqueta y globales.

Output: models/baseline_rf.pkl, outputs/figures/baseline_metrics.png
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, hamming_loss
)
from config import DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED, TEST_SIZE

# ── Cargar datos ──────────────────────────────────────────────────────────────
print("Cargando X e Y...")
X = pd.read_csv(DATA_PROCESSED / "X.csv")
Y = pd.read_csv(DATA_PROCESSED / "Y.csv")
label_names = Y.columns.tolist()

print(f"X: {X.shape}  |  Y: {Y.shape}")

# ── Split 70/30 ───────────────────────────────────────────────────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Entrenar ──────────────────────────────────────────────────────────────────
print("\nEntrenando Random Forest multi-label...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    class_weight="balanced",
)
model = MultiOutputClassifier(rf, n_jobs=-1)
model.fit(X_train, Y_train)
print("Entrenamiento completo.")

# ── Evaluar ───────────────────────────────────────────────────────────────────
Y_pred = model.predict(X_test)

f1_macro   = f1_score(Y_test, Y_pred, average="macro",   zero_division=0)
f1_micro   = f1_score(Y_test, Y_pred, average="micro",   zero_division=0)
f1_samples = f1_score(Y_test, Y_pred, average="samples", zero_division=0)
precision  = precision_score(Y_test, Y_pred, average="macro", zero_division=0)
recall     = recall_score(Y_test, Y_pred, average="macro",    zero_division=0)
h_loss     = hamming_loss(Y_test, Y_pred)

print("\n=== METRICAS GLOBALES ===")
print(f"  F1 macro   : {f1_macro:.4f}")
print(f"  F1 micro   : {f1_micro:.4f}")
print(f"  F1 samples : {f1_samples:.4f}")
print(f"  Precision  : {precision:.4f}")
print(f"  Recall     : {recall:.4f}")
print(f"  Hamming loss: {h_loss:.4f}")

# F1 por etiqueta
f1_per_label = f1_score(Y_test, Y_pred, average=None, zero_division=0)
label_metrics = pd.DataFrame({
    "label": label_names,
    "f1": f1_per_label,
    "support": Y_test.values.sum(axis=0)
}).sort_values("f1", ascending=False)

print("\n=== TOP 15 ETIQUETAS POR F1 ===")
print(label_metrics.head(15).to_string(index=False))
print("\n=== BOTTOM 10 ETIQUETAS POR F1 ===")
print(label_metrics.tail(10).to_string(index=False))

# ── Guardar modelo ────────────────────────────────────────────────────────────
MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODELS_DIR / "baseline_rf.pkl"
with open(model_path, "wb") as f:
    pickle.dump({"model": model, "label_names": label_names, "feature_names": X.columns.tolist()}, f)
print(f"\nModelo guardado en: {model_path}")

# ── Gráfico ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Baseline Random Forest — Resultados", fontsize=13)

# F1 por etiqueta (top 30)
top30 = label_metrics.head(30)
axes[0].barh(top30["label"][::-1], top30["f1"][::-1], color="steelblue")
axes[0].set_title("F1 por etiqueta (Top 30)")
axes[0].set_xlabel("F1 Score")
axes[0].axvline(f1_macro, color="red", linestyle="--", label=f"F1 macro={f1_macro:.3f}")
axes[0].legend()

# Métricas globales
metrics = {"F1 macro": f1_macro, "F1 micro": f1_micro, "F1 samples": f1_samples,
           "Precision": precision, "Recall": recall}
axes[1].bar(metrics.keys(), metrics.values(), color=["steelblue","darkorange","mediumseagreen","salmon","mediumpurple"])
axes[1].set_ylim(0, 1)
axes[1].set_title("Metricas globales")
for i, (k, v) in enumerate(metrics.items()):
    axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

plt.tight_layout()
out = OUTPUTS_DIR / "figures" / "baseline_metrics.png"
plt.savefig(out, dpi=150)
print(f"Grafico guardado en: {out}")

# ── Validacion cruzada (5-Fold) — Unidad 3 ────────────────────────────────────
from sklearn.model_selection import KFold

print("\n=== Validacion cruzada (5-Fold) sobre el train set ===")
cv_model = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=200, class_weight="balanced",
                           random_state=RANDOM_SEED, n_jobs=-1)
)
# cross_val_score no soporta F1 macro multilabel directamente -> loop manual
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_scores = []
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_cv_tr,  X_cv_val  = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    Y_cv_tr,  Y_cv_val  = Y_train.iloc[tr_idx], Y_train.iloc[val_idx]
    cv_model.fit(X_cv_tr, Y_cv_tr)
    f1 = f1_score(Y_cv_val, cv_model.predict(X_cv_val), average="macro", zero_division=0)
    cv_scores.append(f1)
    print(f"  Fold {fold}: F1 macro = {f1:.4f}")
print(f"\n  Media: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

# ── Curva de aprendizaje — Unidad 3 ───────────────────────────────────────────
print("\nGenerando curva de aprendizaje...")
train_sizes_abs = [s for s in [500, 1000, 2000, 3000, 4000] if s < len(X_train)]
train_sizes_abs.append(len(X_train))
train_f1s, val_f1s = [], []
for size in train_sizes_abs:
    X_sub, Y_sub = X_train.iloc[:size], Y_train.iloc[:size]
    m = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, class_weight="balanced",
                               random_state=RANDOM_SEED, n_jobs=-1)
    )
    m.fit(X_sub, Y_sub)
    f1_train = f1_score(Y_sub, m.predict(X_sub), average="macro", zero_division=0)
    f1_test  = f1_score(Y_test, m.predict(X_test), average="macro", zero_division=0)
    train_f1s.append(f1_train)
    val_f1s.append(f1_test)
    print(f"  n={size:5d} -> train F1={f1_train:.4f}, test F1={f1_test:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes_abs, train_f1s, "o-", label="F1 en entrenamiento", color="steelblue")
ax.plot(train_sizes_abs, val_f1s,   "o-", label="F1 en test",          color="coral")
ax.set_xlabel("Cantidad de muestras de entrenamiento")
ax.set_ylabel("F1 macro")
ax.set_title("Curva de aprendizaje — Random Forest baseline")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
lc_out = OUTPUTS_DIR / "figures" / "learning_curve_rf.png"
lc_out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(lc_out, dpi=150)
print(f"\nCurva de aprendizaje guardada en: {lc_out}")

gap = train_f1s[-1] - val_f1s[-1]
if gap > 0.15:
    print(f"Diagnostico: posible OVERFITTING (gap train-test = {gap:.3f}).")
elif val_f1s[-1] < 0.1:
    print(f"Diagnostico: posible UNDERFITTING (F1 test bajo = {val_f1s[-1]:.3f}).")
else:
    print(f"Diagnostico: modelo balanceado (gap = {gap:.3f}).")
