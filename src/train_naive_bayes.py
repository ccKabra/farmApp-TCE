"""
Baseline Naive Bayes sobre features TF-IDF.
Clasificador clasico de texto — Unidad 5 de la materia.

Mismo split (RANDOM_SEED, TEST_SIZE) que el resto de los modelos para que las
metricas sean comparables.
"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from config import DATA_PROCESSED, RANDOM_SEED, TEST_SIZE, OUTPUTS_DIR
import joblib

# ── Cargar datos ──────────────────────────────────────────────────────────────
X = pd.read_csv(DATA_PROCESSED / "X.csv")
Y = pd.read_csv(DATA_PROCESSED / "Y.csv")
print(f"X: {X.shape}  |  Y: {Y.shape}")

# ── Split (mismo seed que los otros modelos) ──────────────────────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# MultinomialNB necesita valores no negativos. X esta normalizado a [0,1], pero
# clippeamos por seguridad ante cambios futuros en las features.
X_train_nb = X_train.clip(lower=0)
X_test_nb = X_test.clip(lower=0)

# ── Entrenar ──────────────────────────────────────────────────────────────────
print("\nEntrenando Naive Bayes (MultinomialNB + OneVsRestClassifier)...")
model = OneVsRestClassifier(MultinomialNB(alpha=1.0))
model.fit(X_train_nb, Y_train)

# ── Predecir y evaluar ────────────────────────────────────────────────────────
Y_pred = model.predict(X_test_nb)

f1_mac = f1_score(Y_test, Y_pred, average="macro",   zero_division=0)
f1_mic = f1_score(Y_test, Y_pred, average="micro",   zero_division=0)
f1_sam = f1_score(Y_test, Y_pred, average="samples", zero_division=0)
hl     = hamming_loss(Y_test, Y_pred)

print(f"\n=== Naive Bayes — Resultados ===")
print(f"  F1 macro    : {f1_mac:.4f}")
print(f"  F1 micro    : {f1_mic:.4f}")
print(f"  F1 samples  : {f1_sam:.4f}")
print(f"  Hamming loss: {hl:.4f}")

# ── Guardar modelo ────────────────────────────────────────────────────────────
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
out = OUTPUTS_DIR / "naive_bayes_model.pkl"
joblib.dump(model, out)
print(f"\nModelo guardado en: {out}")
