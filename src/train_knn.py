"""
Baseline KNN sobre features TF-IDF.
Clasificador por vecinos mas cercanos — Unidad 3 de la materia.
Usa metric="cosine" porque las features son TF-IDF (similitud coseno, Unidad 2).
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from config import DATA_PROCESSED, RANDOM_SEED, TEST_SIZE

# ── Cargar datos ──────────────────────────────────────────────────────────────
X = pd.read_csv(DATA_PROCESSED / "X.csv")
Y = pd.read_csv(DATA_PROCESSED / "Y.csv")
print(f"X: {X.shape}  |  Y: {Y.shape}")

# ── Split (mismo seed que los otros modelos) ──────────────────────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# ── Entrenar KNN con k=5 (valor clasico) ──────────────────────────────────────
print("\nEntrenando KNN (k=5 + OneVsRestClassifier, metric=cosine)...")
model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, metric="cosine"))
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

f1_mac = f1_score(Y_test, Y_pred, average="macro",   zero_division=0)
f1_mic = f1_score(Y_test, Y_pred, average="micro",   zero_division=0)
f1_sam = f1_score(Y_test, Y_pred, average="samples", zero_division=0)
hl     = hamming_loss(Y_test, Y_pred)

print(f"\n=== KNN (k=5) — Resultados ===")
print(f"  F1 macro    : {f1_mac:.4f}")
print(f"  F1 micro    : {f1_mic:.4f}")
print(f"  F1 samples  : {f1_sam:.4f}")
print(f"  Hamming loss: {hl:.4f}")

# ── Comparar distintos k ──────────────────────────────────────────────────────
print("\n--- Comparacion de valores de k ---")
for k in [3, 5, 7, 11]:
    m = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=k, metric="cosine"))
    m.fit(X_train, Y_train)
    yp = m.predict(X_test)
    f1 = f1_score(Y_test, yp, average="macro", zero_division=0)
    print(f"  k={k:2d} -> F1 macro: {f1:.4f}")
