"""
Etapa extra: Naive Bayes como tercer baseline (Unidad 5).
BernoulliNB + OneVsRestClassifier sobre TF-IDF.
Comparativa de los 3 modelos: NB, RF-TF-IDF, RF-BioBERT.

Output: outputs/figures/model_comparison.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from config import DATA_PROCESSED, OUTPUTS_DIR, RANDOM_SEED, TEST_SIZE

X    = pd.read_csv(DATA_PROCESSED / "X.csv")
Y    = pd.read_csv(DATA_PROCESSED / "Y.csv")
X_bb = pd.read_csv(DATA_PROCESSED / "X_biobert.csv")

label_names = Y.columns.tolist()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
X_bb_train, X_bb_test = X_bb.loc[X_train.index], X_bb.loc[X_test.index]

def evaluate(model, X_tr, X_te, Y_tr, Y_te, name):
    print(f"\nEntrenando {name}...")
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_te)
    return {
        "modelo": name,
        "F1 macro":   round(f1_score(Y_te, Y_pred, average="macro",   zero_division=0), 4),
        "F1 micro":   round(f1_score(Y_te, Y_pred, average="micro",   zero_division=0), 4),
        "F1 samples": round(f1_score(Y_te, Y_pred, average="samples", zero_division=0), 4),
        "Precision":  round(precision_score(Y_te, Y_pred, average="macro", zero_division=0), 4),
        "Recall":     round(recall_score(Y_te, Y_pred, average="macro",    zero_division=0), 4),
    }

nb = OneVsRestClassifier(BernoulliNB(), n_jobs=-1)
rf = MultiOutputClassifier(RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_leaf=5,
    n_jobs=-1, random_state=RANDOM_SEED, class_weight="balanced"), n_jobs=-1)
rf_bb = MultiOutputClassifier(RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_leaf=5,
    n_jobs=-1, random_state=RANDOM_SEED, class_weight="balanced"), n_jobs=-1)

results = [
    evaluate(nb,    X_train,    X_test,    Y_train, Y_test, "Naive Bayes (BernoulliNB)"),
    evaluate(rf,    X_train,    X_test,    Y_train, Y_test, "Random Forest + TF-IDF"),
    evaluate(rf_bb, X_bb_train, X_bb_test, Y_train, Y_test, "Random Forest + BioBERT"),
]
# Agregar BioBERT fine-tuned (resultados del script anterior)
results.append({
    "modelo": "BioBERT Fine-tuned",
    "F1 macro": 0.1283, "F1 micro": 0.1063, "F1 samples": 0.0980,
    "Precision": 0.1064, "Recall": 0.3878,
})

df_res = pd.DataFrame(results)
print("\n=== COMPARATIVA DE MODELOS ===")
print(df_res.to_string(index=False))

# ── Grafico ───────────────────────────────────────────────────────────────────
metrics = ["F1 macro", "F1 micro", "F1 samples", "Precision", "Recall"]
colors  = ["steelblue", "darkorange", "mediumseagreen", "mediumpurple"]
x = np.arange(len(metrics))
w = 0.2

fig, ax = plt.subplots(figsize=(14, 6))
for i, (_, row) in enumerate(df_res.iterrows()):
    vals = [row[m] for m in metrics]
    bars = ax.bar(x + i * w, vals, w, label=row["modelo"], color=colors[i])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x + w * 1.5)
ax.set_xticklabels(metrics)
ax.set_ylabel("Score"); ax.set_ylim(0, 0.7)
ax.set_title("Comparativa de modelos: Naive Bayes vs RF vs BioBERT")
ax.legend()
plt.tight_layout()
out = OUTPUTS_DIR / "figures" / "model_comparison.png"
plt.savefig(out, dpi=150)
print(f"\nGrafico guardado: {out}")
