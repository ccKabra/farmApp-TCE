"""
Etapa extra: Similitud coseno entre embeddings de farmacos.
Aplica el modelo vectorial de la Unidad 2 sobre los embeddings BioBERT
para encontrar farmacos semanticamente similares en el espacio de representacion.

Output: outputs/figures/drug_similarity.png, outputs/drug_similarity.csv
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from config import DATA_PROCESSED, OUTPUTS_DIR, BIOBERT_MODEL, DEVICE

print(f"Cargando BioBERT para embeddings de farmacos...")
tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
model = AutoModel.from_pretrained(BIOBERT_MODEL).to(DEVICE)
model.eval()

def embed_text(texts):
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=32, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    return (out.last_hidden_state * mask).sum(1) / mask.sum(1)

# Top 25 farmacos mas frecuentes
df = pd.read_csv(DATA_PROCESSED / "dataset.csv", dtype=str)
all_drugs = [d for row in df["drug"].dropna() for d in row.split("|")]
top_drugs = [d for d, _ in Counter(all_drugs).most_common(25)]
print(f"Calculando embeddings para {len(top_drugs)} farmacos...")

# Embeber en batches de 5
embeddings = []
for i in range(0, len(top_drugs), 5):
    batch = top_drugs[i:i+5]
    emb = embed_text(batch).cpu().numpy()
    embeddings.append(emb)
embeddings = np.vstack(embeddings)

# Similitud coseno
sim_matrix = cosine_similarity(embeddings)
sim_df = pd.DataFrame(sim_matrix, index=top_drugs, columns=top_drugs)

# Top pares mas similares (excluyendo diagonal)
pairs = []
for i in range(len(top_drugs)):
    for j in range(i+1, len(top_drugs)):
        pairs.append((top_drugs[i], top_drugs[j], sim_matrix[i, j]))
pairs.sort(key=lambda x: x[2], reverse=True)

print("\nTop 15 pares de farmacos mas similares semanticamente:")
for d1, d2, sim in pairs[:15]:
    print(f"  {d1:<35} <-> {d2:<35} sim={sim:.4f}")

pd.DataFrame(pairs, columns=["drug_1", "drug_2", "cosine_similarity"])\
  .to_csv(OUTPUTS_DIR / "drug_similarity.csv", index=False)

# ── Graficos ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Similitud coseno entre embeddings BioBERT de farmacos", fontsize=13)

# Heatmap
mask_diag = np.eye(len(top_drugs), dtype=bool)
sns.heatmap(sim_df, ax=axes[0], cmap="RdYlGn", vmin=0.7, vmax=1.0,
            mask=mask_diag, xticklabels=True, yticklabels=True,
            cbar_kws={"label": "Similitud coseno"})
axes[0].set_title("Heatmap de similitud entre farmacos")
axes[0].tick_params(axis="x", rotation=45, labelsize=7)
axes[0].tick_params(axis="y", labelsize=7)

# Top 15 pares
pair_labels = [f"{d1[:15]}–{d2[:15]}" for d1, d2, _ in pairs[:15]]
pair_sims   = [s for _, _, s in pairs[:15]]
axes[1].barh(pair_labels[::-1], pair_sims[::-1], color="mediumseagreen")
axes[1].set_xlim(0.7, 1.0)
axes[1].set_title("Top 15 pares mas similares")
axes[1].set_xlabel("Similitud coseno")

plt.tight_layout()
out = OUTPUTS_DIR / "figures" / "drug_similarity.png"
plt.savefig(out, dpi=150)
print(f"\nGrafico guardado: {out}")
