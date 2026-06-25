"""
Etapa 4a: Generar embeddings BioBERT para cada caso del dataset.
Reemplaza el TF-IDF basico por representaciones semanticas densas.
USA EL MISMO filtrado y formato de texto que el resto del pipeline
(labels.build_label_vocab + patient_text.row_to_text) para evitar
inconsistencias de filas y distribuciones de entrada.

Output: data/processed/X_biobert.csv
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from config import DATA_PROCESSED, BIOBERT_MODEL, DEVICE
from labels import build_label_vocab
from patient_text import row_to_text

BATCH_SIZE = 64

def mean_pool(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def get_embeddings(texts, tokenizer, model, device, batch_size=BATCH_SIZE):
    all_embeddings = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=160,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        embeddings = mean_pool(output.last_hidden_state, encoded["attention_mask"])
        all_embeddings.append(embeddings.cpu().numpy())
        if (i // batch_size) % 5 == 0:
            print(f"  Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
    return np.vstack(all_embeddings)

print(f"Cargando BioBERT desde HuggingFace: {BIOBERT_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
model = AutoModel.from_pretrained(BIOBERT_MODEL).to(DEVICE)
print(f"Modelo cargado en {DEVICE.upper()}.")

df_raw = pd.read_csv(DATA_PROCESSED / "dataset.csv", dtype=str)
df_raw["age_years"] = pd.to_numeric(df_raw["age_years"], errors="coerce")
df_raw["weight_kg"] = pd.to_numeric(df_raw.get("weight_kg"), errors="coerce")

# Mismo filtrado que en entrenamiento (build_label_vocab usa config.MIN_REACTION_FREQ)
df, _ = build_label_vocab(df_raw)
df = df.reset_index(drop=True)
Y = pd.read_csv(DATA_PROCESSED / "Y.csv")

# Mismo formato de texto que en entrenamiento e inferencia
texts = df.apply(row_to_text, axis=1).tolist()

print(f"\nGenerando embeddings para {len(texts):,} casos...")
embeddings = get_embeddings(texts, tokenizer, model, DEVICE)
print(f"Embeddings shape: {embeddings.shape}")

# Reconstruir X combinando features demograficos + embeddings BioBERT
# X.csv fue construido con el mismo filtrado (prepare_features.py usa build_label_vocab)
X_old = pd.read_csv(DATA_PROCESSED / "X.csv")

if len(X_old) != len(df):
    raise ValueError(
        f"X.csv tiene {len(X_old)} filas pero el dataset filtrado tiene {len(df)}. "
        "Regenera X.csv con prepare_features.py antes de correr este script."
    )

non_indi_cols = [c for c in X_old.columns if not c.startswith("indi_")]
X_demo_drug = X_old[non_indi_cols]

emb_df = pd.DataFrame(embeddings, columns=[f"bb_{i}" for i in range(embeddings.shape[1])])
X_biobert = pd.concat([X_demo_drug.reset_index(drop=True), emb_df], axis=1)

out = DATA_PROCESSED / "X_biobert.csv"
X_biobert.to_csv(out, index=False)
print(f"X_biobert guardado: {X_biobert.shape} en {out}")
print(f"Texto de ejemplo: {texts[0]}")
