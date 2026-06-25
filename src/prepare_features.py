"""
Etapa 2b: Construcción de features y matriz multi-label.
CORREGIDO: split primero, TF-IDF y mediana de edad fitean solo sobre train
para evitar data leakage.

Input:  data/processed/dataset.csv
Output: data/processed/X.csv, data/processed/Y.csv, data/processed/label_names.txt
"""

import pandas as pd
import numpy as np
import pickle
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from config import DATA_PROCESSED, RANDOM_SEED, TEST_SIZE

MIN_REACTION_FREQ = 50
MIN_DRUG_FREQ = 10

df = pd.read_csv(DATA_PROCESSED / "dataset.csv", dtype=str)
df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")

# ── Etiquetas (Y) — calcular frecuencias sobre TODO el dataset es correcto
# porque solo estamos definiendo el vocabulario de etiquetas, no ajustando un estimador
all_reac = [r for row in df["reactions"].dropna() for r in row.split("|")]
freq_reac = {r for r, n in Counter(all_reac).items() if n >= MIN_REACTION_FREQ}
print(f"Etiquetas unicas con freq >= {MIN_REACTION_FREQ}: {len(freq_reac)}")

df["reaction_list"] = df["reactions"].apply(
    lambda s: [r for r in s.split("|") if r in freq_reac] if pd.notna(s) else []
)
mask = df["reaction_list"].apply(len) > 0
df = df[mask].reset_index(drop=True)
print(f"Casos con al menos una etiqueta frecuente: {len(df):,}")

mlb_reac = MultiLabelBinarizer()
Y = mlb_reac.fit_transform(df["reaction_list"])
Y_df = pd.DataFrame(Y, columns=mlb_reac.classes_)

# ── Split PRIMERO (antes de fitear cualquier transformador) ────────────────────
train_idx, test_idx = train_test_split(
    df.index, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
df_train = df.loc[train_idx]
df_test  = df.loc[test_idx]

# ── Features — fitear solo en train, transformar en ambos ─────────────────────

# 1. Sexo → one-hot (fit sobre train, align con test)
sex_dummies_train = pd.get_dummies(df_train["sex"], prefix="sex")
sex_dummies_test  = pd.get_dummies(df_test["sex"],  prefix="sex")
all_sex_cols = sex_dummies_train.columns.union(sex_dummies_test.columns)
sex_dummies_train = sex_dummies_train.reindex(columns=all_sex_cols, fill_value=0)
sex_dummies_test  = sex_dummies_test.reindex(columns=all_sex_cols,  fill_value=0)

# 2. Edad → mediana de train para imputar ambos
age_median_train = df_train["age_years"].median()
age_train = df_train["age_years"].fillna(age_median_train) / 100.0
age_test  = df_test["age_years"].fillna(age_median_train)  / 100.0

# 3. Fármaco → frecuencias calculadas solo en train
all_drugs_train = [d for row in df_train["drug"].dropna() for d in row.split("|")]
freq_drugs = {d for d, n in Counter(all_drugs_train).items() if n >= MIN_DRUG_FREQ}

def encode_drugs(drug_str):
    drugs = drug_str.split("|") if pd.notna(drug_str) else []
    known = [d for d in drugs if d in freq_drugs]
    return known if known else ["OTHER"]

mlb_drug = MultiLabelBinarizer()
drug_train = mlb_drug.fit_transform(df_train["drug"].apply(encode_drugs))
drug_test  = mlb_drug.transform(df_test["drug"].apply(encode_drugs))
drug_cols = [f"drug_{d}" for d in mlb_drug.classes_]
drug_train_df = pd.DataFrame(drug_train, columns=drug_cols)
drug_test_df  = pd.DataFrame(drug_test,  columns=drug_cols)

# 4. TF-IDF: fit solo en train
tfidf = TfidfVectorizer(max_features=100, min_df=5, token_pattern=r"[A-Za-z][A-Za-z ]{2,}")
indi_train_mat = tfidf.fit_transform(df_train["indications"].fillna("unknown")).toarray()
indi_test_mat  = tfidf.transform(df_test["indications"].fillna("unknown")).toarray()
indi_cols = [f"indi_{t}" for t in tfidf.get_feature_names_out()]
indi_train_df = pd.DataFrame(indi_train_mat, columns=indi_cols)
indi_test_df  = pd.DataFrame(indi_test_mat,  columns=indi_cols)

# ── Combinar y re-ordenar según índice original ───────────────────────────────
def build_X(age_s, sex_d, drug_d, indi_d):
    return pd.concat([
        age_s.reset_index(drop=True).rename("age_norm").to_frame(),
        sex_d.reset_index(drop=True),
        drug_d.reset_index(drop=True),
        indi_d.reset_index(drop=True),
    ], axis=1)

X_train = build_X(age_train, sex_dummies_train, drug_train_df, indi_train_df)
X_test  = build_X(age_test,  sex_dummies_test,  drug_test_df,  indi_test_df)

# Reconstruir X completo en el orden original del dataset (para biobert_embeddings.py)
X_full = pd.concat([X_train, X_test]).sort_index().reset_index(drop=True)
Y_df_aligned = Y_df  # Y_df ya tiene el mismo orden que df

print(f"\nX completo: {X_full.shape}  |  Y: {Y_df_aligned.shape}")
print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"Densidad de Y: {Y.mean():.4f}")

# ── Guardar ───────────────────────────────────────────────────────────────────
X_full.to_csv(DATA_PROCESSED / "X.csv", index=False)
Y_df_aligned.to_csv(DATA_PROCESSED / "Y.csv", index=False)

# Guardar índices de split para que los demás scripts usen el mismo
np.save(DATA_PROCESSED / "train_idx.npy", train_idx.to_numpy())
np.save(DATA_PROCESSED / "test_idx.npy",  test_idx.to_numpy())
# Archivo único con ambos índices (conveniencia para otros scripts)
np.savez(DATA_PROCESSED / "split_indices.npz",
         train_idx=train_idx.to_numpy(), test_idx=test_idx.to_numpy())

# Guardar transformadores para reutilizar en inference
with open(DATA_PROCESSED / "tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open(DATA_PROCESSED / "mlb_drug.pkl", "wb") as f:
    pickle.dump(mlb_drug, f)

with open(DATA_PROCESSED / "label_names.txt", "w") as f:
    f.write("\n".join(mlb_reac.classes_))

print(f"\nTop 10 etiquetas mas frecuentes:")
print(Y_df_aligned.sum().sort_values(ascending=False).head(10).to_string())
print(f"\nData leakage corregido: TF-IDF y mediana de edad fitean solo en train.")
