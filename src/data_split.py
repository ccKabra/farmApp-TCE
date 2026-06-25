"""
Split train/test canonico (70/30, semilla fija).

Replica EXACTAMENTE el filtrado y el split de train_biobert_finetune.py para
que cualquier script (evaluacion, app) pueda saber que casos son de test
(nunca vistos por el modelo) y cuales de entrenamiento.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_PROCESSED, RANDOM_SEED, TEST_SIZE
from labels import build_label_vocab


def load_split():
    """
    Retorna (df, label_names, train_idx, test_idx).
    df esta filtrado igual que en entrenamiento; los indices refieren a df.
    """
    df = pd.read_csv(DATA_PROCESSED / "dataset.csv", dtype=str)
    df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
    df["weight_kg"] = pd.to_numeric(df.get("weight_kg"), errors="coerce")

    df, label_names = build_label_vocab(df)

    idx = list(range(len(df)))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    return df, label_names, train_idx, test_idx
