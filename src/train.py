"""
Entrenamiento RESUMABLE y POR TANDAS de BioBERT (multi-label).

Reemplaza al flujo viejo de 2 scripts (train_biobert_finetune + tune_threshold).
Caracteristicas pensadas para no fundir la PC y poder cortar cuando quieras:

  * Guarda un checkpoint DESPUES DE CADA EPOCA (de forma atomica: escribe a un
    archivo temporal y lo renombra, asi un corte a mitad no corrompe nada).
  * Cada corrida hace solo EPOCHS_PER_RUN epocas y termina. Volves a correrlo
    y CONTINUA donde quedo, hasta llegar a TOTAL_EPOCHS.
  * Si interrumpis a mitad de una epoca, perdes solo esa epoca; el resto esta
    guardado. Al reanudar arranca desde la ultima epoca completa.

Uso normal:  paso2_entrenar.bat   (correrlo varias veces)
Estado:      models/biobert_finetuned/checkpoint.pt
Al terminar TOTAL_EPOCHS escribe el modelo final (model.pt + thresholds.npy).
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

from config import (MODELS_DIR, BIOBERT_MODEL, DEVICE, TEST_SIZE,
                    TOTAL_EPOCHS, EPOCHS_PER_RUN, BATCH_TRAIN, POS_WEIGHT_CAP)
from model import BioBERTClassifier
from patient_text import row_to_text, MAX_LEN
from data_split import load_split

BATCH_EVAL = 64
LR = 2e-5

SAVE_DIR = MODELS_DIR / "biobert_finetuned"
CKPT = SAVE_DIR / "checkpoint.pt"


class FAERSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], padding="max_length", truncation=True,
                             max_length=MAX_LEN, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": self.labels[idx]}


def get_probs(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for b in loader:
            logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(b["labels"].numpy())
    return np.vstack(probs), np.vstack(labels)


def best_thresholds(probs, labels, grid=np.arange(0.1, 0.9, 0.05)):
    th = np.full(probs.shape[1], 0.5)
    for j in range(probs.shape[1]):
        best_f1, best = 0, 0.5
        for t in grid:
            f1 = f1_score(labels[:, j], (probs[:, j] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best = f1, t
        th[j] = best
    return th


def atomic_save(obj, path):
    """Guarda a un .tmp y renombra: un corte durante la escritura no corrompe."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    # ── Datos (split determinista: misma semilla siempre) ─────────────────────
    df, label_names, train_idx, test_idx = load_split()
    num_labels = len(label_names)

    # Val = 10% del train (para calibrar umbral); resto entrena
    val_n = int(len(train_idx) * 0.1)
    val_idx, tr_idx = train_idx[:val_n], train_idx[val_n:]

    texts = df.apply(row_to_text, axis=1).tolist()
    Y = np.zeros((len(df), num_labels), dtype=np.float32)
    lab2i = {l: i for i, l in enumerate(label_names)}
    for i, reacs in enumerate(df["reaction_list"]):
        for r in reacs:
            Y[i, lab2i[r]] = 1.0

    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
    mk = lambda ix, shuf, bs: DataLoader(
        FAERSDataset([texts[i] for i in ix], Y[ix], tokenizer),
        batch_size=bs, shuffle=shuf, num_workers=0)
    train_loader = mk(tr_idx, True, BATCH_TRAIN)
    val_loader   = mk(val_idx, False, BATCH_EVAL)

    print(f"Casos: {len(df):,} | Etiquetas: {num_labels} | "
          f"Train: {len(tr_idx):,} Val: {len(val_idx):,} Test: {len(test_idx):,}")

    # ── Modelo / optimizador / scheduler ──────────────────────────────────────
    model = BioBERTClassifier(AutoModel.from_pretrained(BIOBERT_MODEL), num_labels).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * TOTAL_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    pos = Y[tr_idx].sum(axis=0)
    pos_weight = torch.tensor(np.minimum((len(tr_idx) - pos) / (pos + 1e-6), POS_WEIGHT_CAP),
                              dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Reanudar si hay checkpoint ────────────────────────────────────────────
    start_epoch = 0
    history = {"train_loss": [], "val_f1": []}
    best_val_f1, best_state, best_th = 0.0, None, np.full(num_labels, 0.5)

    if CKPT.exists():
        # weights_only=False: el checkpoint es nuestro y trae arrays numpy
        # (umbrales, historial). PyTorch 2.6+ exige declararlo explicitamente.
        ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
        if ck.get("num_labels") != num_labels:
            print(f"AVISO: el checkpoint tiene {ck.get('num_labels')} etiquetas y ahora hay "
                  f"{num_labels}. Cambiaron los datos -> empezando de cero.")
        else:
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            start_epoch = ck["epoch"]
            history = ck["history"]
            best_val_f1, best_state, best_th = ck["best_val_f1"], ck["best_state"], ck["best_th"]
            print(f"Reanudando desde epoca {start_epoch}/{TOTAL_EPOCHS} "
                  f"(mejor val F1 hasta ahora: {best_val_f1:.4f})")

    if start_epoch >= TOTAL_EPOCHS:
        print(f"Ya se completaron las {TOTAL_EPOCHS} epocas. Nada que hacer.")
        print("Para entrenar mas, subi TOTAL_EPOCHS en src/config.py y volve a correr.")
        return

    end_epoch = min(start_epoch + EPOCHS_PER_RUN, TOTAL_EPOCHS)
    print(f"\nEsta tanda: epocas {start_epoch + 1} a {end_epoch} "
          f"(de {TOTAL_EPOCHS} totales). Device: {DEVICE.upper()}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()
        total = 0.0
        for step, b in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
            loss = criterion(logits, b["labels"].to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
            if (step + 1) % 50 == 0:
                print(f"  Epoca {epoch} | Step {step+1}/{len(train_loader)} | "
                      f"Loss {total/(step+1):.4f}")

        avg_loss = total / len(train_loader)
        val_probs, val_labels = get_probs(model, val_loader)
        th = best_thresholds(val_probs, val_labels)
        val_f1 = f1_score(val_labels, (val_probs >= th).astype(int),
                          average="macro", zero_division=0)
        history["train_loss"].append(avg_loss)
        history["val_f1"].append(val_f1)
        print(f"[Epoca {epoch}/{TOTAL_EPOCHS}] loss={avg_loss:.4f} | val_F1_macro={val_f1:.4f}")

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_th = th.copy()
            print(f"  ** mejor modelo (val F1={best_val_f1:.4f}) **")

        # Checkpoint atomico tras CADA epoca
        atomic_save({
            "epoch": epoch, "num_labels": num_labels, "label_names": label_names,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "history": history,
            "best_val_f1": best_val_f1, "best_state": best_state, "best_th": best_th,
        }, CKPT)

    # ── Guardar SIEMPRE el mejor modelo hasta ahora (para poder probar entre tandas) ──
    torch.save(best_state, SAVE_DIR / "model.pt")
    np.save(SAVE_DIR / "thresholds.npy", best_th)
    tokenizer.save_pretrained(SAVE_DIR)
    pd.DataFrame({"label": label_names}).to_csv(SAVE_DIR / "label_names.csv", index=False)

    done = end_epoch >= TOTAL_EPOCHS
    print(f"\nGuardado mejor modelo (val F1={best_val_f1:.4f}) en {SAVE_DIR}")
    if done:
        print(f"ENTRENAMIENTO COMPLETO ({TOTAL_EPOCHS} epocas).")
        print("Ahora corre paso3_evaluar.bat para regenerar la tabla de test.")
    else:
        print(f"Tanda terminada en epoca {end_epoch}/{TOTAL_EPOCHS}. "
              f"Volve a correr paso2_entrenar.bat para seguir.")


if __name__ == "__main__":
    main()
