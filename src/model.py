"""
Definicion centralizada del modelo BioBERTClassifier.
Importar desde aqui en todos los scripts.
"""

import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from config import BIOBERT_MODEL, DEVICE, MODELS_DIR
import numpy as np
import pandas as pd


class BioBERTClassifier(nn.Module):
    def __init__(self, bert_model, num_labels, dropout=0.3):
        super().__init__()
        self.bert = bert_model
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


def load_finetuned_model():
    """Carga el modelo fine-tuned desde disco. Retorna (model, tokenizer, label_names, thresholds)."""
    import torch
    save_dir = MODELS_DIR / "biobert_finetuned"
    label_names = pd.read_csv(save_dir / "label_names.csv")["label"].tolist()
    thresholds = np.load(save_dir / "thresholds.npy")
    tokenizer = AutoTokenizer.from_pretrained(str(save_dir))
    bert_base = AutoModel.from_pretrained(BIOBERT_MODEL)
    model = BioBERTClassifier(bert_base, len(label_names)).to(DEVICE)
    model.load_state_dict(torch.load(save_dir / "model.pt", map_location=DEVICE))
    model.eval()
    return model, tokenizer, label_names, thresholds
