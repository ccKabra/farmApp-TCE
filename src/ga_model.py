"""
Definicion centralizada del clasificador evolucionado por AG.
Es el analogo de model.py (BioBERTClassifier + load_finetuned_model) para la
version de Computacion Evolutiva. Importar desde aqui en todos los scripts.

FENOTIPO: red neuronal feed-forward de UNA capa oculta
    entrada (vector de features)  ->  Linear -> ReLU  ->  Linear -> Sigmoide
    salida = probabilidad por cada una de las ~98 etiquetas (multi-label).

GENOTIPO: el vector real (1D) con TODOS los pesos y sesgos de esa red
    genome = [ W1.ravel() | b1 | W2.ravel() | b2 ]
    Es una representacion NO BINARIA (Modulo 3): cada gen es un peso real. El
    Algoritmo Genetico (src/train_ga.py) busca el genoma que maximiza el F1.

Se implementa en NumPy puro (sin PyTorch) para tener control total sobre los
pesos y poder manipularlos como un cromosoma.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import GA_MODELS_DIR


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _relu(z):
    return np.maximum(0.0, z)


class GAClassifier:
    """Red de 1 capa oculta cuyos pesos viven en un cromosoma real (NumPy)."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        # Layout del genoma
        self.n_W1 = self.input_dim * self.hidden_dim
        self.n_b1 = self.hidden_dim
        self.n_W2 = self.hidden_dim * self.output_dim
        self.n_b2 = self.output_dim
        self.n_params = self.n_W1 + self.n_b1 + self.n_W2 + self.n_b2
        self.W1 = self.b1 = self.W2 = self.b2 = None

    # ── genotipo <-> parametros ───────────────────────────────────────────────
    def set_genome(self, g):
        """Desempaqueta el cromosoma 1D en las matrices de la red."""
        g = np.asarray(g, dtype=np.float64)
        d, h, o = self.input_dim, self.hidden_dim, self.output_dim
        i = 0
        self.W1 = g[i:i + self.n_W1].reshape(d, h); i += self.n_W1
        self.b1 = g[i:i + self.n_b1];               i += self.n_b1
        self.W2 = g[i:i + self.n_W2].reshape(h, o); i += self.n_W2
        self.b2 = g[i:i + self.n_b2];               i += self.n_b2
        return self

    def get_genome(self):
        """Aplana los parametros actuales a un cromosoma 1D."""
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, X):
        """X: [n, input_dim] -> probabilidades [n, output_dim]."""
        X = np.asarray(X, dtype=np.float64)
        hidden = _relu(X @ self.W1 + self.b1)
        return _sigmoid(hidden @ self.W2 + self.b2)

    def predict_proba(self, X):
        """Alias compatible con la interfaz de scikit-learn / la app."""
        return self.forward(X)


def load_ga_model():
    """
    Carga el modelo evolucionado desde disco.
    Retorna (model, featurizer, label_names, thresholds) — misma forma que
    model.load_finetuned_model(), con el featurizer en el lugar del tokenizer.
    """
    from ga_features import PatientFeaturizer  # import diferido (evita ciclos)

    d = Path(GA_MODELS_DIR)
    with open(d / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    model = GAClassifier(cfg["input_dim"], cfg["hidden_dim"], cfg["output_dim"])
    model.set_genome(np.load(d / "weights.npy"))

    featurizer = PatientFeaturizer.load(d)
    label_names = pd.read_csv(d / "label_names.csv")["label"].tolist()
    thresholds = np.load(d / "thresholds.npy")
    return model, featurizer, label_names, thresholds
