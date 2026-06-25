"""
Featurizacion del paciente para el clasificador evolucionado por AG.

ESTA ES LA UNICA DEFINICION del vector de entrada a la red (el analogo, para el
AG, de lo que `patient_text.py` es para BioBERT). La red optimizada por el
Algoritmo Genetico no consume texto crudo sino un vector numerico; este modulo
garantiza que entrenamiento e inferencia construyan EXACTAMENTE el mismo vector.

Composicion del vector:
  1. TF-IDF sobre el texto canonico del paciente (patient_text.row_to_text), el
     MISMO texto que ve BioBERT. Captura farmaco, indicaciones y concomitantes.
  2. Features estructurados: edad normalizada, peso normalizado y sexo one-hot.
     (los numeros se pierden en TF-IDF, por eso se agregan aparte)

El featurizer se fitea SOLO sobre train (mediana de edad/peso y vocabulario
TF-IDF) para no filtrar informacion del test (consistente con prepare_features.py).
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from patient_text import row_to_text, build_patient_text
from config import TFIDF_MAX_FEATURES, TFIDF_MIN_DF

# Nombres de los features estructurados (en orden, van al final del vector).
STRUCT_FEATURES = ["age_norm", "weight_norm", "sex_M", "sex_F"]


class PatientFeaturizer:
    """Convierte atributos del paciente -> vector real para la red del AG."""

    def __init__(self, max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF):
        self.max_features = max_features
        self.min_df = min_df
        self.tfidf = None
        self.tfidf_features = []
        self.age_median = 50.0
        self.weight_median = 70.0

    # ── helpers ───────────────────────────────────────────────────────────────
    def _structured_row(self, age, sex, weight):
        """[age_norm, weight_norm, sex_M, sex_F] para valores escalares."""
        a = float(age) if pd.notna(age) else self.age_median
        w = float(weight) if pd.notna(weight) else self.weight_median
        s = str(sex).strip().upper() if pd.notna(sex) else ""
        return [a / 100.0, w / 100.0, 1.0 if s == "M" else 0.0, 1.0 if s == "F" else 0.0]

    # ── fit / transform ───────────────────────────────────────────────────────
    def fit(self, df):
        """Ajusta TF-IDF y medianas SOLO sobre el dataframe de train."""
        texts = df.apply(row_to_text, axis=1).tolist()
        self.tfidf = TfidfVectorizer(max_features=self.max_features, min_df=self.min_df)
        self.tfidf.fit(texts)
        self.tfidf_features = list(self.tfidf.get_feature_names_out())

        age_med = pd.to_numeric(df["age_years"], errors="coerce").median()
        wt_med = pd.to_numeric(df.get("weight_kg"), errors="coerce").median()
        self.age_median = float(age_med) if np.isfinite(age_med) else 50.0
        self.weight_median = float(wt_med) if np.isfinite(wt_med) else 70.0
        return self

    def transform(self, df):
        """Dataframe de casos FAERS -> matriz [n, dim] (float32)."""
        texts = df.apply(row_to_text, axis=1).tolist()
        tf = self.tfidf.transform(texts).toarray().astype(np.float32)
        struct = np.array(
            [self._structured_row(r.get("age_years"), r.get("sex"), r.get("weight_kg"))
             for _, r in df.iterrows()],
            dtype=np.float32,
        )
        return np.hstack([tf, struct])

    def transform_one(self, age_years=None, sex=None, weight_kg=None,
                      drugs="", other_drugs="", indications=""):
        """Atributos de UN paciente nuevo -> vector [1, dim]. Usa el mismo texto
        canonico que en entrenamiento (build_patient_text)."""
        text = build_patient_text(age_years=age_years, sex=sex, weight_kg=weight_kg,
                                  drugs=drugs, other_drugs=other_drugs,
                                  indications=indications)
        tf = self.tfidf.transform([text]).toarray().astype(np.float32)
        struct = np.array([self._structured_row(age_years, sex, weight_kg)], dtype=np.float32)
        return np.hstack([tf, struct])

    # ── propiedades ───────────────────────────────────────────────────────────
    @property
    def feature_names(self):
        return list(self.tfidf_features) + STRUCT_FEATURES

    @property
    def dim(self):
        return len(self.tfidf_features) + len(STRUCT_FEATURES)

    # ── persistencia ──────────────────────────────────────────────────────────
    def save(self, d):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "featurizer_tfidf.pkl", "wb") as f:
            pickle.dump(self.tfidf, f)
        meta = {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "age_median": self.age_median,
            "weight_median": self.weight_median,
            "tfidf_features": self.tfidf_features,
            "struct_features": STRUCT_FEATURES,
        }
        with open(d / "featurizer.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, d):
        d = Path(d)
        with open(d / "featurizer.json", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(meta["max_features"], meta["min_df"])
        with open(d / "featurizer_tfidf.pkl", "rb") as f:
            obj.tfidf = pickle.load(f)
        obj.age_median = meta["age_median"]
        obj.weight_median = meta["weight_median"]
        obj.tfidf_features = meta["tfidf_features"]
        return obj
