"""
Featurizacion del paciente para el clasificador evolucionado por AG.

ESTA ES LA UNICA DEFINICION del vector de entrada a la red (el analogo, para el
AG, de lo que `patient_text.py` es para BioBERT). La red optimizada por el
Algoritmo Genetico no consume texto crudo sino un vector numerico; este modulo
garantiza que entrenamiento e inferencia construyan EXACTAMENTE el mismo vector.

REPRESENTACION (v2) — knowledge injection a traves de la codificacion (Modulo 3):
  1. **Farmaco sospechoso**: multi-hot sobre el top-N de farmacos vistos en train
     (N = TFIDF_MAX_FEATURES // 2 por defecto). El nombre del farmaco es la
     senial mas fuerte del dataset y aca le damos UNA COLUMNA POR FARMACO en vez
     de enterrarla en una bolsa de palabras.
  2. **Indicaciones**: multi-hot sobre el top-M de indicaciones de train.
  3. **Medicaciones concomitantes**: multi-hot sobre el top-K mas frecuentes.
  4. **Demografia**: edad/peso normalizados + sexo one-hot.
  5. **TF-IDF residual**: una bolsa chica (~40 terminos) para texto libre que no
     entro en los vocabularios cerrados.

Por que es mejor que TF-IDF crudo: el AG ya no tiene que "descubrir" que la
palabra LENALIDOMIDE significa el farmaco; lo lee directo de una columna
dedicada. La localidad del genotipo (Modulo 3) mejora porque cada gen tiene un
significado claro.

El featurizer se fitea SOLO sobre train (vocabularios, medianas) para no filtrar
informacion del test.
"""

import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from patient_text import build_patient_text
from config import TFIDF_MAX_FEATURES, TFIDF_MIN_DF

# Capacidades de los vocabularios cerrados (lo que sobra va a TF-IDF residual)
TOP_DRUGS_DEFAULT = 90        # top farmacos sospechosos (one-hot/multi-hot)
TOP_INDICATIONS_DEFAULT = 50  # top indicaciones (multi-hot)
TOP_OTHER_DRUGS_DEFAULT = 40  # top concomitantes (multi-hot)
TFIDF_RESIDUAL = 40           # bolsa chica para el texto residual

# Nombres de los features estructurados (van despues de los multi-hot).
STRUCT_FEATURES = [
    "age_norm", "weight_norm", "sex_M", "sex_F",
    "age_lt30", "age_30_60", "age_gt60",   # binning suave para no-linealidades faciles
    "no_other_drugs",                       # bandera: ningun concomitante
]


def _split_pipe(s):
    """'A|B|C' -> ['A','B','C']; NaN -> []."""
    if not isinstance(s, str) or not s.strip():
        return []
    return [t.strip().upper() for t in s.split("|") if t.strip()]


def _residual_text(drug, other_drugs, indications, known_drugs, known_indi, known_other):
    """Texto residual = lo que NO esta en los vocabularios cerrados (alimenta TF-IDF)."""
    bag = []
    for t in _split_pipe(drug):
        if t not in known_drugs:
            bag.append(t.lower())
    for t in _split_pipe(other_drugs):
        if t not in known_other:
            bag.append(t.lower())
    for t in _split_pipe(indications):
        if t not in known_indi:
            bag.append(t.lower())
    return " ".join(bag) if bag else "unknown"


class PatientFeaturizer:
    """Convierte atributos del paciente -> vector real para la red del AG."""

    def __init__(self, max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF,
                 top_drugs=TOP_DRUGS_DEFAULT, top_indi=TOP_INDICATIONS_DEFAULT,
                 top_other=TOP_OTHER_DRUGS_DEFAULT, tfidf_residual=TFIDF_RESIDUAL):
        # Limites de los vocabularios cerrados
        self.top_drugs = top_drugs
        self.top_indi = top_indi
        self.top_other = top_other
        self.tfidf_residual = tfidf_residual
        # Compat con la API vieja
        self.max_features = max_features
        self.min_df = min_df
        # Estado aprendido
        self.drug_vocab = []      # lista ordenada de farmacos top
        self.indi_vocab = []
        self.other_vocab = []
        self.tfidf = None
        self.tfidf_features = []
        self.age_median = 50.0
        self.weight_median = 70.0

    # ── helpers internos ──────────────────────────────────────────────────────
    def _structured_row(self, age, sex, weight, other_drugs_str):
        a = float(age) if pd.notna(age) else self.age_median
        w = float(weight) if pd.notna(weight) else self.weight_median
        s = str(sex).strip().upper() if pd.notna(sex) else ""
        no_other = 1.0 if not _split_pipe(other_drugs_str) else 0.0
        return [
            a / 100.0, w / 100.0,
            1.0 if s == "M" else 0.0,
            1.0 if s == "F" else 0.0,
            1.0 if a < 30 else 0.0,
            1.0 if 30 <= a <= 60 else 0.0,
            1.0 if a > 60 else 0.0,
            no_other,
        ]

    def _multi_hot(self, items, vocab_index):
        v = np.zeros(len(vocab_index), dtype=np.float32)
        for t in items:
            j = vocab_index.get(t)
            if j is not None:
                v[j] = 1.0
        return v

    # ── fit ────────────────────────────────────────────────────────────────────
    def fit(self, df):
        """Aprende vocabularios cerrados y TF-IDF residual SOLO sobre train."""
        # Vocabularios cerrados por frecuencia
        drugs = Counter(t for s in df["drug"].dropna() for t in _split_pipe(s))
        indis = Counter(t for s in df["indications"].dropna() for t in _split_pipe(s))
        others = Counter(t for s in df.get("other_drugs", pd.Series(dtype=str)).dropna()
                         for t in _split_pipe(s))
        self.drug_vocab = [t for t, _ in drugs.most_common(self.top_drugs)]
        self.indi_vocab = [t for t, _ in indis.most_common(self.top_indi)]
        self.other_vocab = [t for t, _ in others.most_common(self.top_other)]

        known_d, known_i, known_o = (set(self.drug_vocab), set(self.indi_vocab),
                                     set(self.other_vocab))

        # TF-IDF residual: solo sobre lo que NO entro en los vocabularios cerrados
        residuals = [
            _residual_text(r.get("drug"), r.get("other_drugs"), r.get("indications"),
                           known_d, known_i, known_o)
            for _, r in df.iterrows()
        ]
        self.tfidf = TfidfVectorizer(max_features=self.tfidf_residual, min_df=self.min_df,
                                     token_pattern=r"[a-z][a-z0-9]+")
        try:
            self.tfidf.fit(residuals)
            self.tfidf_features = list(self.tfidf.get_feature_names_out())
        except ValueError:
            # Caso extremo: sin texto residual util; dejamos TF-IDF vacio
            self.tfidf_features = []

        age_med = pd.to_numeric(df["age_years"], errors="coerce").median()
        wt_med = pd.to_numeric(df.get("weight_kg"), errors="coerce").median()
        self.age_median = float(age_med) if np.isfinite(age_med) else 50.0
        self.weight_median = float(wt_med) if np.isfinite(wt_med) else 70.0
        return self

    # ── transform ──────────────────────────────────────────────────────────────
    def transform(self, df):
        """Dataframe de casos FAERS -> matriz [n, dim] (float32)."""
        d_idx = {t: i for i, t in enumerate(self.drug_vocab)}
        i_idx = {t: i for i, t in enumerate(self.indi_vocab)}
        o_idx = {t: i for i, t in enumerate(self.other_vocab)}
        known_d, known_i, known_o = set(self.drug_vocab), set(self.indi_vocab), set(self.other_vocab)

        n = len(df)
        n_d, n_i, n_o = len(self.drug_vocab), len(self.indi_vocab), len(self.other_vocab)
        n_tf = len(self.tfidf_features)
        n_st = len(STRUCT_FEATURES)
        out = np.zeros((n, n_d + n_i + n_o + n_tf + n_st), dtype=np.float32)

        residuals = []
        for k, (_, r) in enumerate(df.iterrows()):
            out[k, :n_d] = self._multi_hot(_split_pipe(r.get("drug")), d_idx)
            out[k, n_d:n_d + n_i] = self._multi_hot(_split_pipe(r.get("indications")), i_idx)
            out[k, n_d + n_i:n_d + n_i + n_o] = self._multi_hot(_split_pipe(r.get("other_drugs")), o_idx)
            residuals.append(_residual_text(r.get("drug"), r.get("other_drugs"),
                                            r.get("indications"), known_d, known_i, known_o))
            out[k, n_d + n_i + n_o + n_tf:] = self._structured_row(
                r.get("age_years"), r.get("sex"), r.get("weight_kg"), r.get("other_drugs"))

        if n_tf and self.tfidf is not None:
            out[:, n_d + n_i + n_o:n_d + n_i + n_o + n_tf] = (
                self.tfidf.transform(residuals).toarray().astype(np.float32))
        return out

    def transform_one(self, age_years=None, sex=None, weight_kg=None,
                      drugs="", other_drugs="", indications=""):
        """Atributos de UN paciente nuevo -> vector [1, dim]. Usa exactamente el
        mismo encoding que en entrenamiento."""
        row = {"age_years": age_years, "sex": sex, "weight_kg": weight_kg,
               "drug": drugs or "", "other_drugs": other_drugs or "",
               "indications": indications or ""}
        df = pd.DataFrame([row])
        return self.transform(df)

    # ── propiedades ───────────────────────────────────────────────────────────
    @property
    def feature_names(self):
        return ([f"drug_{t}" for t in self.drug_vocab]
                + [f"indi_{t}" for t in self.indi_vocab]
                + [f"other_{t}" for t in self.other_vocab]
                + [f"tf_{t}" for t in self.tfidf_features]
                + list(STRUCT_FEATURES))

    @property
    def dim(self):
        return (len(self.drug_vocab) + len(self.indi_vocab) + len(self.other_vocab)
                + len(self.tfidf_features) + len(STRUCT_FEATURES))

    # ── persistencia ──────────────────────────────────────────────────────────
    def save(self, d):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "featurizer_tfidf.pkl", "wb") as f:
            pickle.dump(self.tfidf, f)
        meta = {
            "version": 2,
            "max_features": self.max_features,
            "min_df": self.min_df,
            "top_drugs": self.top_drugs,
            "top_indi": self.top_indi,
            "top_other": self.top_other,
            "tfidf_residual": self.tfidf_residual,
            "drug_vocab": self.drug_vocab,
            "indi_vocab": self.indi_vocab,
            "other_vocab": self.other_vocab,
            "tfidf_features": self.tfidf_features,
            "struct_features": STRUCT_FEATURES,
            "age_median": self.age_median,
            "weight_median": self.weight_median,
        }
        with open(d / "featurizer.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, d):
        d = Path(d)
        with open(d / "featurizer.json", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(max_features=meta.get("max_features", TFIDF_MAX_FEATURES),
                  min_df=meta.get("min_df", TFIDF_MIN_DF),
                  top_drugs=meta.get("top_drugs", TOP_DRUGS_DEFAULT),
                  top_indi=meta.get("top_indi", TOP_INDICATIONS_DEFAULT),
                  top_other=meta.get("top_other", TOP_OTHER_DRUGS_DEFAULT),
                  tfidf_residual=meta.get("tfidf_residual", TFIDF_RESIDUAL))
        with open(d / "featurizer_tfidf.pkl", "rb") as f:
            obj.tfidf = pickle.load(f)
        obj.drug_vocab = meta.get("drug_vocab", [])
        obj.indi_vocab = meta.get("indi_vocab", [])
        obj.other_vocab = meta.get("other_vocab", [])
        obj.tfidf_features = meta["tfidf_features"]
        obj.age_median = meta["age_median"]
        obj.weight_median = meta["weight_median"]
        return obj
