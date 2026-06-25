"""
Etapa NER: Extraccion de entidades biomedicas con scispaCy (o spaCy como fallback).
Demuestra extraccion de entidades del texto libre de indicaciones FAERS.
Cubre Unidad 6 (Procesamiento de Lenguaje Natural).

Modelos intentados en orden:
  1. en_core_sci_md  (scispaCy, biomedico completo)
  2. en_core_sci_sm  (scispaCy, biomedico ligero)
  3. en_core_web_sm  (spaCy general, fallback garantizado)
"""

import spacy
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from config import DATA_PROCESSED, OUTPUTS_DIR

def load_best_model():
    """
    scispaCy requiere thinc<8.3 pero nuestro spaCy 3.8 usa thinc 8.3 -> incompatible en Py3.13.
    Usamos en_core_web_sm (ya instalado) y agregamos un EntityRuler con patrones biomedicos
    para simular reconocimiento de entidades de indicaciones medicas.
    """
    nlp = spacy.load("en_core_web_sm")

    # EntityRuler con terminos medicos frecuentes en FAERS
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [
        {"label": "DISEASE",  "pattern": [{"LOWER": {"IN": [
            "diabetes", "hypertension", "cancer", "asthma", "arthritis", "depression",
            "anxiety", "osteoporosis", "psoriasis", "crohn", "colitis", "melanoma",
            "lymphoma", "leukemia", "anemia", "fibromyalgia", "lupus", "sclerosis",
            "parkinson", "alzheimer", "epilepsy", "migraine", "obesity", "hypothyroidism",
            "hyperthyroidism", "atrial", "fibrillation", "pneumonia", "hepatitis",
        ]}}]},
        {"label": "CONDITION", "pattern": [{"LOWER": {"IN": [
            "pain", "fatigue", "nausea", "insomnia", "inflammation", "infection",
            "disorder", "syndrome", "disease", "failure", "deficiency", "insufficiency",
        ]}}]},
        {"label": "DRUG_USE",  "pattern": [{"LOWER": {"IN": [
            "treatment", "therapy", "prophylaxis", "prevention", "management",
        ]}}]},
    ]
    ruler.add_patterns(patterns)
    print(f"Modelo cargado: en_core_web_sm + EntityRuler biomedico")
    return nlp, "en_core_web_sm+EntityRuler"

print("Cargando modelo NLP...")
nlp, model_used = load_best_model()

# ── Cargar datos ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PROCESSED / "dataset.csv", dtype=str)
sample = df[df["indications"].notna()].head(500).copy()
print(f"Analizando {len(sample)} indicaciones...")

# ── NER ────────────────────────────────────────────────────────────────────────
all_entities = []
ner_results = []

for _, row in sample.iterrows():
    text = str(row["indications"])
    doc = nlp(text)
    entities = [(ent.text.strip(), ent.label_) for ent in doc.ents if len(ent.text.strip()) > 2]
    all_entities.extend(entities)
    ner_results.append({
        "primaryid":       row["primaryid"],
        "drug":            row["drug"],
        "indications_text": text,
        "entities_found":  "|".join(f"{t}[{l}]" for t, l in entities),
        "n_entities":      len(entities),
    })

ner_df = pd.DataFrame(ner_results)
ner_df.to_csv(DATA_PROCESSED / "ner_results.csv", index=False)

# ── Estadísticas ───────────────────────────────────────────────────────────────
entity_counts = Counter(t for t, _ in all_entities)
label_counts  = Counter(l for _, l in all_entities)

print(f"\nModelo usado       : {model_used}")
print(f"Entidades totales  : {len(all_entities):,}")
print(f"Entidades unicas   : {len(entity_counts):,}")
print(f"Casos con >= 1 ent.: {(ner_df['n_entities'] > 0).sum():,}/{len(ner_df)}")
print(f"\nTipos de entidad encontrados:")
for lbl, n in label_counts.most_common():
    print(f"  {lbl:<20} {n:>4}")

print(f"\nTop 20 entidades mas frecuentes:")
for ent, n in entity_counts.most_common(20):
    print(f"  {ent:<45} {n:>4}")

print("\n=== EJEMPLOS ===")
for _, row in ner_df[ner_df["n_entities"] > 0].head(5).iterrows():
    doc = nlp(str(row["indications_text"]))
    print(f"\nTexto : {row['indications_text'][:120]}")
    for ent in doc.ents:
        print(f"  [{ent.label_}] '{ent.text}'")

# ── Gráficos ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"NER Biomedico ({model_used}) — Indicaciones FAERS", fontsize=13)

top20 = entity_counts.most_common(20)
axes[0].barh([e for e, _ in top20][::-1], [n for _, n in top20][::-1], color="mediumseagreen")
axes[0].set_title("Top 20 entidades extraidas")
axes[0].set_xlabel("Frecuencia")

axes[1].hist(ner_df["n_entities"], bins=20, color="steelblue", edgecolor="white")
axes[1].axvline(ner_df["n_entities"].mean(), color="red", linestyle="--",
                label=f"Media={ner_df['n_entities'].mean():.1f}")
axes[1].set_title("Entidades por reporte")
axes[1].set_xlabel("Numero de entidades")
axes[1].legend()

plt.tight_layout()
out = OUTPUTS_DIR / "figures" / "ner_extraction.png"
plt.savefig(out, dpi=150)
print(f"\nGrafico guardado: {out}")
print(f"CSV guardado    : {DATA_PROCESSED / 'ner_results.csv'}")
