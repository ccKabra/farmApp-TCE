"""
Demo de NER (Named Entity Recognition) con scispaCy.
Extrae entidades biomedicas de las indicaciones terapeuticas.
Unidad 5 de la materia — Extraccion de informacion.

Intenta cargar un modelo biomedico de scispaCy (en_core_sci_sm). Si no esta
instalado, cae a en_core_web_sm (spaCy base) y lo documenta como demo.
"""
import json
import pandas as pd
import spacy
from collections import Counter
from config import DATA_PROCESSED, OUTPUTS_DIR

# ── Cargar modelo ─────────────────────────────────────────────────────────────
print("Cargando modelo scispaCy...")
try:
    nlp = spacy.load("en_core_sci_sm")
    model_name = "en_core_sci_sm (scispaCy)"
except OSError:
    try:
        nlp = spacy.load("en_core_web_sm")
        model_name = "en_core_web_sm (spaCy base — fallback)"
    except OSError:
        print("ERROR: no se encontro ningun modelo de spaCy.")
        print("Instalar con: python -m spacy download en_core_web_sm")
        raise SystemExit(1)

print(f"Modelo cargado: {model_name}")

# ── Cargar dataset y elegir campo de indicaciones ─────────────────────────────
df = pd.read_csv(DATA_PROCESSED / "dataset.csv", dtype=str)

if "indications" in df.columns:
    source = "indications"
elif "indi_pt" in df.columns:
    source = "indi_pt"
elif "indication" in df.columns:
    source = "indication"
else:
    print("No se encontro columna de indicaciones.")
    print(f"Columnas disponibles: {list(df.columns)}")
    raise SystemExit(1)

# Las indicaciones vienen separadas por "|"; las separamos en terminos unicos
raw = df[source].dropna()
terms = sorted({t.strip() for row in raw for t in row.split("|") if t.strip()})
texts = terms[:500]  # max 500 para que no tarde demasiado

print(f"\nProcesando {len(texts)} indicaciones unicas con NER (campo: {source})...")

# ── Extraer entidades ─────────────────────────────────────────────────────────
all_entities = []
entity_counter = Counter()
label_counter = Counter()

for i, text in enumerate(texts):
    if not isinstance(text, str) or len(text.strip()) < 3:
        continue
    doc = nlp(text)
    for ent in doc.ents:
        all_entities.append({
            "texto_original": text,
            "entidad": ent.text,
            "etiqueta": ent.label_,
            "inicio": ent.start_char,
            "fin": ent.end_char,
        })
        entity_counter[ent.text] += 1
        label_counter[ent.label_] += 1
    if (i + 1) % 100 == 0:
        print(f"  Procesadas {i + 1}/{len(texts)} indicaciones...")

# ── Resultados ────────────────────────────────────────────────────────────────
print(f"\n=== Resultados NER ===")
print(f"Indicaciones procesadas: {len(texts)}")
print(f"Entidades encontradas:   {len(all_entities)}")
print(f"Entidades unicas:        {len(entity_counter)}")

print(f"\nTipos de entidades:")
for label, count in label_counter.most_common():
    print(f"  {label}: {count}")

print(f"\nTop 20 entidades mas frecuentes:")
for ent, count in entity_counter.most_common(20):
    print(f"  {ent}: {count}")

# ── Guardar CSV + resumen JSON ────────────────────────────────────────────────
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
ent_df = pd.DataFrame(all_entities)
csv_out = OUTPUTS_DIR / "ner_entities.csv"
ent_df.to_csv(csv_out, index=False)
print(f"\nEntidades guardadas en: {csv_out}")

summary = {
    "modelo": model_name,
    "campo_fuente": source,
    "indicaciones_procesadas": len(texts),
    "entidades_totales": len(all_entities),
    "entidades_unicas": len(entity_counter),
    "tipos_entidades": dict(label_counter),
    "top_20": dict(entity_counter.most_common(20)),
}
summary_out = OUTPUTS_DIR / "ner_summary.json"
with open(summary_out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"Resumen guardado en: {summary_out}")
