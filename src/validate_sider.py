"""
Etapa 6: Validación contra SIDER 4.1.

Version del proyecto farmApp-TCE: valida el clasificador entrenado por ALGORITMO
GENETICO (ga_model.load_ga_model) contra los efectos adversos conocidos por
farmaco en SIDER 4.1. Descarga SIDER desde su URL publica (si no esta en cache)
y compara predicciones vs. evidencia documentada. Produce el mismo
outputs/sider_validation.csv que consume la app.

SIDER usa MedDRA PT; FAERS usa MedDRA PT -> compatibles directamente.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import urllib.request
import gzip
from config import DATA_PROCESSED, DATA_RAW, OUTPUTS_DIR
from ga_model import load_ga_model

# ── Descargar SIDER 4.1 ───────────────────────────────────────────────────────
SIDER_URL = "http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz"
SIDER_LOCAL = DATA_RAW / "sider_meddra_all_se.tsv"

if not SIDER_LOCAL.exists():
    print(f"Descargando SIDER 4.1 desde {SIDER_URL}...")
    try:
        req = urllib.request.Request(SIDER_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = gzip.decompress(resp.read())
        with open(SIDER_LOCAL, "wb") as f:
            f.write(raw)
        print(f"SIDER guardado en: {SIDER_LOCAL}")
    except Exception as e:
        print(f"Error descargando SIDER: {e}")
        SIDER_LOCAL = None
else:
    print(f"SIDER ya existe: {SIDER_LOCAL}")

# ── Cargar SIDER ──────────────────────────────────────────────────────────────
if SIDER_LOCAL and SIDER_LOCAL.exists():
    sider = pd.read_csv(SIDER_LOCAL, sep="\t", header=None,
                        names=["stitch_flat", "stitch_stereo", "umls_se",
                               "meddra_type", "umls_meddra", "side_effect"])
    sider = sider[sider["meddra_type"] == "PT"].copy()
    sider["side_effect"] = sider["side_effect"].str.strip().str.title()
    print(f"SIDER cargado: {len(sider):,} entradas PT | {sider['stitch_flat'].nunique():,} farmacos")

    sider_effects = defaultdict(set)
    for _, row in sider.iterrows():
        sider_effects[row["stitch_flat"]].add(row["side_effect"])

    DRUG_NAMES_URL = "http://sideeffects.embl.de/media/download/drug_names.tsv"
    drug_names_local = DATA_RAW / "sider_drug_names.tsv"
    if not drug_names_local.exists():
        try:
            req2 = urllib.request.Request(DRUG_NAMES_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req2, timeout=60) as resp:
                with open(drug_names_local, "wb") as f:
                    f.write(resp.read())
            print("drug_names.tsv descargado.")
        except Exception as e:
            print(f"Error descargando drug_names: {e}")
            drug_names_local = None

    if drug_names_local and drug_names_local.exists():
        drug_names = pd.read_csv(drug_names_local, sep="\t", header=None,
                                 names=["stitch_id", "drug_name"])
        drug_names["drug_name_upper"] = drug_names["drug_name"].str.upper().str.strip()
        name_to_stitch = dict(zip(drug_names["drug_name_upper"], drug_names["stitch_id"]))
        print(f"drug_names.tsv: {len(name_to_stitch):,} farmacos mapeados")
    else:
        name_to_stitch = {}
else:
    sider_effects = {}
    name_to_stitch = {}

# ── Cargar modelo evolucionado por AG ─────────────────────────────────────────
print("\nCargando modelo evolucionado por AG...")
model, featurizer, label_names, thresholds = load_ga_model()
num_labels = len(label_names)
print(f"Modelo listo ({num_labels} etiquetas).")

# ── Predicción para fármacos con datos en SIDER ───────────────────────────────
df = pd.read_csv(DATA_PROCESSED / "dataset.csv", dtype=str)
top_drugs = [d for d, _ in Counter(
    [d for row in df["drug"].dropna() for d in row.split("|")]
).most_common(30)]

print(f"\nValidando {len(top_drugs)} farmacos frecuentes contra SIDER...")

results = []
for drug in top_drugs:
    stitch_id = name_to_stitch.get(drug.upper())
    sider_known = sider_effects.get(stitch_id, set()) if stitch_id else set()

    # Predecir para este farmaco (paciente generico: atributos unknown; SIDER
    # describe efectos del farmaco, no de un paciente particular).
    X = featurizer.transform_one(drugs=drug)
    probs = model.predict_proba(X)[0]

    predicted = {label_names[j] for j, p in enumerate(probs) if p >= thresholds[j]}
    sider_in_labels = sider_known & set(label_names)

    if sider_in_labels:
        tp = len(predicted & sider_in_labels)
        precision_d = tp / len(predicted) if predicted else 0
        recall_d = tp / len(sider_in_labels)
        f1_d = 2 * precision_d * recall_d / (precision_d + recall_d + 1e-9)
    else:
        precision_d = recall_d = f1_d = None

    results.append({
        "drug": drug,
        "stitch_found": stitch_id is not None,
        "sider_effects_in_vocab": len(sider_in_labels),
        "predicted": len(predicted),
        "tp": len(predicted & sider_in_labels) if sider_in_labels else 0,
        "precision_vs_sider": round(precision_d, 3) if precision_d is not None else None,
        "recall_vs_sider": round(recall_d, 3) if recall_d is not None else None,
        "f1_vs_sider": round(f1_d, 3) if f1_d is not None else None,
        "predicted_effects": "|".join(sorted(predicted))
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUTS_DIR / "sider_validation.csv", index=False)

valid = results_df.dropna(subset=["f1_vs_sider"])
print(f"\n=== VALIDACION SIDER (farmacos mapeados: {len(valid)}/{len(results_df)}) ===")
if len(valid) > 0:
    print(f"  F1 medio vs SIDER    : {valid['f1_vs_sider'].mean():.3f}")
    print(f"  Precision media      : {valid['precision_vs_sider'].mean():.3f}")
    print(f"  Recall medio         : {valid['recall_vs_sider'].mean():.3f}")
    print("\n  Por farmaco:")
    print(valid[["drug", "sider_effects_in_vocab", "predicted", "tp", "f1_vs_sider"]].to_string(index=False))
else:
    print("No se pudieron mapear farmacos FAERS->SIDER (nombres no coinciden directamente).")
    for _, row in results_df.head(10).iterrows():
        effects = row["predicted_effects"].split("|")[:5] if row["predicted_effects"] else []
        print(f"  {row['drug']:<35} -> {', '.join(effects)}")

print(f"\nResultados completos en: outputs/sider_validation.csv")

# ── Gráfico ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Validacion vs SIDER 4.1 (modelo AG)", fontsize=13)

if len(valid) > 0:
    valid_sorted = valid.sort_values("f1_vs_sider", ascending=True)
    axes[0].barh(valid_sorted["drug"], valid_sorted["f1_vs_sider"], color="mediumseagreen")
    axes[0].axvline(valid["f1_vs_sider"].mean(), color="red", linestyle="--",
                    label=f"Media={valid['f1_vs_sider'].mean():.3f}")
    axes[0].set_title("F1 por farmaco vs SIDER"); axes[0].legend()
else:
    axes[0].text(0.5, 0.5, "Sin mapeo FAERS->SIDER", ha="center", va="center")
    axes[0].set_title("F1 por farmaco vs SIDER")

pred_counts = results_df.set_index("drug")["predicted"].sort_values(ascending=True)
axes[1].barh(pred_counts.index, pred_counts.values, color="steelblue")
axes[1].set_title("Efectos adversos predichos por farmaco")
axes[1].set_xlabel("Cantidad de efectos predichos")

plt.tight_layout()
out = OUTPUTS_DIR / "figures" / "sider_validation.png"
plt.savefig(out, dpi=150)
print(f"Grafico guardado: {out}")
