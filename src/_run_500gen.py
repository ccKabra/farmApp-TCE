"""Script temporal: correr AG con 500 gens capturando fitness/F1 sin tocar el modelo entregado.
Uso: python src/_run_500gen.py
Guarda resultados en outputs/_500gen_metrics.json y NO sobreescribe models/ga_model
(termina restaurando el backup 300gen).
"""
import json, shutil, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "models" / "ga_model_300backup"
MODEL = ROOT / "models" / "ga_model"

# monkey-patch config
sys.path.insert(0, str(ROOT / "src"))
import config
config.GA_GENERATIONS = 500

import train_ga
train_ga.GA_GENERATIONS = 500

# ejecutar
train_ga.main()

# capturar metricas del config.json que acabo de escribir
with open(MODEL / "config.json") as f:
    cfg = json.load(f)

metrics = {
    "generations": cfg["ga"]["generations"],
    "best_fitness_train": cfg["best_fitness_train"],
    "f1_macro_test": cfg["test_metrics"]["f1_macro"],
    "precision_macro": cfg["test_metrics"]["precision_macro"],
    "seconds": cfg["seconds"],
}
out = ROOT / "outputs" / "_500gen_metrics.json"
out.write_text(json.dumps(metrics, indent=2))
print("METRICS:", json.dumps(metrics))

# restaurar backup del modelo 300gen (el entregable real)
for name in ("weights.npy", "thresholds.npy", "config.json", "label_names.csv",
             "featurizer.json", "featurizer_tfidf.pkl"):
    src = BACKUP / name
    dst = MODEL / name
    if src.exists():
        shutil.copy2(src, dst)
print("BACKUP RESTORED")
