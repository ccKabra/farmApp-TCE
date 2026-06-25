"""
Genera la tabla de comparacion prediccion vs realidad sobre el 30% de test.

Version del proyecto farmApp-TCE: usa el clasificador entrenado por ALGORITMO
GENETICO (ga_model.load_ga_model) en lugar de BioBERT fine-tuned, pero produce
EXACTAMENTE el mismo outputs/test_cases.csv (misma estructura) que consume la app.

Corre el modelo sobre TODOS los casos de test (nunca vistos en entrenamiento) y
guarda outputs/test_cases.csv con:
  - atributos del paciente (edad, sexo, peso, farmaco, concomitantes, indicacion)
  - reacciones reales reportadas en FAERS (las que estan en el vocabulario de 98)
  - reacciones predichas por el modelo
  - aciertos (TP), falsos positivos (FP) y no detectadas (FN) por caso
"""

import pandas as pd
from config import OUTPUTS_DIR
from ga_model import load_ga_model
from data_split import load_split

df, label_names, train_idx, test_idx = load_split()
print(f"Dataset filtrado: {len(df):,} casos | Train: {len(train_idx):,} (70%) | Test: {len(test_idx):,} (30%)")

# Modelo evolucionado por AG: (model, featurizer, label_names, thresholds)
model, featurizer, model_labels, thresholds = load_ga_model()
assert model_labels == label_names, "El vocabulario de etiquetas no coincide con el del modelo"

df_test = df.iloc[test_idx].reset_index(drop=True)

print(f"Featurizando y prediciendo {len(df_test):,} casos de test...")
X_test = featurizer.transform(df_test)
probs = model.predict_proba(X_test)

rows = []
for i, (_, r) in enumerate(df_test.iterrows()):
    real = set(r["reaction_list"])
    pred = {label_names[j] for j in range(len(label_names)) if probs[i, j] >= thresholds[j]}
    tp, fp, fn = pred & real, pred - real, real - pred
    rows.append({
        "primaryid": r["primaryid"],
        "edad": r["age_years"],
        "sexo": r["sex"],
        "peso_kg": r["weight_kg"],
        "farmaco": r["drug"],
        "medicaciones_previas": r["other_drugs"],
        "indicaciones": r["indications"],
        "reacciones_reales": "|".join(sorted(real)),
        "reacciones_predichas": "|".join(sorted(pred)),
        "aciertos_TP": "|".join(sorted(tp)),
        "falsos_positivos_FP": "|".join(sorted(fp)),
        "no_detectadas_FN": "|".join(sorted(fn)),
        "n_reales": len(real), "n_predichas": len(pred), "n_aciertos": len(tp),
    })

out_df = pd.DataFrame(rows)
out = OUTPUTS_DIR / "test_cases.csv"
out_df.to_csv(out, index=False)

n_hit = (out_df["n_aciertos"] > 0).sum()
print(f"\nGuardado: {out}")
print(f"Casos de test: {len(out_df):,}")
print(f"Casos con al menos 1 acierto: {n_hit:,} ({n_hit/len(out_df):.1%})")
print(f"Promedio reales por caso: {out_df['n_reales'].mean():.2f} | predichas: {out_df['n_predichas'].mean():.2f}")
