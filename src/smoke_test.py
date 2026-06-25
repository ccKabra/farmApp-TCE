"""Smoke test: prediccion end-to-end con perfiles de paciente distintos.
Version farmApp-TCE: usa el clasificador evolucionado por Algoritmo Genetico."""
from ga_model import load_ga_model
from patient_text import build_patient_text

model, feat, labels, th = load_ga_model()


def pred(**kw):
    X = feat.transform_one(**kw)
    probs = model.predict_proba(X)[0]
    top = sorted(zip(labels, probs, th), key=lambda x: -x[1])[:10]
    n_pred = sum(1 for l, p, t in zip(labels, probs, th) if p >= t)
    print(build_patient_text(**kw))
    print(f"  [{n_pred} efectos predichos]")
    for l, p, t in top:
        print(f"   {l:<34} p={p:.3f} pred={p >= t}")
    print()


pred(age_years=72, sex="F", weight_kg=82, drugs="GABAPENTIN",
     other_drugs="TRAMADOL", indications="Pain")
pred(age_years=25, sex="M", drugs="ADALIMUMAB",
     other_drugs="METHOTREXATE|PREDNISONE", indications="Crohn'S Disease")
