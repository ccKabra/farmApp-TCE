"""
Representacion canonica del paciente como texto para BioBERT.

ESTA ES LA UNICA DEFINICION del formato de entrada al modelo.
La usan: train_biobert_finetune.py, tune_threshold.py, validate_sider.py y app.py.
Si entrenamiento e inferencia usan formatos distintos, el modelo recibe en
produccion una distribucion de texto que nunca vio -> predicciones invalidas.
"""

import pandas as pd

MAX_LEN = 160  # tokens; el texto enriquecido es mas largo que "drug + indication"


def build_patient_text(age_years=None, sex=None, weight_kg=None,
                       drugs="", other_drugs="", indications=""):
    """
    Construye el texto de entrada al modelo a partir de los atributos del paciente.
    Acepta los mismos campos que tiene cada caso de FAERS en dataset.csv.

    - age_years: float o None
    - sex: "M" / "F" / otro -> unknown
    - weight_kg: float o None
    - drugs: farmacos sospechosos, separados por "|"
    - other_drugs: medicaciones concomitantes/previas, separadas por "|"
    - indications: indicaciones, separadas por "|"
    """
    age = f"age {int(float(age_years))} years" if pd.notna(age_years) else "age unknown"
    sex_str = {"M": "sex male", "F": "sex female"}.get(
        str(sex).strip().upper() if pd.notna(sex) else "", "sex unknown")
    wt = f"weight {int(float(weight_kg))} kg" if pd.notna(weight_kg) else "weight unknown"

    def clean(s, max_chars):
        if pd.isna(s) or not str(s).strip():
            return "unknown"
        return str(s).replace("|", ", ").strip()[:max_chars]

    return (
        f"patient: {age}, {sex_str}, {wt}. "
        f"drug: {clean(drugs, 120)}. "
        f"other medications: {clean(other_drugs, 150)}. "
        f"indication: {clean(indications, 200)}"
    )


def row_to_text(row):
    """Convierte una fila de dataset.csv al texto canonico."""
    return build_patient_text(
        age_years=row.get("age_years"),
        sex=row.get("sex"),
        weight_kg=row.get("weight_kg"),
        drugs=row.get("drug"),
        other_drugs=row.get("other_drugs"),
        indications=row.get("indications"),
    )
