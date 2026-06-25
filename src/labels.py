"""
Construccion UNICA del vocabulario de etiquetas (efectos adversos a predecir).

Centraliza dos decisiones que antes estaban duplicadas en cada script:
  1. frecuencia minima (MIN_REACTION_FREQ)
  2. exclusion de terminos MedDRA que NO son reacciones adversas farmacologicas

FAERS mezcla en el campo de reacciones muchos PT administrativos/contextuales:
errores de medicacion, problemas de dispositivo/producto, circunstancias
sociales (embarazo), desenlaces (hospitalizacion) y no-eventos. Si quedan en el
target, el modelo predice cosas como "Maternal Exposure During Pregnancy" para
un paciente de 72 anios. Aca se filtran.
"""

from collections import Counter
from config import MIN_REACTION_FREQ

# PTs de MedDRA que no son reacciones adversas farmacologicas.
NON_ADR_LABELS = {
    # Errores de medicacion / administracion
    "Inappropriate Schedule Of Product Administration",
    "Incorrect Dose Administered",
    "Product Dose Omission Issue",
    "Drug Dose Omission By Device",
    "Wrong Technique In Product Usage Process",
    "Overdose",
    "Accidental Exposure To Product",
    # Problemas de producto / dispositivo
    "Device Breakage",
    "Device Issue",
    "Product Storage Error",
    "Product Use Issue",
    "Product Use In Unapproved Indication",
    "Off Label Use",
    # Eficacia / administrativos
    "Drug Ineffective",
    "Therapeutic Product Effect Incomplete",
    "Therapy Interrupted",
    "No Adverse Event",
    # Circunstancias / contexto, no reaccion
    "Maternal Exposure During Pregnancy",
    # Desenlaces / inespecificos
    "Hospitalisation",
    "Illness",
    "General Physical Health Deterioration",
    "Condition Aggravated",
    "Toxicity To Various Agents",
}


def build_label_vocab(df):
    """
    Recibe el dataframe (columna 'reactions' con PTs separados por '|').
    Retorna (df_filtrado, label_names) donde:
      - label_names: lista ordenada de etiquetas validas (freq >= min y no excluidas)
      - df_filtrado: solo casos que conservan al menos una etiqueta valida
    Agrega/sobrescribe la columna 'reaction_list' con las etiquetas validas.
    """
    all_reac = [r for row in df["reactions"].dropna() for r in row.split("|")]
    freq_reac = {
        r for r, n in Counter(all_reac).items()
        if n >= MIN_REACTION_FREQ and r not in NON_ADR_LABELS
    }

    df = df.copy()
    df["reaction_list"] = df["reactions"].apply(
        lambda s: [r for r in s.split("|") if r in freq_reac] if isinstance(s, str) else []
    )
    df = df[df["reaction_list"].apply(len) > 0].reset_index(drop=True)
    return df, sorted(freq_reac)
