"""
Etapa 1: Carga, limpieza y sampleo de datos FAERS.
Combina TODOS los trimestres presentes en data/raw (2025Q1..2026Q1) y samplea
SAMPLE_SIZE casos. Produce: data/processed/dataset.csv
"""

import pandas as pd
import numpy as np
from config import (
    faers_files, DATA_PROCESSED, SAMPLE_SIZE, RANDOM_SEED
)

SEP = "$"


def _read_all(prefix):
    """Lee y concatena todos los trimestres de un tipo de archivo FAERS."""
    files = faers_files(prefix)
    if not files:
        raise FileNotFoundError(f"No hay archivos {prefix}*.txt en data/raw")
    parts = []
    for f in files:
        d = pd.read_csv(f, sep=SEP, dtype=str, on_bad_lines="skip", low_memory=False)
        d.columns = d.columns.str.strip().str.lower()
        parts.append(d)
    print(f"  {prefix}: {len(files)} trimestres -> {sum(len(p) for p in parts):,} filas")
    return pd.concat(parts, ignore_index=True)


def load_demo():
    df = _read_all("DEMO")
    # Conservar solo columnas útiles
    cols = ["primaryid", "age", "age_cod", "sex", "wt", "wt_cod", "reporter_country"]
    df = df[[c for c in cols if c in df.columns]].copy()

    # Normalizar edad a años
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    age_map = {"DEC": 10, "YR": 1, "MON": 1/12, "WK": 1/52, "DY": 1/365, "HR": 1/8760}
    df["age_cod"] = df["age_cod"].str.strip().str.upper()
    df["age_years"] = df.apply(
        lambda r: r["age"] * age_map.get(r["age_cod"], 1) if pd.notna(r["age"]) else np.nan,
        axis=1
    )
    df = df.drop(columns=["age", "age_cod"])

    # Normalizar peso a kg
    df["wt"] = pd.to_numeric(df["wt"], errors="coerce")
    wt_map = {"KG": 1, "KGS": 1, "LBS": 0.4536, "GMS": 0.001}
    df["wt_cod"] = df["wt_cod"].str.strip().str.upper()
    df["weight_kg"] = df.apply(
        lambda r: r["wt"] * wt_map.get(r["wt_cod"], 1) if pd.notna(r["wt"]) else np.nan,
        axis=1
    )
    df = df.drop(columns=["wt", "wt_cod"])

    # Normalizar sexo
    df["sex"] = df["sex"].str.strip().str.upper().map({"M": "M", "F": "F"}).fillna("U")
    # Un caso puede repetirse entre trimestres (version actualizada); quedarse
    # con la mas reciente (los archivos vienen ordenados, 2026Q1 al final)
    return df.drop_duplicates("primaryid", keep="last")


def load_drugs():
    df = _read_all("DRUG")
    df["role_cod"] = df["role_cod"].str.strip().str.upper()
    # Usar ingrediente activo si existe, sino nombre comercial
    df["drug"] = df["prod_ai"].str.strip().fillna(df["drugname"].str.strip())
    df["drug"] = df["drug"].str.upper().str.strip()

    # Fármacos primarios sospechosos (PS = Primary Suspect)
    primary = (
        df[df["role_cod"] == "PS"]
        .groupby("primaryid")["drug"]
        .apply(lambda x: "|".join(sorted(set(x.dropna()))))
        .reset_index()
    )

    # Medicaciones previas/concomitantes del paciente (SS/C/I) — son las
    # "características individuales" que pide la consigna y antes se descartaban
    other = (
        df[df["role_cod"].isin(["SS", "C", "I"])]
        .groupby("primaryid")["drug"]
        .apply(lambda x: "|".join(sorted(set(x.dropna()))))
        .reset_index()
        .rename(columns={"drug": "other_drugs"})
    )

    merged = primary.merge(other, on="primaryid", how="left")

    # Quitar de other_drugs los farmacos que ya son sospechosos primarios
    # (FAERS repite el mismo farmaco con rol SS en algunos reportes)
    def dedupe(row):
        if pd.isna(row["other_drugs"]):
            return row["other_drugs"]
        ps = set(row["drug"].split("|"))
        rest = [d for d in row["other_drugs"].split("|") if d not in ps]
        return "|".join(rest) if rest else np.nan

    merged["other_drugs"] = merged.apply(dedupe, axis=1)
    return merged


def load_reactions():
    df = _read_all("REAC")
    df["pt"] = df["pt"].str.strip().str.title()
    reac = (
        df.groupby("primaryid")["pt"]
        .apply(lambda x: "|".join(sorted(set(x.dropna()))))
        .reset_index()
        .rename(columns={"pt": "reactions"})
    )
    return reac


def load_indications():
    df = _read_all("INDI")
    df["indi_pt"] = df["indi_pt"].str.strip().str.title()
    indi = (
        df.groupby("primaryid")["indi_pt"]
        .apply(lambda x: "|".join(sorted(set(x.dropna()))))
        .reset_index()
        .rename(columns={"indi_pt": "indications"})
    )
    return indi


def build_dataset():
    print("Cargando archivos FAERS...")
    demo  = load_demo()
    drugs = load_drugs()
    reac  = load_reactions()
    indi  = load_indications()

    print(f"  DEMO:  {len(demo):,} pacientes")
    print(f"  DRUGS: {len(drugs):,} casos con fármaco PS")
    print(f"  REAC:  {len(reac):,} casos con reacciones")
    print(f"  INDI:  {len(indi):,} casos con indicaciones")

    # Join por primaryid
    df = demo.merge(drugs, on="primaryid", how="inner")
    df = df.merge(reac,  on="primaryid", how="inner")
    df = df.merge(indi,  on="primaryid", how="left")

    # Eliminar filas sin información mínima
    df = df.dropna(subset=["drug", "reactions"])
    df = df[df["drug"].str.len() > 0]
    df = df[df["reactions"].str.len() > 0]

    print(f"\nCasos válidos antes de samplear: {len(df):,}")

    # Samplear
    n = min(SAMPLE_SIZE, len(df))
    df = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Muestra final: {n:,} casos")

    # Guardar
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "dataset.csv"
    df.to_csv(out, index=False)
    print(f"\nGuardado en: {out}")
    print(df.head(3).to_string())
    return df


if __name__ == "__main__":
    build_dataset()
