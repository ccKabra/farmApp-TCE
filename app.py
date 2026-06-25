r"""
Demo: Prediccion de efectos adversos de farmacos
Ejecutar: venv\Scripts\streamlit run app.py
"""

import sys
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from ga_model import load_ga_model
from patient_text import build_patient_text
from translations import translate_effect
from pipeline_story import pipeline_html

DATA_DIR = ROOT / "data" / "processed"

@st.cache_resource
def load_model():
    # Clasificador entrenado por Algoritmo Genetico (Computacion Evolutiva).
    # Retorna (model, featurizer, label_names, thresholds).
    return load_ga_model()

@st.cache_data
def get_vocab():
    """Vocabularios del dataset de entrenamiento para poblar el formulario."""
    df = pd.read_csv(DATA_DIR / "dataset.csv", dtype=str)
    from collections import Counter

    def top_values(col, n):
        if col not in df.columns:
            return []
        vals = [v for row in df[col].dropna() for v in row.split("|")]
        return [v for v, _ in Counter(vals).most_common(n)]

    return {
        "drugs": top_values("drug", 100),
        "other_drugs": top_values("other_drugs", 200),
        "indications": top_values("indications", 100),
    }

def predict(model, featurizer, label_names, thresholds,
            drug, age, sex, weight, other_drugs, indications):
    sex_code = {"Masculino": "M", "Femenino": "F", "No especificado": None}[sex]
    # MISMA representacion del paciente que en entrenamiento.
    # build_patient_text se usa solo para mostrar el texto; el vector que entra a
    # la red lo construye el featurizer (ga_features.py), que internamente usa el
    # mismo texto canonico -> entrenamiento e inferencia ven lo mismo.
    text = build_patient_text(
        age_years=age,
        sex=sex_code,
        weight_kg=weight if weight else None,
        drugs=drug,
        other_drugs="|".join(other_drugs),
        indications="|".join(indications),
    )
    X = featurizer.transform_one(
        age_years=age,
        sex=sex_code,
        weight_kg=weight if weight else None,
        drugs=drug,
        other_drugs="|".join(other_drugs),
        indications="|".join(indications),
    )
    probs = model.predict_proba(X)[0]
    results = [
        {"effect": label_names[i], "probability": float(probs[i]), "predicted": probs[i] >= thresholds[i]}
        for i in range(len(label_names))
    ]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results, text

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FarmApp - Prediccion de Efectos Adversos",
                   page_icon="Rx", layout="wide")

st.title("Sistema de Prediccion de Efectos Adversos")
st.caption("Clasificador entrenado por **Algoritmo Genetico** (Computacion Evolutiva) sobre FDA FAERS — Proyecto Tecnicas de Computacion Evolutiva")

with st.spinner("Cargando modelo (Algoritmo Genetico)..."):
    model, featurizer, label_names, thresholds = load_model()
    vocab = get_vocab()

st.success(f"Modelo listo | {len(label_names)} efectos adversos | Red de 1 capa oculta optimizada por AG (NumPy)")

@st.cache_data
def load_test_cases():
    path = ROOT / "outputs" / "test_cases.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype=str).fillna("")

@st.cache_data
def load_sider():
    """Carga SIDER 4.1 para validacion: (name_to_stitch, stitch_to_effects).
    Si los archivos no existen, devuelve diccionarios vacios (no rompe la app)."""
    raw_dir = ROOT / "data" / "raw"
    drug_names_path = raw_dir / "sider_drug_names.tsv"
    se_path = raw_dir / "sider_meddra_all_se.tsv"

    if not drug_names_path.exists() or not se_path.exists():
        return {}, {}

    # drug name -> stitch id
    dn = pd.read_csv(drug_names_path, sep="\t", header=None, names=["stitch_id", "drug_name"])
    name_to_stitch = dict(zip(dn["drug_name"].str.upper().str.strip(), dn["stitch_id"]))

    # stitch id -> set de efectos (solo PT = Preferred Terms), en Title Case
    se = pd.read_csv(se_path, sep="\t", header=None,
                     names=["stitch_flat", "stitch_stereo", "umls_se",
                            "meddra_type", "umls_meddra", "side_effect"])
    se_pt = se[se["meddra_type"] == "PT"].copy()
    se_pt["side_effect"] = se_pt["side_effect"].str.strip().str.title()

    from collections import defaultdict
    sider_effects = defaultdict(set)
    for stitch, effect in zip(se_pt["stitch_flat"], se_pt["side_effect"]):
        sider_effects[stitch].add(effect)

    return name_to_stitch, dict(sider_effects)

# Sufijos de sales farmaceuticas: FAERS usa "METFORMIN HYDROCHLORIDE",
# SIDER el ingrediente base "metformin". Se prueban para mejorar el mapeo.
SALT_SUFFIXES = [" HYDROCHLORIDE", " SODIUM", " POTASSIUM", " CALCIUM", " MESYLATE",
                 " MALEATE", " TARTRATE", " FUMARATE", " ACETATE", " SULFATE",
                 " CITRATE", " BESYLATE", " SUCCINATE"]

def map_drug_to_sider(drug_name, name_to_stitch):
    """Mapea un nombre de farmaco FAERS a su STITCH id en SIDER.
    Prueba el nombre completo y, si falla, va quitando sufijos de sales."""
    drug_upper = drug_name.upper().strip()
    stitch_id = name_to_stitch.get(drug_upper)
    if not stitch_id:
        for suffix in SALT_SUFFIXES:
            if drug_upper.endswith(suffix):
                stitch_id = name_to_stitch.get(drug_upper[:-len(suffix)].strip())
                if stitch_id:
                    break
    return stitch_id

tab_story, tab_pred, tab_test, tab_analysis = st.tabs([
    "Como lo hicimos",
    "Predecir paciente nuevo",
    "Casos reales de test (30%)",
    "Analisis del modelo",
])

# ── Pestania 0: el trasfondo del proyecto (para defender oralmente) ───────────
with tab_story:
    st.markdown(
        "El nucleo del proyecto es de **Computacion Evolutiva**: un **Algoritmo "
        "Genetico** optimiza los pesos de una red neuronal (sin retropropagacion) "
        "para predecir efectos adversos. Este es el recorrido, con foco en la "
        "evolucion: como se **codifica** el problema, como es el **genotipo/fenotipo**, "
        "como se **evalua el fitness**, que **operadores** se aplican y como **converge**."
    )

    st.info(
        "**Que parte es Computacion Evolutiva.** Etapas 5 a 11 del diagrama: "
        "genotipo real = pesos de la red (Modulo 3); poblacion + funcion de fitness "
        "F1-macro (Modulo 4); seleccion por torneo, cruce, mutacion gaussiana y "
        "elitismo (Modulo 2); control deterministico de parametros (Modulo 5); "
        "metricas de convergencia (Modulo 6); NSGA-II multi-objetivo (Modulo 7) y "
        "explicabilidad del individuo (Modulo 9). Las etapas 1 a 4 (datos FAERS y "
        "codificacion TF-IDF) son **soporte tecnico**, no la contribucion central."
    )

    components.html(pipeline_html(), height=640, scrolling=False)
    st.caption(
        "Cada etapa de arriba se corresponde con un modulo real de `src/`. "
        "La animacion avanza sola; pasa el mouse por una etapa para detenerte en ella."
    )

# ── Sidebar: mismos atributos con los que se entreno el modelo ────────────────
st.sidebar.header("Datos del paciente")

drug_input = st.sidebar.selectbox("Farmaco sospechoso", options=[""] + vocab["drugs"] + ["Otro (escribir abajo)"])
if drug_input == "Otro (escribir abajo)":
    drug_input = st.sidebar.text_input("Nombre del farmaco (ingrediente activo)").upper()

age_input = st.sidebar.slider("Edad (anos)", min_value=0, max_value=100, value=55)
sex_input = st.sidebar.selectbox("Sexo", ["No especificado", "Masculino", "Femenino"])
weight_input = st.sidebar.number_input("Peso (kg, 0 = desconocido)", min_value=0, max_value=300, value=0)
other_drugs_input = st.sidebar.multiselect(
    "Medicaciones previas / concomitantes", options=vocab["other_drugs"])
indications_input = st.sidebar.multiselect(
    "Indicaciones (motivo del tratamiento)", options=vocab["indications"])
indication_free = st.sidebar.text_input("Otra indicacion (opcional)",
                                        placeholder="ej: Type 2 Diabetes Mellitus")
if indication_free.strip():
    indications_input = indications_input + [indication_free.strip().title()]

st.sidebar.markdown("---")
sensitivity = st.sidebar.slider(
    "Sensibilidad del modelo", 0.1, 1.0, 0.5, 0.05,
    help="Mas alto = mas efectos predichos (mas recall, menos precision). "
         "0.5 = umbrales calibrados originales.")

predict_btn = st.sidebar.button("Predecir efectos adversos", type="primary")

# ── Pestania 1: Prediccion ────────────────────────────────────────────────────
with tab_pred:
    if predict_btn and drug_input:
        with st.spinner("Analizando..."):
            # Sensibilidad: factor sobre los umbrales. 0.5 -> factor 1.0 (umbral
            # original); mas sensibilidad -> umbral mas bajo -> mas efectos predichos.
            thresholds_adj = np.clip(thresholds * (1.5 - sensitivity), 0.0, 1.0)
            results, input_text = predict(model, featurizer, label_names, thresholds_adj,
                                           drug_input, age_input, sex_input, weight_input,
                                           other_drugs_input, indications_input)

        predicted = [r for r in results if r["predicted"]]

        col1, col2, col3 = st.columns(3)
        col1.metric("Farmaco", drug_input)
        col2.metric("Efectos adversos predichos", len(predicted))
        col3.metric("Confianza maxima", f"{results[0]['probability']:.1%}")

        st.markdown("---")
        col_a, col_b = st.columns([3, 2])

        with col_a:
            st.subheader("Efectos adversos predichos")
            if predicted:
                pred_df = pd.DataFrame(predicted)[["effect", "probability"]]
                pred_df["effect"] = pred_df["effect"].apply(translate_effect)
                pred_df["probability"] = pred_df["probability"].map("{:.1%}".format)
                pred_df.columns = ["Efecto adverso", "Probabilidad"]
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
            else:
                st.info("No se predijeron efectos adversos con los umbrales actuales.")

        with col_b:
            st.subheader("Top 15 probabilidades")
            top15 = results[:15]
            fig = go.Figure(go.Bar(
                x=[r["probability"] for r in top15],
                y=[translate_effect(r["effect"]) for r in top15],
                orientation="h",
                marker_color=["#e74c3c" if r["predicted"] else "#3498db" for r in top15],
                text=[f"{r['probability']:.1%}" for r in top15],
                textposition="outside"
            ))
            fig.update_layout(
                height=450, margin=dict(l=0, r=40, t=10, b=10),
                xaxis_title="Probabilidad", yaxis=dict(autorange="reversed"),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Texto de entrada al modelo"):
            st.code(input_text)

        with st.expander("Tabla completa de probabilidades"):
            all_df = pd.DataFrame(results)[["effect", "probability", "predicted"]]
            all_df["effect"] = all_df["effect"].apply(translate_effect)
            all_df["probability"] = all_df["probability"].map("{:.3f}".format)
            all_df.columns = ["Efecto adverso", "Probabilidad", "Predicho"]
            st.dataframe(all_df, use_container_width=True, hide_index=True)

        # ── Validacion contra SIDER 4.1 ──────────────────────────────────────
        st.markdown("---")
        st.subheader("Validacion contra SIDER 4.1")

        name_to_stitch, sider_effects = load_sider()

        if not name_to_stitch:
            st.warning("Archivos SIDER no encontrados en data/raw/. Coloca "
                       "sider_drug_names.tsv y sider_meddra_all_se.tsv ahi.")
        else:
            stitch_id = map_drug_to_sider(drug_input, name_to_stitch)

            if not stitch_id:
                st.info(f"'{drug_input}' no se encontro en SIDER 4.1. No todos los "
                        "farmacos estan en la base de datos (1430 farmacos disponibles).")
            else:
                known_effects = sider_effects.get(stitch_id, set())
                label_set = set(label_names)
                known_in_vocab = known_effects & label_set

                predicted_effects = set(r["effect"] for r in results if r["predicted"])
                top15_effects = set(r["effect"] for r in results[:15])

                if not known_in_vocab:
                    st.info(f"SIDER tiene {len(known_effects)} registros para este "
                            f"farmaco pero ninguno coincide con las {len(label_names)} "
                            "etiquetas del modelo.")
                else:
                    tp_pred = predicted_effects & known_in_vocab
                    tp_top15 = top15_effects & known_in_vocab

                    precision_pred = len(tp_pred) / len(predicted_effects) if predicted_effects else 0
                    recall_pred = len(tp_pred) / len(known_in_vocab)
                    precision_top15 = len(tp_top15) / len(top15_effects) if top15_effects else 0
                    recall_top15 = len(tp_top15) / len(known_in_vocab)

                    st.markdown(
                        f"**SIDER documenta {len(known_effects)} efectos para este farmaco, "
                        f"de los cuales {len(known_in_vocab)} estan en las "
                        f"{len(label_names)} etiquetas del modelo.**")

                    met1, met2, met3, met4 = st.columns(4)
                    met1.metric("Precision (predichos)", f"{precision_pred:.0%}")
                    met2.metric("Recall (predichos)", f"{recall_pred:.0%}")
                    met3.metric("Precision (top 15)", f"{precision_top15:.0%}")
                    met4.metric("Recall (top 15)", f"{recall_top15:.0%}")

                    st.markdown("#### Comparacion detallada — Top 15 del modelo vs SIDER")
                    comparison_rows = []
                    for r in results[:15]:
                        effect = r["effect"]
                        in_sider = effect in known_in_vocab
                        was_predicted = r["predicted"]
                        comparison_rows.append({
                            "Efecto": translate_effect(effect),
                            "Probabilidad": f"{r['probability']:.1%}",
                            "Predicho": "Si" if was_predicted else "No",
                            "En SIDER": "Confirmado" if in_sider else "No registrado",
                            "Resultado": "Acierto" if (was_predicted and in_sider) else
                                         "Probable" if (not was_predicted and in_sider) else
                                         "Sin evidencia SIDER" if was_predicted else "—",
                        })
                    st.dataframe(pd.DataFrame(comparison_rows),
                                 use_container_width=True, hide_index=True)

                    missed = known_in_vocab - top15_effects
                    if missed:
                        with st.expander(f"Efectos en SIDER no detectados por el modelo ({len(missed)})"):
                            missed_with_prob = [
                                {"Efecto": translate_effect(r["effect"]), "Probabilidad": f"{r['probability']:.1%}"}
                                for r in results if r["effect"] in missed
                            ]
                            st.dataframe(pd.DataFrame(missed_with_prob),
                                         use_container_width=True, hide_index=True)
                            st.caption("Estos efectos estan documentados en SIDER pero el "
                                       "modelo les asigno baja probabilidad.")

    elif predict_btn and not drug_input:
        st.warning("Por favor selecciona o escribe un farmaco.")
    else:
        st.info("Selecciona un farmaco y presiona 'Predecir efectos adversos' para comenzar.")

# ── Pestania 2: Casos reales de test (30% nunca visto por el modelo) ─────────
with tab_test:
    tc = load_test_cases()
    if tc is None:
        st.warning("No existe outputs/test_cases.csv. Generarlo con: venv\\Scripts\\python.exe src\\eval_test_cases.py")
    else:
        st.markdown(
            "Estos casos son el **30% de test** (mismo split 70/30, semilla 42, "
            "que se uso para entrenar): el modelo **nunca los vio**. Para cada paciente "
            "real de FAERS se muestran las reacciones reportadas vs. las predichas."
        )

        tc_num = tc.copy()
        tc_num["n_aciertos"] = pd.to_numeric(tc_num["n_aciertos"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Casos de test", f"{len(tc):,}")
        c2.metric("Con al menos 1 acierto", f"{(tc_num['n_aciertos'] > 0).mean():.1%}")
        c3.metric("Split", "70% train / 30% test")

        # Filtros
        fcol1, fcol2 = st.columns([2, 1])
        all_drugs_test = sorted({d for row in tc["farmaco"] for d in row.split("|") if d})
        drug_filter = fcol1.selectbox("Filtrar por farmaco", ["(todos)"] + all_drugs_test)
        only_hits = fcol2.checkbox("Solo casos con aciertos", value=False)

        view = tc_num
        if drug_filter != "(todos)":
            view = view[view["farmaco"].str.contains(drug_filter, regex=False)]
        if only_hits:
            view = view[view["n_aciertos"] > 0]

        # Traducir SOLO las columnas de reacciones (efectos). Farmaco e indicaciones
        # quedan en ingles (nombres internacionales / no son efectos del vocabulario).
        def translate_pipe(s):
            return "|".join(translate_effect(e) for e in s.split("|")) if s else s

        view_show = view.copy()
        view_show["reacciones_reales"] = view_show["reacciones_reales"].apply(translate_pipe)
        view_show["reacciones_predichas"] = view_show["reacciones_predichas"].apply(translate_pipe)

        st.dataframe(
            view_show[["primaryid", "edad", "sexo", "peso_kg", "farmaco",
                       "medicaciones_previas", "indicaciones",
                       "reacciones_reales", "reacciones_predichas", "n_aciertos"]],
            use_container_width=True, hide_index=True, height=350
        )

        # Detalle de un caso
        st.markdown("### Detalle de un caso")
        case_id = st.selectbox("Elegir caso (primaryid)", view["primaryid"].tolist())
        if case_id:
            row = tc[tc["primaryid"] == case_id].iloc[0]
            st.markdown(
                f"**Paciente:** edad {row['edad'] or '?'} | sexo {row['sexo'] or '?'} | "
                f"peso {row['peso_kg'] or '?'} kg  \n"
                f"**Farmaco sospechoso:** {row['farmaco']}  \n"
                f"**Medicaciones previas:** {row['medicaciones_previas'] or '—'}  \n"
                f"**Indicaciones:** {row['indicaciones'] or '—'}"
            )
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown("#### Aciertos (TP)")
                for e in (row["aciertos_TP"].split("|") if row["aciertos_TP"] else []):
                    st.markdown(f"- {translate_effect(e)}")
                if not row["aciertos_TP"]:
                    st.caption("Ninguno")
            with d2:
                st.markdown("#### No detectadas (FN)")
                for e in (row["no_detectadas_FN"].split("|") if row["no_detectadas_FN"] else []):
                    st.markdown(f"- {translate_effect(e)}")
                if not row["no_detectadas_FN"]:
                    st.caption("Ninguna")
            with d3:
                st.markdown("#### Predichas de mas (FP)")
                for e in (row["falsos_positivos_FP"].split("|") if row["falsos_positivos_FP"] else []):
                    st.markdown(f"- {translate_effect(e)}")
                if not row["falsos_positivos_FP"]:
                    st.caption("Ninguna")

# ── Pestania 3: Analisis del modelo (temas de la materia) ─────────────────────
with tab_analysis:
    st.markdown("Resultados de evaluacion y conceptos de **Computacion Evolutiva** "
                "aplicados en el proyecto. Todos los numeros provienen de correr los "
                "scripts de `src/` (el modelo de este proyecto se entrena con "
                "`src/train_ga.py`).")

    @st.cache_data
    def load_ga_artifacts():
        cfg_p = ROOT / "models" / "ga_model" / "config.json"
        evo_p = ROOT / "outputs" / "ga_evolution.csv"
        cfg = json.load(open(cfg_p, encoding="utf-8")) if cfg_p.exists() else {}
        evo = pd.read_csv(evo_p) if evo_p.exists() else None
        return cfg, evo

    ga_cfg, ga_evo = load_ga_artifacts()

    # ── Seccion 0: El Algoritmo Genetico (modelo de este proyecto) ────────────
    st.markdown("---")
    st.subheader("0 · Algoritmo Genetico — el modelo de este proyecto")
    if ga_cfg:
        ga = ga_cfg.get("ga", {})
        m = ga_cfg.get("test_metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1 macro (test)", f"{m.get('f1_macro', 0):.4f}")
        c2.metric("Precision macro", f"{m.get('precision_macro', 0):.4f}")
        c3.metric("Recall macro", f"{m.get('recall_macro', 0):.4f}")
        c4.metric("Genoma (pesos)", f"{ga_cfg.get('n_params', 0):,}")
        st.caption(
            f"Fenotipo: red {ga_cfg.get('input_dim','?')}->{ga_cfg.get('hidden_dim','?')}"
            f"->{ga_cfg.get('output_dim','?')} (1 capa oculta, ReLU + sigmoide). "
            f"AG: poblacion {ga.get('pop_size','?')}, {ga.get('generations','?')} generaciones, "
            f"seleccion por torneo (k={ga.get('tournament_k','?')}), cruce {ga.get('crossover','?')} "
            f"(pc={ga.get('p_crossover','?')}), mutacion gaussiana (pm={ga.get('p_mutation','?')}, "
            f"sigma {ga.get('sigma0','?')}->{ga.get('sigma_end','?')}), elitismo {ga.get('elitism','?')}."
        )
        if ga_evo is not None:
            fig_ga = go.Figure()
            fig_ga.add_trace(go.Scatter(x=ga_evo["gen"], y=ga_evo["best"], mode="lines",
                                        name="Mejor fitness", line=dict(color="#2ecc71")))
            fig_ga.add_trace(go.Scatter(x=ga_evo["gen"], y=ga_evo["mean"], mode="lines",
                                        name="Fitness medio", line=dict(color="#3498db")))
            fig_ga.update_layout(height=360, xaxis_title="Generacion",
                                 yaxis_title="F1-macro (fitness)",
                                 margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_ga, use_container_width=True)
        st.info(
            "El clasificador NO se entrena por retropropagacion: sus pesos son un "
            "**cromosoma de numeros reales** que un Algoritmo Genetico optimiza maximizando "
            "F1-macro (seleccion por torneo + cruce + mutacion gaussiana + elitismo). La "
            "curva muestra la convergencia: el mejor individuo y la media de la poblacion "
            "suben generacion a generacion. Conceptos de la materia: genotipo no binario "
            "(Modulo 3), seleccion y fitness (Modulo 4), control de parametros (Modulo 5) "
            "y metricas de convergencia (Modulo 6).")
    else:
        st.warning("No se encontro models/ga_model/config.json. Entrena el modelo con: "
                   "python src/train_ga.py")

    # ── Seccion 1: Comparativa de modelos ─────────────────────────────────────
    st.markdown("---")
    st.subheader("1 · Comparativa: AG vs baselines historicos")
    st.caption("El modelo del proyecto es el **Algoritmo Genetico** (primera fila). "
               "Los demas son baselines de una iteracion previa de Mineria de Texto, "
               "incluidos solo como referencia/contraste, no como contribucion.")
    _gm = ga_cfg.get("test_metrics", {}) if ga_cfg else {}
    ga_macro = round(_gm.get("f1_macro", 0.073), 4)
    ga_micro = round(_gm.get("f1_micro", 0.059), 4)
    ga_samp = round(_gm.get("f1_samples", 0.053), 4)
    ga_ham = round(_gm.get("hamming_loss", 0.283), 4)
    comparativa = pd.DataFrame({
        "Modelo": [
            "Algoritmo Genetico (red 1 capa) — ESTE PROYECTO",
            "Naive Bayes (MultinomialNB)",
            "KNN (k=5, coseno)",
            "Random Forest + TF-IDF",
            "Random Forest + BioBERT embeddings",
            "BioBERT fine-tuning + umbral optimo",
        ],
        "F1 macro":     [ga_macro, 0.0001, 0.0006, 0.0389, 0.130, 0.128],
        "F1 micro":     [ga_micro, 0.0006, 0.0012, 0.0485, 0.150, 0.106],
        "F1 samples":   [ga_samp, 0.0006, 0.0006, 0.0455, 0.127, 0.098],
        "Hamming loss": [ga_ham, 0.0217, 0.0218, 0.2447, 0.059, 0.059],
        "Rol en el proyecto": [
            "★ Modelo evolutivo (este proyecto)",
            "Baseline historico (Mineria de Texto)",
            "Baseline historico (Mineria de Texto)",
            "Baseline historico (Mineria de Texto)",
            "Baseline historico (Mineria de Texto)",
            "Baseline historico (Mineria de Texto)",
        ],
    })
    st.dataframe(comparativa, use_container_width=True, hide_index=True)

    f1_models = ["AG (este proyecto)", "Naive Bayes", "KNN (k=5)", "RF + TF-IDF", "RF + BioBERT"]
    f1_values = [ga_macro, 0.0001, 0.0006, 0.0389, 0.130]
    fig_cmp = go.Figure(go.Bar(
        x=f1_models, y=f1_values,
        marker_color=["#e74c3c", "#bdc3c7", "#95a5a6", "#3498db", "#2ecc71"],
        text=[f"{v:.4f}" for v in f1_values], textposition="outside",
    ))
    fig_cmp.update_layout(height=360, yaxis_title="F1 macro",
                          margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
    st.plotly_chart(fig_cmp, use_container_width=True)
    st.info(
        "El **Algoritmo Genetico** supera ampliamente a los baselines clasicos (Naive "
        "Bayes, KNN) y se acerca a Random Forest, demostrando que la evolucion de pesos "
        "puede entrenar un clasificador multi-label competitivo SIN retropropagacion. "
        "Los baselines de Mineria de Texto (RF, BioBERT) se muestran solo como contraste "
        "de una metodologia distinta; el foco del proyecto es la Computacion Evolutiva. "
        "El F1 absoluto bajo es inherente al problema (98 etiquetas, densidad ~2%) y "
        "comun a todos los modelos.")

    # ── Seccion 2: Desbalance de clases (condiciona el fitness del AG) ────────
    st.markdown("---")
    st.subheader("2 · Desbalance de clases (condiciona el fitness del AG)")
    from data_split import load_split as _load_split
    _df, _labels, _, _ = _load_split()
    counts = {}
    for reacs in _df["reaction_list"]:
        for r in reacs:
            counts[r] = counts.get(r, 0) + 1
    label_counts = pd.Series(counts).sort_values(ascending=False)
    top15, bottom15 = label_counts.head(15), label_counts.tail(15)
    cold1, cold2 = st.columns(2)
    with cold1:
        fig_t = go.Figure(go.Bar(
            x=top15.values[::-1],
            y=[translate_effect(l) for l in top15.index[::-1]],
            orientation="h", marker_color="#2ecc71"))
        fig_t.update_layout(height=420, title="Top 15 mas frecuentes",
                            margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_t, use_container_width=True)
    with cold2:
        fig_b = go.Figure(go.Bar(
            x=bottom15.values[::-1],
            y=[translate_effect(l) for l in bottom15.index[::-1]],
            orientation="h", marker_color="#e67e22"))
        fig_b.update_layout(height=420, title="15 menos frecuentes",
                            margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_b, use_container_width=True)
    st.info(
        "El dataset presenta un desbalance severo: las etiquetas mas frecuentes superan "
        "los 200 casos mientras que las menos frecuentes apenas llegan a 50. Esto "
        "**condiciona el diseno del fitness** del AG: medir F1-macro con un umbral por "
        "etiqueta (en vez de accuracy o un umbral global de 0.5) es lo que le da al "
        "Algoritmo Genetico una senal util para optimizar pese al desbalance.")

    # ── Seccion 3: Extras de Computacion Evolutiva (Modulos 7 y 9) ────────────
    st.markdown("---")
    st.subheader("3 · Analisis evolutivo avanzado (XAI + NSGA-II)")

    fi_png = ROOT / "outputs" / "figures" / "ga_feature_importance.png"
    fi_csv = ROOT / "outputs" / "ga_feature_importance.csv"
    pf_png = ROOT / "outputs" / "figures" / "nsga2_pareto_front.png"
    pf_csv = ROOT / "outputs" / "nsga2_pareto.csv"

    st.markdown("**XAI — Importancia de features (Modulo 9).** Que terminos del texto "
                "del paciente y que variables pesan mas en el modelo evolucionado, "
                "derivado de las magnitudes de los pesos. Generar con `python src/ga_explain.py`.")
    if fi_png.exists():
        cga1, cga2 = st.columns([3, 2])
        with cga1:
            st.image(str(fi_png), use_container_width=True)
        with cga2:
            if fi_csv.exists():
                st.dataframe(pd.read_csv(fi_csv).head(15), use_container_width=True,
                             hide_index=True)
    else:
        st.caption("Aun no generado (corre src/ga_explain.py).")

    st.markdown("**NSGA-II — Frente de Pareto Precision vs Recall (Modulo 7).** En vez "
                "de un solo F1, se evoluciona el conjunto de soluciones no dominadas que "
                "balancean precision y recall. Generar con `python src/train_ga_nsga2.py`.")
    if pf_png.exists():
        st.image(str(pf_png), use_container_width=True)
        if pf_csv.exists():
            with st.expander("Soluciones del frente de Pareto"):
                st.dataframe(pd.read_csv(pf_csv), use_container_width=True, hide_index=True)
    else:
        st.caption("Aun no generado (corre src/train_ga_nsga2.py).")
    st.info(
        "Ambos extras se apoyan en el mismo modelo evolucionado: la importancia de "
        "features lo hace interpretable (XAI) y NSGA-II muestra que precision y recall "
        "son objetivos en conflicto, evolucionando todo el frente de compromiso.")

