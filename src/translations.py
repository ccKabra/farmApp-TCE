"""
Traduccion de la capa de presentacion (UI) ingles -> espanol.

Los efectos adversos del modelo son terminos MedDRA en ingles. Aqui se traducen
SOLO para mostrarlos al usuario: internamente (label_names, datos, metricas)
todo sigue en ingles. Cubre las 98 etiquetas de
models/biobert_finetuned/label_names.csv.

Si una etiqueta no esta en el diccionario, translate_effect() devuelve el
original en ingles (no rompe nada).
"""

# Efectos adversos MedDRA (ingles -> espanol). Mismas claves que label_names.
EFFECTS_EN_ES = {
    "Abdominal Discomfort": "Malestar abdominal",
    "Abdominal Distension": "Distension abdominal",
    "Abdominal Pain": "Dolor abdominal",
    "Abdominal Pain Upper": "Dolor abdominal superior",
    "Acute Kidney Injury": "Lesion renal aguda",
    "Adverse Drug Reaction": "Reaccion adversa al medicamento",
    "Alopecia": "Alopecia",
    "Anaemia": "Anemia",
    "Anxiety": "Ansiedad",
    "Arthralgia": "Artralgia",
    "Asthenia": "Astenia",
    "Asthma": "Asma",
    "Back Pain": "Dolor de espalda",
    "Blood Glucose Increased": "Glucosa en sangre aumentada",
    "Blood Pressure Increased": "Presion arterial aumentada",
    "Cerebrovascular Accident": "Accidente cerebrovascular",
    "Chest Discomfort": "Malestar toracico",
    "Chest Pain": "Dolor toracico",
    "Chills": "Escalofrios",
    "Confusional State": "Estado confusional",
    "Constipation": "Estrenimiento",
    "Contusion": "Contusion",
    "Cough": "Tos",
    "Covid-19": "Covid-19",
    "Crohn'S Disease": "Enfermedad de Crohn",
    "Death": "Muerte",
    "Decreased Appetite": "Disminucion del apetito",
    "Dehydration": "Deshidratacion",
    "Depression": "Depresion",
    "Dermatitis Atopic": "Dermatitis atopica",
    "Device Delivery System Issue": "Problema del sistema de administracion del dispositivo",
    "Device Malfunction": "Mal funcionamiento del dispositivo",
    "Diarrhoea": "Diarrea",
    "Disease Progression": "Progresion de la enfermedad",
    "Dizziness": "Mareos",
    "Drug Hypersensitivity": "Hipersensibilidad al farmaco",
    "Drug Interaction": "Interaccion medicamentosa",
    "Drug Intolerance": "Intolerancia al farmaco",
    "Dry Skin": "Piel seca",
    "Dyspepsia": "Dispepsia",
    "Dyspnoea": "Disnea",
    "Eczema": "Eccema",
    "Erythema": "Eritema",
    "Exposure Via Skin Contact": "Exposicion por contacto cutaneo",
    "Fall": "Caida",
    "Fatigue": "Fatiga",
    "Feeling Abnormal": "Sensacion anormal",
    "Gait Disturbance": "Alteracion de la marcha",
    "Gastrointestinal Disorder": "Trastorno gastrointestinal",
    "Haemorrhage": "Hemorragia",
    "Headache": "Cefalea",
    "Hot Flush": "Sofoco",
    "Hypersensitivity": "Hipersensibilidad",
    "Hypertension": "Hipertension",
    "Hypoaesthesia": "Hipoestesia",
    "Hypotension": "Hipotension",
    "Infection": "Infeccion",
    "Influenza": "Gripe",
    "Infusion Related Reaction": "Reaccion relacionada con la infusion",
    "Injection Site Erythema": "Eritema en el sitio de inyeccion",
    "Injection Site Pain": "Dolor en el sitio de inyeccion",
    "Insomnia": "Insomnio",
    "Intentional Product Use Issue": "Problema de uso intencional del producto",
    "Joint Swelling": "Hinchazon articular",
    "Malaise": "Malestar general",
    "Malignant Neoplasm Progression": "Progresion de neoplasia maligna",
    "Memory Impairment": "Deterioro de la memoria",
    "Migraine": "Migrana",
    "Muscle Spasms": "Espasmos musculares",
    "Myalgia": "Mialgia",
    "Myelosuppression": "Mielosupresion",
    "Nasopharyngitis": "Nasofaringitis",
    "Nausea": "Nauseas",
    "Neuropathy Peripheral": "Neuropatia periferica",
    "Neutropenia": "Neutropenia",
    "Pain": "Dolor",
    "Pain In Extremity": "Dolor en extremidades",
    "Palpitations": "Palpitaciones",
    "Paraesthesia": "Parestesia",
    "Peripheral Swelling": "Hinchazon periferica",
    "Pneumonia": "Neumonia",
    "Pruritus": "Prurito",
    "Psoriasis": "Psoriasis",
    "Pyrexia": "Fiebre",
    "Rash": "Erupcion cutanea",
    "Rheumatoid Arthritis": "Artritis reumatoide",
    "Seizure": "Convulsion",
    "Sinusitis": "Sinusitis",
    "Somnolence": "Somnolencia",
    "Tremor": "Temblor",
    "Urinary Tract Infection": "Infeccion urinaria",
    "Urticaria": "Urticaria",
    "Vision Blurred": "Vision borrosa",
    "Visual Impairment": "Deterioro visual",
    "Vomiting": "Vomitos",
    "Weight Decreased": "Perdida de peso",
    "Weight Increased": "Aumento de peso",
    "White Blood Cell Count Decreased": "Recuento de globulos blancos disminuido",
}


def translate_effect(name_en):
    """Traduce un efecto adverso ingles -> espanol. Si no hay traduccion,
    devuelve el original (no rompe la UI)."""
    return EFFECTS_EN_ES.get(name_en, name_en)


def translate_effects_list(effects_en):
    """Traduce una lista de efectos."""
    return [translate_effect(e) for e in effects_en]
