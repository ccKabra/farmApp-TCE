"""
Narrativa visual del pipeline para la defensa oral.

Genera un diagrama HTML/CSS animado que cuenta el trasfondo del proyecto: de
donde salen los datos, que les hacemos, como los ordenamos, en que concepto de
la materia nos apoyamos, y -lo importante- un ejemplo real que va recorriendo
cada etapa para mostrar el antes y el despues y como queda en codigo para que
el modelo lo entienda. Pensado para proyectarse en clase en vez de leer codigo.

Ejemplo conductor (caso real de dataset.csv, primaryid 254258761):
    mujer, 74 anios, farmaco LENALIDOMIDE, indicacion "Plasma Cell Myeloma".

La app (app.py) solo lo embebe con st.components.v1.html(pipeline_html()).
Cada etapa se corresponde 1:1 con un modulo real de src/ (campo 'archivo').
"""


# ── Helpers para dibujar el "antes y despues" de cada ejemplo ─────────────────
def _box(cap, code, kind="proc"):
    """Una cajita con etiqueta (cap) y contenido en mono (code). kind = color."""
    return f'<div class="box {kind}"><span class="cap">{cap}</span><code>{code}</code></div>'


def _flow(*boxes, note=None):
    """Encadena cajas con flechas; note = aclaracion debajo del flujo."""
    inner = '<span class="arr">&rarr;</span>'.join(boxes)
    extra = f'<div class="fnote">{note}</div>' if note else ""
    return f'<div class="flow">{inner}</div>{extra}'


# Cada etapa: titulo, que hace (contado en lenguaje llano), en que nos apoyamos
# (el concepto de la materia), el archivo de src/ que lo implementa, y un
# ejemplo real que muestra la transformacion.
STAGES = [
    {
        "titulo": "Origen de los datos: FDA FAERS",
        "que": "Arrancamos de reportes reales de eventos adversos que medicos y "
               "pacientes le mandan a la FDA. Lo primero que descubrimos es que un "
               "caso no viene en un solo lugar: llega partido en cuatro archivos de "
               "texto que solo comparten un numero de reporte (primaryid).",
        "base": "Fuente de datos abierta y real. Datos crudos, sin estructura.",
        "archivo": "data/raw/ + config.faers_files()",
        "ejemplo": _flow(
            _box("DEMO.txt", "primaryid=254258761<br>age=74 . age_cod=YR . sex=F", "raw"),
            _box("DRUG.txt", "drugname=LENALIDOMIDE<br>role_cod=PS", "raw"),
            _box("REAC.txt", "pt=ARTHROPATHY", "raw"),
            _box("INDI.txt", "indi_pt=PLASMA CELL MYELOMA", "raw"),
            note="Es el mismo paciente repartido en cuatro archivos. El primaryid es "
                 "el hilo que nos deja volver a unirlos.",
        ),
    },
    {
        "titulo": "Limpieza y normalizacion",
        "que": "Antes de poder usar nada tuvimos que ordenar el desorden: la edad "
               "podia estar en anios, meses o decadas; el peso en kilos o en libras; "
               "y el sexo aparecia escrito de muchas maneras. Llevamos todo a una "
               "unica forma con tablas de equivalencia.",
        "base": "Preprocesamiento y limpieza. La calidad del dato condiciona todo lo demas.",
        "archivo": "preprocess.py, load_demo()",
        "ejemplo": _flow(
            _box("CRUDO", "age=\"74\"<br>age_cod=\"YR\"", "raw"),
            _box("regla", "age_map = {'YR':1,<br>'MON':1/12, 'DEC':10}", "proc"),
            _box("LIMPIO", "age_years = 74.0<br>sex = \"F\" . weight = NaN", "out"),
            note="'YR' significa anios, asi que 74 queda igual; si dijera 'DEC' "
                 "(decadas) seria 74 x 10. Cuando falta un dato lo dejamos en NaN en "
                 "vez de inventar un valor.",
        ),
    },
    {
        "titulo": "Armado del caso completo",
        "que": "Con los datos ya limpios, volvemos a pegar los cuatro archivos por el "
               "primaryid para reconstruir el caso entero. Aca tomamos una decision "
               "importante: separar el farmaco principal sospechoso (rol PS) de las "
               "otras medicaciones que el paciente venia tomando, porque las dos cosas "
               "le dicen algo distinto al modelo.",
        "base": "Integracion de datos, join relacional y conocimiento del dominio FAERS.",
        "archivo": "preprocess.py, build_dataset()",
        "ejemplo": _flow(
            _box("cuatro archivos", "DEMO + DRUG +<br>REAC + INDI", "raw"),
            _box("union por id", "merge(on='primaryid')", "proc"),
            _box("UNA FILA", "{sex:F, age:74,<br>drug:LENALIDOMIDE (PS),<br>indi:Plasma Cell Myeloma}", "out"),
            note="Pasamos de cuatro archivos sueltos a una fila por caso, que es la "
                 "unidad con la que despues trabaja el modelo.",
        ),
    },
    {
        "titulo": "Que vamos a predecir",
        "que": "Tuvimos que decidir cuales reacciones valia la pena predecir. Nos "
               "quedamos con las que aparecen seguido (al menos 300 veces) y sacamos "
               "los terminos que en realidad no son efectos del farmaco, como errores "
               "de dosis o el embarazo. Si no, el modelo terminaba aprendiendo ruido.",
        "base": "Definicion del objetivo, vocabulario MedDRA y el problema del desbalance.",
        "archivo": "labels.py, build_label_vocab()",
        "ejemplo": _flow(
            _box("REACCIONES CRUDAS", "Nausea . Pyrexia .<br>Off Label Use . Pemphigus", "raw"),
            _box("filtros", "frecuencia >= 300<br>y que sea un efecto real", "proc"),
            _box("ETIQUETAS (Y)", "Nausea (se queda)<br>Pyrexia (se queda)<br>"
                                  "Off Label Use (no es efecto)<br>Pemphigus (muy raro)", "out"),
            note="Al final solo dejamos lo que es un efecto adverso de verdad y que "
                 "aparece lo suficiente como para poder aprenderlo.",
        ),
    },
    {
        "titulo": "El paciente como una frase",
        "que": "Para que BioBERT pueda leer un caso, lo convertimos en una sola frase "
               "armada siempre igual. Usamos la misma funcion al entrenar y en la app, "
               "asi nos aseguramos de que el modelo nunca reciba un formato distinto "
               "del que aprendio.",
        "base": "Ingenieria de features, representacion del texto y consistencia entre "
                "entrenamiento e inferencia.",
        "archivo": "patient_text.py, build_patient_text()",
        "ejemplo": _flow(
            _box("FILA", "{age:74, sex:F,<br>drug:LENALIDOMIDE,<br>indi:Plasma Cell Myeloma}", "raw"),
            _box("FRASE PARA EL MODELO",
                 "\"patient: age 74 years, sex<br>female, weight unknown. drug:<br>"
                 "LENALIDOMIDE. ... indication:<br>Plasma Cell Myeloma\"", "out"),
            note="Una plantilla fija toma los campos sueltos y los convierte en lenguaje "
                 "natural, que es lo que BioBERT sabe interpretar.",
        ),
    },
    {
        "titulo": "De texto a numeros",
        "que": "El modelo no entiende palabras, entiende numeros. El tokenizer parte la "
               "frase en pedacitos y les pone un id, y despues BioBERT transforma esa "
               "secuencia en un vector de 768 numeros que resume el significado medico "
               "de todo el caso.",
        "base": "Tokenizacion, word embeddings y modelos de lenguaje del tipo BERT.",
        "archivo": "biobert_embeddings.py, prepare_features.py",
        "ejemplo": _flow(
            _box("FRASE", "\"patient: age 74 ...\"", "raw"),
            _box("tokenizer", "[101, 5723, 2287,<br>6390, 1010, ...]", "proc"),
            _box("VECTOR 768d", "[0.12, -0.84, 0.05,<br>0.41, ...]  (significado)", "model"),
            note="Cada caso termina siendo un punto en un espacio de 768 dimensiones. "
                 "Ahi adentro el modelo compara casos parecidos y aprende.",
        ),
    },
    {
        "titulo": "De lo simple a lo potente",
        "que": "No empezamos por lo mas complejo. Probamos modelos de menor a mayor "
               "potencia (Naive Bayes y KNN como base, despues Random Forest, y "
               "finalmente fine-tuning de BioBERT) justamente para poder mostrar por "
               "que hizo falta cada salto.",
        "base": "Clasificadores supervisados, ensembles y transformers (BERT).",
        "archivo": "train_*.py, naive_bayes, knn, rf",
        "ejemplo": _flow(
            _box("VECTOR 768d", "[0.12, -0.84, ...]", "raw"),
            _box("BioBERTClassifier", "Linear, ReLU, Linear<br>(98 salidas)", "proc"),
            _box("PROBABILIDADES", "Arthralgia: 0.69<br>Nausea: 0.55<br>Anaemia: 0.22", "model"),
            note="El modelo no elige una sola respuesta: da una probabilidad para cada "
                 "uno de los 98 efectos posibles, porque un caso puede tener varios.",
        ),
    },
    {
        "titulo": "Donde cortamos para decidir",
        "que": "Como hay muchisimos mas 'no' que 'si', cortar siempre en 0.5 daba malos "
               "resultados. Lo que hicimos fue buscar, para cada efecto por separado, el "
               "punto de corte que mejor equilibra acertar sin exagerar.",
        "base": "Optimizacion del umbral de decision y manejo del desbalance (pos_weight).",
        "archivo": "train.py / tune_threshold.py",
        "ejemplo": _flow(
            _box("PROBABILIDAD", "Arthralgia = 0.69<br>Anaemia = 0.22", "raw"),
            _box("umbral propio", "Arthralgia: >= 0.41<br>Anaemia: >= 0.35", "proc"),
            _box("DECISION", "Arthralgia: si<br>Anaemia: no", "out"),
            note="Cada etiqueta tiene su propio umbral, ajustado para que esa etiqueta "
                 "en particular salga lo mejor posible.",
        ),
    },
    {
        "titulo": "Como medimos si funciona",
        "que": "Evaluamos sobre un 30% de casos que el modelo nunca vio, separados desde "
               "el principio con una semilla fija. Comparamos lo que predijo contra lo "
               "que realmente paso y contamos aciertos, falsas alarmas y cosas que se le "
               "escaparon, sin maquillar los errores.",
        "base": "Separacion train/test, validacion cruzada y metricas multi-label (F1).",
        "archivo": "eval_test_cases.py, cross_validation.py",
        "ejemplo": _flow(
            _box("REALES (FAERS)", "{Arthralgia, Nausea}", "raw"),
            _box("PREDICHAS", "{Arthralgia, Pyrexia}", "model"),
            _box("RESULTADO", "acierto: Arthralgia<br>falsa alarma: Pyrexia<br>se escapo: Nausea", "out"),
            note="De esa comparacion salen la precision, el recall y el F1. Lo mostramos "
                 "caso por caso, con los errores incluidos.",
        ),
    },
    {
        "titulo": "Contraste con SIDER 4.1",
        "que": "Como prueba final no quisimos creernos solos. Comparamos nuestras "
               "predicciones contra SIDER, una base de datos curada de efectos adversos "
               "ya conocidos para cada farmaco. Es una fuente independiente de la nuestra.",
        "base": "Validacion externa contra una referencia independiente.",
        "archivo": "validate_sider.py + pestaña de la app",
        "ejemplo": _flow(
            _box("FARMACO", "LENALIDOMIDE", "raw"),
            _box("buscar en SIDER", "nombre -> STITCH id -><br>efectos conocidos", "proc"),
            _box("CONTRASTE", "SIDER: {Arthralgia,<br>Fatigue, ...}<br>coincide con lo que predijimos", "out"),
            note="Si lo que predecimos ya esta documentado en SIDER, tenemos evidencia "
                 "independiente de que el modelo va por buen camino.",
        ),
    },
]


# ── Problemas reales que tuvimos y como los resolvimos (todos verificables en el
# codigo). cat = categoria para el color de la tarjeta. ──────────────────────
DECISIONS = [
    {
        "cat": "datos", "etiqueta": "Juntar los datos",
        "problema": "Cada caso venia partido en cuatro archivos (DEMO, DRUG, REAC e "
                    "INDI) y ademas repartido en varios trimestres, asi que no habia "
                    "un unico lugar de donde leer un paciente completo.",
        "solucion": "Primero concatenamos todos los trimestres de cada tipo y despues "
                    "unimos los cuatro archivos con un merge por primaryid. Para no "
                    "atarnos a una lista fija, los buscamos por patron: si manana hay "
                    "un trimestre nuevo, no hay que tocar el codigo.",
        "conv": "Convencion: los archivos quedan ordenados por trimestre, con el mas "
                "nuevo al final.",
        "archivo": "preprocess.py, config.faers_files()",
    },
    {
        "cat": "datos", "etiqueta": "Casos duplicados",
        "problema": "Un mismo paciente aparece en mas de un trimestre porque la FDA va "
                    "actualizando el reporte, asi que el caso nos quedaba repetido.",
        "solucion": "Como los archivos ya estan ordenados del mas viejo al mas nuevo, "
                    "simplemente nos quedamos con la ultima version de cada paciente.",
        "conv": "Atajo: aprovechamos que el orden de los archivos ya es la fecha, asi "
                "no tuvimos que parsear ninguna marca de tiempo.",
        "archivo": "preprocess.py, load_demo()",
    },
    {
        "cat": "datos", "etiqueta": "Unidades mezcladas",
        "problema": "La edad venia a veces en anios, otras en meses o decadas; el peso "
                    "en kilos o en libras; y el sexo escrito de formas distintas. Nada "
                    "de eso se podia comparar tal cual.",
        "solucion": "Armamos tablas de equivalencia para llevar la edad a anios, el "
                    "peso a kilos y el sexo a M, F o desconocido.",
        "conv": "Convencion: si no sabemos la unidad, no inventamos; dejamos el valor "
                "como desconocido (NaN).",
        "archivo": "preprocess.py, load_demo()",
    },
    {
        "cat": "rigor", "etiqueta": "Ruido en lo que predecimos",
        "problema": "FAERS mete dentro de las reacciones cosas que no son efectos del "
                    "farmaco, como errores de dosis o el embarazo. Por eso al principio "
                    "el modelo llego a predecir 'exposicion materna en el embarazo' para "
                    "un hombre de 72 anios.",
        "solucion": "Hicimos una lista de terminos a excluir y ademas pedimos una "
                    "frecuencia minima, de modo que solo quedan efectos adversos reales "
                    "y bien representados.",
        "conv": "Fue una decision de dominio: curamos a mano lo que el modelo tiene que "
                "aprender en vez de confiar ciegamente en los datos.",
        "archivo": "labels.py, build_label_vocab()",
    },
    {
        "cat": "rigor", "etiqueta": "Filtrado de informacion",
        "problema": "Si calculabamos cosas como el TF-IDF o la mediana de edad usando "
                    "todo el dataset, el modelo terminaba 'viendo' datos de test al "
                    "entrenar y las metricas salian mejores de lo que en realidad eran.",
        "solucion": "Separamos test primero y recien despues ajustamos cualquier "
                    "transformacion, siempre usando solo los datos de entrenamiento.",
        "conv": "La regla que seguimos: nada que dependa de los datos se calcula antes "
                "de apartar el conjunto de test.",
        "archivo": "prepare_features.py",
    },
    {
        "cat": "rigor", "etiqueta": "Que entrenar e inferir coincidan",
        "problema": "Si entrenabamos con un formato de texto y la app armaba otro "
                    "parecido pero distinto, el modelo recibia en produccion algo que "
                    "nunca habia visto.",
        "solucion": "Dejamos una sola funcion encargada de armar la frase del paciente, "
                    "y la importan por igual el entrenamiento, la evaluacion y la app.",
        "conv": "Es el principio de no repetirse: una unica definicion del dato de "
                "entrada, sin copiar y pegar en cada script.",
        "archivo": "patient_text.py",
    },
    {
        "cat": "modelo", "etiqueta": "Muchos mas 'no' que 'si'",
        "problema": "Apenas el 2% de los casos es positivo. Cuando le dimos mucho peso a "
                    "los positivos para compensar, el modelo se fue al otro extremo y "
                    "empezo a decir 'si' a casi todo.",
        "solucion": "Le dimos mas peso a las clases raras pero con un tope, porque sin "
                    "limite ese peso se disparaba y desestabilizaba el entrenamiento.",
        "conv": "Atajo comodo: un unico numero controla el equilibrio entre detectar "
                "mas y equivocarse menos.",
        "archivo": "train.py, config.POS_WEIGHT_CAP",
    },
    {
        "cat": "modelo", "etiqueta": "El corte en 0.5 no servia",
        "problema": "Con las clases tan desbalanceadas, decidir siempre en 0.5 daba un "
                    "F1 muy pobre, porque cada efecto tiene una probabilidad tipica "
                    "diferente.",
        "solucion": "Buscamos el punto de corte ideal para cada efecto por separado y lo "
                    "guardamos junto al modelo para reutilizarlo despues.",
        "conv": "Terminamos con 98 umbrales, uno por efecto, en lugar de un solo 0.5 "
                "para todos.",
        "archivo": "train.py / tune_threshold.py",
    },
    {
        "cat": "modelo", "etiqueta": "Nombres que no coincidian",
        "problema": "FAERS escribe 'METFORMIN HYDROCHLORIDE' y SIDER guarda 'metformin'. "
                    "Por culpa de esos sufijos de sales, el cruce entre las dos bases "
                    "fallaba todo el tiempo.",
        "solucion": "Probamos primero el nombre completo y, si no aparecia, ibamos "
                    "sacando sufijos de sales conocidos hasta dar con el ingrediente base.",
        "conv": "Fue una solucion pragmatica: una lista corta de sufijos cubre la "
                "mayoria de los casos sin armar un normalizador complicado.",
        "archivo": "app.py, map_drug_to_sider()",
    },
    {
        "cat": "ingenieria", "etiqueta": "No depender de una corrida perfecta",
        "problema": "El fine-tuning lleva su tiempo, y si se cortaba la luz o cerrabamos "
                    "la maquina a mitad de camino perdiamos horas de entrenamiento.",
        "solucion": "Hicimos el entrenamiento reanudable por tandas: guarda su progreso "
                    "despues de cada epoca, de forma segura, y al volver a correrlo "
                    "sigue donde habia quedado.",
        "conv": "En la practica se corre por partes, y si los datos cambiaron lo detecta "
                "y arranca de nuevo en vez de mezclar cosas.",
        "archivo": "train.py, atomic_save()",
    },
]


def decisions_html(items=DECISIONS):
    """Devuelve el HTML de las tarjetas 'problema, solucion, convencion'."""
    cards = []
    for d in items:
        cards.append(f"""
        <div class="card cat-{d['cat']}">
          <div class="card-tag">{d['etiqueta']}</div>
          <div class="row"><span class="lab lab-prob">Problema</span><div>{d['problema']}</div></div>
          <div class="row"><span class="lab lab-sol">Como lo encaramos</span><div>{d['solucion']}</div></div>
          <div class="conv">{d['conv']}</div>
          <div class="cfile"><code>{d['archivo']}</code></div>
        </div>""")

    return f"""
<div id="decisions">
  <div class="grid">{''.join(cards)}</div>
</div>
<style>
  #decisions {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; padding: 2px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 12px; }}
  .card {{
    background: #fff; border: 1px solid #e1e8ef; border-top: 4px solid #95a5a6;
    border-radius: 10px; padding: 13px 15px; color: #2c3e50;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
  }}
  .card-tag {{ font-size: 14px; font-weight: 700; margin-bottom: 10px; color: #2c3e50; }}
  .row {{ font-size: 13px; line-height: 1.5; margin-bottom: 9px; color: #34495e; }}
  .lab {{
    display: inline-block; font-size: 10px; font-weight: 700; letter-spacing: .04em;
    text-transform: uppercase; padding: 2px 7px; border-radius: 4px; margin-bottom: 3px;
  }}
  .lab-prob {{ background: #fdecea; color: #c0392b; }}
  .lab-sol  {{ background: #eafaf1; color: #1e7d4f; }}
  .conv {{
    font-size: 12px; background: #f7fafc; border-radius: 6px; padding: 7px 10px;
    color: #5d6d7e; margin: 8px 0 6px; font-style: italic;
  }}
  .cfile code {{
    font-size: 11px; background: #eaf2f8; color: #2471a3;
    padding: 1px 6px; border-radius: 4px;
  }}
  .cat-datos      {{ border-top-color: #3498db; }}
  .cat-rigor      {{ border-top-color: #16a085; }}
  .cat-modelo     {{ border-top-color: #8e44ad; }}
  .cat-ingenieria {{ border-top-color: #e67e22; }}
</style>
"""


def pipeline_html(stages=STAGES):
    """Devuelve el HTML autocontenido (CSS+JS) del diagrama animado del pipeline."""
    import json

    cards = []
    for i, s in enumerate(stages):
        cards.append(f"""
        <div class="stage" data-idx="{i}">
          <div class="connector"></div>
          <div class="node"><span class="num">{i + 1}</span></div>
          <div class="label">{s['titulo']}</div>
        </div>""")

    # Datos que el panel de detalle muestra al activarse cada etapa (via JS).
    details = json.dumps(
        [{"titulo": s["titulo"], "que": s["que"], "base": s["base"],
          "archivo": s["archivo"], "ejemplo": s["ejemplo"]} for s in stages]
    )

    return f"""
<div id="pipeline">
  <div class="track">{''.join(cards)}</div>
  <div id="detail" class="detail">
    <div class="d-head"><h3 id="d-titulo"></h3></div>
    <p id="d-que"></p>
    <div id="d-ejemplo" class="ejemplo"></div>
    <div class="d-meta">
      <div class="d-base"><b>En que nos apoyamos:</b> <span id="d-base"></span></div>
      <div class="d-file"><b>En el codigo:</b> <code id="d-archivo"></code></div>
    </div>
  </div>
  <div class="hint">Avanza solo. Pasa el mouse por una etapa para fijarla.</div>
</div>

<style>
  #pipeline {{
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    color: #2c3e50; padding: 4px 2px 14px;
  }}
  .track {{
    display: flex; flex-wrap: wrap; gap: 4px 0;
    justify-content: center; align-items: flex-start;
  }}
  .stage {{
    position: relative; flex: 1 1 92px; max-width: 120px; min-width: 84px;
    text-align: center; cursor: pointer; padding-top: 14px;
  }}
  .connector {{
    position: absolute; top: 43px; left: -50%; width: 100%; height: 4px;
    background: #e1e8ef; z-index: 0; border-radius: 2px; overflow: hidden;
  }}
  .stage[data-idx="0"] .connector {{ display: none; }}
  .connector::after {{
    content: ""; position: absolute; top: 0; left: -40%; width: 40%; height: 100%;
    background: linear-gradient(90deg, transparent, #3498db, transparent);
    animation: flow 2.4s linear infinite;
  }}
  @keyframes flow {{ from {{ left: -40%; }} to {{ left: 100%; }} }}
  .node {{
    position: relative; z-index: 1; width: 52px; height: 52px; margin: 0 auto;
    border-radius: 50%; background: #fff; border: 3px solid #cdd7e1;
    display: flex; align-items: center; justify-content: center;
    transition: all .35s ease;
  }}
  .node .num {{ font-size: 21px; font-weight: 700; color: #95a5a6; transition: color .35s ease; }}
  .label {{
    font-size: 11px; line-height: 1.25; margin-top: 7px; color: #5d6d7e;
    transition: color .35s ease; padding: 0 3px;
  }}
  .stage.active .node {{
    border-color: #3498db; box-shadow: 0 0 0 6px rgba(52,152,219,.15);
    transform: scale(1.12);
  }}
  .stage.active .num {{ color: #3498db; }}
  .stage.active .label {{ color: #2c3e50; font-weight: 600; }}

  .detail {{
    margin: 16px auto 0; max-width: 820px; background: #f7fafc;
    border: 1px solid #e1e8ef; border-left: 5px solid #3498db;
    border-radius: 10px; padding: 14px 18px; animation: fade .4s ease;
  }}
  @keyframes fade {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; }} }}
  .d-head {{ margin-bottom: 6px; }}
  .d-head h3 {{ margin: 0; font-size: 17px; color: #2c3e50; }}
  #d-que {{ font-size: 14px; line-height: 1.5; margin: 4px 0 10px; color: #34495e; }}

  /* Ejemplo real: flujo de cajas antes y despues */
  .ejemplo {{
    background: #fff; border: 1px dashed #cdd7e1; border-radius: 8px;
    padding: 12px 12px 8px; margin: 4px 0 12px;
  }}
  .flow {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: 6px; }}
  .box {{
    flex: 1 1 130px; border-radius: 7px; padding: 7px 9px; min-width: 120px;
    border: 1px solid; display: flex; flex-direction: column; gap: 3px;
  }}
  .box .cap {{
    font-size: 9.5px; font-weight: 700; letter-spacing: .04em;
    text-transform: uppercase; opacity: .8;
  }}
  .box code {{ font-size: 11.5px; line-height: 1.4; font-family: "Consolas", monospace; }}
  .box.raw   {{ background: #f4f6f7; border-color: #d5dbdb; color: #566573; }}
  .box.proc  {{ background: #fef9e7; border-color: #f7dc6f; color: #7d6608; }}
  .box.out   {{ background: #eafaf1; border-color: #82e0aa; color: #1e7d4f; }}
  .box.model {{ background: #eaf2f8; border-color: #85c1e9; color: #21618c; }}
  .arr {{ display: flex; align-items: center; color: #aeb6bf; font-size: 18px; }}
  .fnote {{ font-size: 11.5px; color: #7f8c8d; margin-top: 8px; font-style: italic; }}

  .d-meta {{ font-size: 12.5px; color: #5d6d7e; display: grid; gap: 4px; }}
  .d-base b, .d-file b {{ color: #2c3e50; }}
  .d-file code {{
    background: #eaf2f8; color: #2471a3; padding: 1px 6px; border-radius: 4px;
    font-size: 12px;
  }}
  .hint {{ text-align: center; font-size: 11.5px; color: #95a5a6; margin-top: 10px; }}
</style>

<script>
  const DATA = {details};
  const stages = Array.from(document.querySelectorAll('.stage'));
  const detail = document.getElementById('detail');
  let current = -1, paused = false;

  function show(i) {{
    stages.forEach(s => s.classList.remove('active'));
    stages[i].classList.add('active');
    const d = DATA[i];
    document.getElementById('d-titulo').textContent = (i + 1) + '. ' + d.titulo;
    document.getElementById('d-que').textContent = d.que;
    document.getElementById('d-ejemplo').innerHTML = d.ejemplo;
    document.getElementById('d-base').textContent = d.base;
    document.getElementById('d-archivo').textContent = d.archivo;
    detail.style.animation = 'none';
    void detail.offsetWidth;            // reinicia la animacion de fade
    detail.style.animation = 'fade .4s ease';
    current = i;
  }}

  function step() {{ if (!paused) show((current + 1) % stages.length); }}

  stages.forEach((s, i) => {{
    s.addEventListener('mouseenter', () => {{ paused = true; show(i); }});
    s.addEventListener('mouseleave', () => {{ paused = false; }});
    s.addEventListener('click', () => {{ show(i); }});
  }});

  show(0);
  setInterval(step, 5000);             // avanza solo cada 5s (mas tiempo para leer el ejemplo)
</script>
"""
