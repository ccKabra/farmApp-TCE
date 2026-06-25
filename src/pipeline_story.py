"""
Narrativa visual del proyecto para la defensa oral — enfoque COMPUTACION EVOLUTIVA.

Genera un diagrama HTML/CSS animado que cuenta el recorrido del proyecto puesto
el foco en el Algoritmo Genetico: de donde sale el dato, como se CODIFICA
(genotipo), como es el FENOTIPO (la red), como se EVALUA (fitness), que OPERADORES
evolutivos se aplican, como CONVERGE y como se EXPLICA y se valida.

La preparacion del dato (FAERS) se cuenta breve, como soporte; el centro del
relato es la evolucion. La app (app.py) solo lo embebe con
st.components.v1.html(pipeline_html()). Cada etapa apunta a un archivo real de src/.
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


# Cada etapa: titulo, que hace (lenguaje llano), en que concepto de la materia se
# apoya (campo 'base'), el archivo de src/ que lo implementa, y un ejemplo real.
STAGES = [
    {
        "titulo": "El problema a optimizar (FDA FAERS)",
        "que": "Partimos de reportes reales de efectos adversos de la FDA. Para un "
               "Algoritmo Genetico esto es, sobre todo, un PROBLEMA DE OPTIMIZACION "
               "concreto: dado un paciente, predecir sus efectos adversos. El dato es "
               "el soporte; la contribucion del proyecto es como lo resolvemos.",
        "base": "Definicion del problema de optimizacion. El dato es el medio, no el fin.",
        "archivo": "preprocess.py, data_split.py",
        "ejemplo": _flow(
            _box("CASO REAL", "mujer, 74 anios<br>farmaco: LENALIDOMIDE<br>indic.: Plasma Cell Myeloma", "raw"),
            _box("OBJETIVO", "predecir el conjunto de<br>~98 efectos adversos posibles", "out"),
            note="El AG buscara un clasificador que, para casos como este, acierte que "
                 "efectos adversos van a aparecer.",
        ),
    },
    {
        "titulo": "Preparacion del dato (soporte)",
        "que": "Tarea de soporte, no el aporte central: unimos los cuatro archivos de "
               "FAERS por su numero de reporte, normalizamos edad/peso/sexo y nos "
               "quedamos con la ultima version de cada caso. Queda una fila por paciente.",
        "base": "Integracion y limpieza de datos. Necesario para tener un problema bien definido.",
        "archivo": "preprocess.py",
        "ejemplo": _flow(
            _box("4 ARCHIVOS", "DEMO + DRUG +<br>REAC + INDI", "raw"),
            _box("union + limpieza", "merge(primaryid)<br>edad/peso/sexo normalizados", "proc"),
            _box("UNA FILA", "{sex:F, age:74, drug:<br>LENALIDOMIDE, indi:...}", "out"),
            note="De cuatro archivos sueltos a una fila por caso: la unidad sobre la que "
                 "trabaja el AG.",
        ),
    },
    {
        "titulo": "Objetivo y restricciones (constraints)",
        "que": "Definimos QUE se predice: nos quedamos con las reacciones frecuentes "
               "(>= 300 casos) y descartamos terminos que no son efectos del farmaco "
               "(errores de dosis, embarazo). Es acotar el espacio del problema con "
               "restricciones de dominio.",
        "base": "Objetivo y constraints del problema (Modulo 4: fitness y restricciones).",
        "archivo": "labels.py, build_label_vocab()",
        "ejemplo": _flow(
            _box("REACCIONES CRUDAS", "Nausea . Off Label Use .<br>Pemphigus . Arthralgia", "raw"),
            _box("constraints", "frecuencia >= 300<br>+ que sea efecto real", "proc"),
            _box("98 ETIQUETAS (Y)", "Nausea, Arthralgia, ...<br>(las que el AG aprende)", "out"),
            note="El vocabulario de 98 efectos define las salidas de la red que el AG va "
                 "a optimizar.",
        ),
    },
    {
        "titulo": "Codificacion del dato de entrada (soporte)",
        "que": "Para meter el caso a la red lo convertimos en un vector numerico: TF-IDF "
               "sobre el texto del paciente + edad/sexo/peso. IMPORTANTE: esto es una "
               "codificacion inicial MINIMA del dato, no el aporte del proyecto. Lo "
               "mantenemos chico (154 numeros) justamente para que el cromosoma del AG "
               "sea manejable.",
        "base": "Codificacion minima del dato. Soporte tecnico, no contribucion central.",
        "archivo": "ga_features.py (PatientFeaturizer)",
        "ejemplo": _flow(
            _box("FRASE DEL PACIENTE", "\"patient: age 74 ...<br>drug: LENALIDOMIDE ...\"", "raw"),
            _box("codificacion minima", "TF-IDF + edad/sexo/peso", "proc"),
            _box("VECTOR 154d", "[0.0, 0.31, 0.0, ...,<br>age=0.74, sexF=1]", "out"),
            note="El vector es solo la ENTRADA. Lo interesante (los pesos) lo descubre el "
                 "Algoritmo Genetico en las etapas siguientes.",
        ),
    },
    {
        "titulo": "Genotipo y fenotipo",
        "que": "Aca empieza la Computacion Evolutiva. El FENOTIPO es una red de una capa "
               "oculta (154 -> 24 -> 98). Su GENOTIPO es el vector real con TODOS sus "
               "pesos y sesgos: 6.170 numeros reales. Optimizar esos pesos ES entrenar "
               "la red, pero con evolucion en vez de gradiente.",
        "base": "Representacion NO binaria / genotipo real (Modulo 3). Genotipo vs fenotipo.",
        "archivo": "ga_model.py (GAClassifier)",
        "ejemplo": _flow(
            _box("GENOTIPO (cromosoma)", "[w1, w2, ..., w6170]<br>(numeros reales)", "raw"),
            _box("set_genome()", "desempaqueta en<br>W1, b1, W2, b2", "proc"),
            _box("FENOTIPO (red)", "154 -> ReLU(24) -><br>sigmoide(98)", "model"),
            note="Cada individuo de la poblacion es un cromosoma; al desempaquetarlo se "
                 "obtiene una red clasificadora concreta.",
        ),
    },
    {
        "titulo": "Poblacion y funcion de fitness",
        "que": "Creamos una poblacion de 60 redes con pesos aleatorios. La calidad de "
               "cada individuo (su FITNESS) es el F1-macro que logra prediciendo sobre "
               "los datos, calibrando el mejor umbral por etiqueta. Ese numero es lo que "
               "la evolucion va a maximizar.",
        "base": "Funcion de fitness y poblacion (Modulo 2 y 4).",
        "archivo": "train_ga.py, fitness_of()",
        "ejemplo": _flow(
            _box("INDIVIDUO", "una red (cromosoma)", "raw"),
            _box("evaluar", "predice -> compara con<br>las etiquetas reales", "proc"),
            _box("FITNESS", "F1-macro = 0.058<br>(que tan buena es)", "out"),
            note="Mejor fitness = mejor clasificador. Es la presion que guia toda la "
                 "busqueda.",
        ),
    },
    {
        "titulo": "Operadores evolutivos",
        "que": "Generacion a generacion: elegimos padres por TORNEO (compiten de a 3, "
               "gana el mas apto), los combinamos con CRUCE, y aplicamos MUTACION "
               "GAUSSIANA (ruido a algunos pesos). Ademas, los 2 mejores pasan intactos "
               "por ELITISMO: garantia de que el mejor nunca empeora.",
        "base": "Seleccion, cruce, mutacion y elitismo (Modulo 2).",
        "archivo": "train_ga.py (tournament/crossover/mutate)",
        "ejemplo": _flow(
            _box("PADRES", "individuo A + individuo B<br>(ganaron su torneo)", "raw"),
            _box("cruce + mutacion", "mezcla de pesos +<br>ruido gaussiano N(0,sigma)", "proc"),
            _box("HIJOS", "nuevas redes, algunas<br>mejores que los padres", "out"),
            note="La recombinacion explota lo bueno que ya existe; la mutacion mantiene "
                 "diversidad y explora variantes nuevas.",
        ),
    },
    {
        "titulo": "Control de parametros y convergencia",
        "que": "La mutacion arranca grande (explorar) y se achica con las generaciones "
               "(explotar): es control deterministico de parametros. Registramos, por "
               "generacion, el mejor fitness, el promedio, la diversidad y la presion de "
               "seleccion, y los graficamos: la curva de convergencia.",
        "base": "Control de parametros (Modulo 5) y metricas de EAs mono-objetivo (Modulo 6).",
        "archivo": "train_ga.py -> outputs/ga_evolution.csv",
        "ejemplo": _flow(
            _box("GEN 1", "best=0.048<br>sigma=0.20 (explora)", "raw"),
            _box("...80 generaciones...", "sigma baja a 0.02<br>(explota)", "proc"),
            _box("GEN 80", "best=0.088<br>la poblacion convergio", "out"),
            note="El mejor y el promedio suben juntos: la poblacion entera mejora, no un "
                 "solo individuo con suerte.",
        ),
    },
    {
        "titulo": "Umbral de decision por etiqueta",
        "que": "Las salidas son probabilidades. Como hay muchos mas 'no' que 'si', "
               "cortar en 0.5 funciona mal. Para el mejor individuo buscamos, sobre "
               "validacion, el punto de corte ideal de CADA efecto por separado.",
        "base": "Manejo del desbalance / ajuste de la decision (constraint multi-label).",
        "archivo": "train_ga.py, best_thresholds_and_f1()",
        "ejemplo": _flow(
            _box("PROBABILIDAD", "Arthralgia = 0.69<br>Anaemia = 0.22", "raw"),
            _box("umbral propio", "Arthralgia >= 0.41<br>Anaemia >= 0.35", "proc"),
            _box("DECISION", "Arthralgia: si<br>Anaemia: no", "out"),
            note="98 umbrales (uno por efecto) calibrados sobre validacion, no un 0.5 "
                 "para todos.",
        ),
    },
    {
        "titulo": "Multi-objetivo: NSGA-II",
        "que": "Precision y recall estan en conflicto: predecir mas sube uno y baja el "
               "otro. En vez de un solo F1, con NSGA-II evolucionamos el FRENTE DE "
               "PARETO: el conjunto de soluciones no dominadas que ofrecen distintos "
               "compromisos precision/recall.",
        "base": "EAs multi-objetivo, dominancia de Pareto, crowding distance (Modulo 7).",
        "archivo": "train_ga_nsga2.py",
        "ejemplo": _flow(
            _box("UN SOLO F1", "esconde el compromiso", "raw"),
            _box("NSGA-II", "non-dominated sort +<br>crowding distance", "proc"),
            _box("FRENTE DE PARETO", "21 soluciones:<br>alta precision <-> alto recall", "model"),
            note="El profesor puede elegir el punto del frente que prefiera segun quiera "
                 "mas precision o mas cobertura.",
        ),
    },
    {
        "titulo": "Explicabilidad evolutiva (XAI)",
        "que": "Una ventaja de evolucionar una red CHICA es que el individuo resultante "
               "es inspeccionable. Propagando las magnitudes de sus pesos sacamos QUE "
               "features pesan mas en sus decisiones: que terminos del texto y que "
               "variables guian al modelo.",
        "base": "XAI + Computacion Evolutiva (Modulo 9).",
        "archivo": "ga_explain.py",
        "ejemplo": _flow(
            _box("INDIVIDUO EVOLUCIONADO", "pesos W1, W2", "raw"),
            _box("|W1| propagado por |W2|", "importancia por feature", "proc"),
            _box("EXPLICACION", "pesan: gabapentin,<br>hypertension, edad, sexo...", "out"),
            note="El modelo no es una caja negra: podemos mostrar en que se apoya para "
                 "predecir.",
        ),
    },
    {
        "titulo": "Validacion externa: SIDER 4.1",
        "que": "Como prueba independiente comparamos las predicciones del individuo "
               "evolucionado contra SIDER, una base curada de efectos adversos conocidos "
               "por farmaco. Es una fuente externa, ajena a nuestros datos.",
        "base": "Validacion externa contra una referencia independiente.",
        "archivo": "validate_sider.py",
        "ejemplo": _flow(
            _box("FARMACO", "LENALIDOMIDE", "raw"),
            _box("buscar en SIDER", "nombre -> efectos<br>documentados", "proc"),
            _box("CONTRASTE", "F1 ~ 0.32 vs SIDER<br>(coincide bastante)", "out"),
            note="Si lo que predecimos ya esta documentado, tenemos evidencia "
                 "independiente de que el AG aprendio algo util.",
        ),
    },
]


# ── Decisiones de diseno EVOLUTIVO (todas verificables en el codigo).
# cat = categoria para el color de la tarjeta. ────────────────────────────────
DECISIONS = [
    {
        "cat": "evolutivo", "etiqueta": "Por que un AG y no retropropagacion",
        "problema": "El objetivo de la materia es resolver el problema con Computacion "
                    "Evolutiva, no con el entrenamiento clasico por gradiente.",
        "solucion": "Tratamos los pesos de la red como un cromosoma y los optimizamos con "
                    "un Algoritmo Genetico (seleccion + cruce + mutacion). Optimizar los "
                    "pesos ES entrenar la red, pero por evolucion.",
        "conv": "El aporte del proyecto es el AG; la red y las features son el soporte.",
        "archivo": "train_ga.py, ga_model.py",
    },
    {
        "cat": "evolutivo", "etiqueta": "Genotipo real, no binario",
        "problema": "Los pesos de una red son numeros continuos; una cadena de bits no es "
                    "la representacion natural.",
        "solucion": "Usamos un genotipo de numeros reales (Modulo 3) con operadores para "
                    "reales: cruce uniforme/aritmetico y mutacion gaussiana. Como extra, "
                    "mostramos que tambien podria usarse un cromosoma BINARIO para "
                    "seleccionar features.",
        "conv": "La representacion se elige segun el problema; aca lo natural es real.",
        "archivo": "ga_model.py, ga_features.py",
    },
    {
        "cat": "evolutivo", "etiqueta": "El fitness tenia que tener pendiente",
        "problema": "Con densidad de positivos ~2%, si inicializabamos los sesgos en "
                    "logit(frecuencia) las salidas colapsaban en ~0.02, ninguna cruzaba "
                    "el umbral y el fitness quedaba PLANO en 0: el AG no tenia por donde "
                    "mejorar.",
        "solucion": "Inicializamos los sesgos en ~0 (las salidas arrancan repartidas) y "
                    "medimos el fitness con el mejor umbral POR ETIQUETA, que premia "
                    "cualquier mejora de ranking. Asi el paisaje de fitness es suave.",
        "conv": "Diseniar la funcion de fitness fue tan importante como el algoritmo.",
        "archivo": "train_ga.py, init_population()",
    },
    {
        "cat": "evolutivo", "etiqueta": "Seleccion por torneo, no ruleta",
        "problema": "La ruleta sufre con superindividuos (uno muy apto acapara la "
                    "reproduccion -> convergencia prematura) y es sensible a la escala "
                    "del fitness.",
        "solucion": "Usamos seleccion por torneo (k=3): presion de seleccion controlada e "
                    "invariante a la escala. Monitoreamos SelPres = MaxFit/AveFit.",
        "conv": "Presion de seleccion moderada para no perder diversidad temprano.",
        "archivo": "train_ga.py, tournament_select()",
    },
    {
        "cat": "evolutivo", "etiqueta": "Diversidad y convergencia prematura",
        "problema": "En poblaciones finitas la diversidad se pierde y la busqueda se "
                    "estanca en un optimo local.",
        "solucion": "Elitismo moderado (solo 2 elites), mutacion gaussiana que reinyecta "
                    "variacion, y registro de la diversidad (desvio del fitness) por "
                    "generacion para vigilarla.",
        "conv": "El elitismo garantiza no empeorar; la mutacion garantiza seguir explorando.",
        "archivo": "train_ga.py -> outputs/ga_evolution.csv",
    },
    {
        "cat": "evolutivo", "etiqueta": "Control de parametros (exploracion->explotacion)",
        "problema": "Conviene explorar mucho al principio y afinar al final, pero un sigma "
                    "fijo no logra las dos cosas.",
        "solucion": "Control deterministico (Modulo 5): la sigma de la mutacion decrece de "
                    "0.20 a 0.02 a lo largo de las generaciones.",
        "conv": "Schedule simple y predecible, facil de defender.",
        "archivo": "config.py, train_ga.py",
    },
    {
        "cat": "evolutivo", "etiqueta": "Precision vs recall en conflicto",
        "problema": "Un unico F1 esconde que mejorar la precision empeora el recall y "
                    "viceversa.",
        "solucion": "NSGA-II (Modulo 7) evoluciona el frente de Pareto completo: muchas "
                    "soluciones no dominadas con distintos balances precision/recall.",
        "conv": "Mostramos el compromiso entero, no un solo punto elegido a dedo.",
        "archivo": "train_ga_nsga2.py",
    },
    {
        "cat": "evolutivo", "etiqueta": "Modelo chico = explicable",
        "problema": "Queriamos poder defender QUE aprende el modelo, no solo su numero.",
        "solucion": "Al evolucionar una red chica, sus pesos son inspeccionables: derivamos "
                    "la importancia de cada feature propagando |W1| por |W2| (XAI, Modulo 9).",
        "conv": "La explicabilidad sale del propio individuo evolucionado.",
        "archivo": "ga_explain.py",
    },
    {
        "cat": "dato", "etiqueta": "TF-IDF como soporte, no como aporte",
        "problema": "Necesitabamos convertir el texto del paciente en numeros para la red, "
                    "sin que la codificacion se volviera el centro del proyecto.",
        "solucion": "Usamos TF-IDF como codificacion inicial MINIMA (150 terminos) sobre el "
                    "mismo texto canonico del paciente. Es soporte tecnico; lo que se "
                    "defiende es el AG que optimiza sobre ese vector.",
        "conv": "Si hiciera falta, el cromosoma binario de seleccion de features lo "
                "reemplaza/justifica.",
        "archivo": "ga_features.py",
    },
    {
        "cat": "rigor", "etiqueta": "Ruido en las etiquetas (constraints)",
        "problema": "FAERS mezcla en las reacciones cosas que no son efectos del farmaco "
                    "(errores de dosis, embarazo): el modelo aprendia ruido.",
        "solucion": "Restricciones de dominio: lista de terminos a excluir + frecuencia "
                    "minima. Asi el objetivo del AG queda bien definido.",
        "conv": "Curar el objetivo es parte del diseno del problema de optimizacion.",
        "archivo": "labels.py, build_label_vocab()",
    },
    {
        "cat": "rigor", "etiqueta": "Sin fuga de informacion",
        "problema": "Si la codificacion (TF-IDF, medianas) miraba todo el dataset, el "
                    "modelo 'veia' test al entrenar.",
        "solucion": "El featurizer se ajusta SOLO sobre train; el split 70/30 es "
                    "determinista (semilla 42) y compartido por la app, la evaluacion y "
                    "SIDER.",
        "conv": "Mismo split en todo el proyecto -> los casos de test son siempre los mismos.",
        "archivo": "ga_features.py, data_split.py",
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
  .cat-evolutivo  {{ border-top-color: #e74c3c; }}
  .cat-dato       {{ border-top-color: #3498db; }}
  .cat-rigor      {{ border-top-color: #16a085; }}
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
    background: linear-gradient(90deg, transparent, #e74c3c, transparent);
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
    border-color: #e74c3c; box-shadow: 0 0 0 6px rgba(231,76,60,.15);
    transform: scale(1.12);
  }}
  .stage.active .num {{ color: #e74c3c; }}
  .stage.active .label {{ color: #2c3e50; font-weight: 600; }}

  .detail {{
    margin: 16px auto 0; max-width: 820px; background: #f7fafc;
    border: 1px solid #e1e8ef; border-left: 5px solid #e74c3c;
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
  .box.model {{ background: #fdecea; border-color: #f1948a; color: #a93226; }}
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
  setInterval(step, 5000);             // avanza solo cada 5s
</script>
"""
