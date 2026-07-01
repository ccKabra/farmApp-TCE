// build_presentation.js — Genera PRESENTACION.pptx siguiendo las 7 secciones
// pedidas, expandidas con slides de detalle para defensa de ~20 min.
// Correr con: node build_presentation.js

const pptxgen = require("pptxgenjs");
const path = require("path");

const ROOT = __dirname;
const FIG = path.join(ROOT, "outputs", "figures");

const C = {
  navy:    "1A2B4A",
  blue:    "2B6CB0",
  red:     "E53E3E",
  green:   "38A169",
  gold:    "D69E2E",
  purple:  "805AD5",
  bgLight: "FAFAFA",
  card:    "F0F5FA",
  cardAlt: "FFF5F5",
  cardGr:  "F0FFF4",
  cardPu:  "FAF5FF",
  textDk:  "1A202C",
  textMd:  "4A5568",
  textLt:  "FFFFFF",
  line:    "CBD5E0",
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Alzugaray · Cabrera Tamalet";
pres.title  = "farmApp-TCE — Computación Evolutiva";

const W = 13.3, H = 7.5;

function header(slide, num, title) {
  slide.background = { color: C.bgLight };
  slide.addText(num, {
    x: 0.5, y: 0.3, w: 0.9, h: 0.7,
    fontSize: 36, fontFace: "Calibri", bold: true,
    color: C.red, align: "left", valign: "middle", margin: 0,
  });
  slide.addText(title, {
    x: 1.45, y: 0.3, w: W - 2, h: 0.7,
    fontSize: 26, fontFace: "Calibri", bold: true,
    color: C.navy, align: "left", valign: "middle", margin: 0,
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.05, w: W - 1, h: 0.02,
    fill: { color: C.line }, line: { color: C.line, width: 0 },
  });
}

function card(slide, x, y, w, h, fill = C.card) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.1,
    fill: { color: fill }, line: { color: fill, width: 0 },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.08 },
  });
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — TITLE
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addShape(pres.shapes.OVAL, {
    x: -2, y: -2, w: 5, h: 5,
    fill: { color: C.red, transparency: 85 }, line: { color: C.red, width: 0 },
  });
  s.addShape(pres.shapes.OVAL, {
    x: W - 3, y: H - 3, w: 5, h: 5,
    fill: { color: C.blue, transparency: 80 }, line: { color: C.blue, width: 0 },
  });

  s.addText("Predicción de Efectos Adversos", {
    x: 1, y: 2.2, w: W - 2, h: 1.2,
    fontSize: 48, fontFace: "Calibri", bold: true,
    color: C.textLt, align: "center", valign: "middle", margin: 0,
  });
  s.addText("con Computación Evolutiva", {
    x: 1, y: 3.3, w: W - 2, h: 0.8,
    fontSize: 32, fontFace: "Calibri", italic: true,
    color: "CADCFC", align: "center", valign: "middle", margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: W/2 - 0.6, y: 4.3, w: 1.2, h: 0.04,
    fill: { color: C.red }, line: { color: C.red, width: 0 },
  });
  s.addText(
    "Un Algoritmo Genético optimiza los pesos de una red neuronal\npara predecir efectos adversos de fármacos sin retropropagación",
    { x: 1, y: 4.5, w: W - 2, h: 1.2,
      fontSize: 18, fontFace: "Calibri",
      color: "CADCFC", align: "center", valign: "top" }
  );
  s.addText("Alzugaray Agustín Ezequiel  ·  Lautaro Cabrera Tamalet", {
    x: 1, y: H - 1.3, w: W - 2, h: 0.5,
    fontSize: 16, fontFace: "Calibri",
    color: C.textLt, align: "center", valign: "middle",
  });
  s.addText("Técnicas de Computación Evolutiva", {
    x: 1, y: H - 0.8, w: W - 2, h: 0.4,
    fontSize: 14, fontFace: "Calibri", italic: true,
    color: "CADCFC", align: "center", valign: "middle",
  });
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — 1 · DESCRIPCIÓN DEL PROBLEMA
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "1", "Descripción del problema");

  card(s, 0.5, 1.4, W - 1, 1.3, C.cardAlt);
  s.addText("Dado un paciente que toma un fármaco, ¿qué efectos adversos va a sufrir?", {
    x: 0.7, y: 1.5, w: W - 1.4, h: 1.1,
    fontSize: 22, fontFace: "Cambria", italic: true, bold: true,
    color: C.navy, align: "center", valign: "middle",
  });

  card(s, 0.5, 3.0, 6.1, 4.0);
  s.addText("Características del problema", {
    x: 0.8, y: 3.15, w: 5.8, h: 0.5,
    fontSize: 18, fontFace: "Calibri", bold: true,
    color: C.blue, valign: "middle", margin: 0,
  });
  s.addText([
    { text: "Multi-label: un paciente puede tener varios efectos a la vez", options: { bullet: true, breakLine: true } },
    { text: "98 efectos posibles del vocabulario MedDRA", options: { bullet: true, breakLine: true } },
    { text: "Severamente desbalanceado: solo ~2 % de cada efecto es positivo", options: { bullet: true, breakLine: true } },
    { text: "Texto clínico ruidoso y heterogéneo", options: { bullet: true } },
  ], { x: 0.8, y: 3.7, w: 5.6, h: 3.0,
       fontSize: 15, fontFace: "Calibri", color: C.textDk, paraSpaceAfter: 8 });

  card(s, 6.7, 3.0, 6.1, 4.0);
  s.addText("Por qué es difícil y por qué importa", {
    x: 7.0, y: 3.15, w: 5.7, h: 0.5,
    fontSize: 18, fontFace: "Calibri", bold: true,
    color: C.red, valign: "middle", margin: 0,
  });
  s.addText([
    { text: "Un modelo que predice todo \"no\" tiene 98 % accuracy pero F1 ≈ 0", options: { bullet: true, breakLine: true } },
    { text: "98 etiquetas con frecuencias muy distintas", options: { bullet: true, breakLine: true } },
    { text: "Detectar efectos adversos antes de que ocurran ayuda a farmacovigilancia", options: { bullet: true, breakLine: true } },
    { text: "Mejora la seguridad del paciente (Metformina, Adalimumab, etc.)", options: { bullet: true } },
  ], { x: 7.0, y: 3.7, w: 5.7, h: 3.0,
       fontSize: 15, fontFace: "Calibri", color: C.textDk, paraSpaceAfter: 8 });
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — 2 · DATASETS
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "2", "Datasets — qué usamos y para qué");

  card(s, 0.5, 1.4, 6.1, 2.6);
  s.addText("FAERS (FDA)", { x: 0.8, y: 1.55, w: 5.8, h: 0.5,
    fontSize: 22, fontFace: "Calibri", bold: true, color: C.blue, margin: 0 });
  s.addText("Entrenamiento + test", { x: 0.8, y: 2.05, w: 5.8, h: 0.4,
    fontSize: 13, fontFace: "Calibri", italic: true, color: C.textMd, margin: 0 });
  s.addText([
    { text: "Reportes reales de pacientes con efectos adversos", options: { bullet: true, breakLine: true } },
    { text: "~55 000 casos · 5 trimestres (2025Q1 → 2026Q1)", options: { bullet: true, breakLine: true } },
    { text: "4 archivos por trimestre: DEMO + DRUG + REAC + INDI", options: { bullet: true } },
  ], { x: 0.8, y: 2.45, w: 5.6, h: 1.5, fontSize: 13, color: C.textDk, paraSpaceAfter: 4 });

  card(s, 6.7, 1.4, 6.1, 2.6);
  s.addText("SIDER 4.1", { x: 7.0, y: 1.55, w: 5.7, h: 0.5,
    fontSize: 22, fontFace: "Calibri", bold: true, color: C.red, margin: 0 });
  s.addText("Validación externa (después del entrenamiento)", { x: 7.0, y: 2.05, w: 5.7, h: 0.4,
    fontSize: 13, fontFace: "Calibri", italic: true, color: C.textMd, margin: 0 });
  s.addText([
    { text: "Base curada por farmacólogos", options: { bullet: true, breakLine: true } },
    { text: "Para cada fármaco, sus efectos adversos conocidos", options: { bullet: true, breakLine: true } },
    { text: "Independiente de FAERS — sirve como prueba externa", options: { bullet: true } },
  ], { x: 7.0, y: 2.45, w: 5.7, h: 1.5, fontSize: 13, color: C.textDk, paraSpaceAfter: 4 });

  card(s, 0.5, 4.2, W - 1, 2.9);
  s.addText("De FAERS sacamos X e Y", { x: 0.8, y: 4.35, w: 12, h: 0.5,
    fontSize: 18, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.8, y: 4.95, w: 5.8, h: 1.95, rectRadius: 0.08,
    fill: { color: "FFFFFF" }, line: { color: C.blue, width: 1.5 },
  });
  s.addText("X = entrada", { x: 1.0, y: 5.05, w: 5.4, h: 0.4,
    fontSize: 15, fontFace: "Calibri", bold: true, color: C.blue, margin: 0 });
  s.addText("Perfil del paciente: fármaco · indicación · otras medicaciones · edad · sexo · peso", {
    x: 1.0, y: 5.45, w: 5.4, h: 1.4, fontSize: 13, fontFace: "Calibri", color: C.textDk });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 6.7, y: 4.95, w: 6.1, h: 1.95, rectRadius: 0.08,
    fill: { color: "FFFFFF" }, line: { color: C.red, width: 1.5 },
  });
  s.addText("Y = lo que queremos predecir", { x: 6.9, y: 5.05, w: 5.7, h: 0.4,
    fontSize: 15, fontFace: "Calibri", bold: true, color: C.red, margin: 0 });
  s.addText("Vector binario de 98 columnas: ¿tuvo Nausea? ¿Anxiety? ¿Headache? ...", {
    x: 6.9, y: 5.45, w: 5.7, h: 1.4, fontSize: 13, fontFace: "Calibri", color: C.textDk });
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — 3 · ARQUITECTURA
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "3", "Arquitectura — cómo funciona el sistema");

  const steps = [
    ["Datos FAERS", "1 fila por paciente", C.blue],
    ["Featurizer", "vector X de 228 números", C.blue],
    ["Red neuronal 228 → 24 → 98", "predice Ŷ (probabilidades)", C.navy],
    ["Fitness = F1-macro", "compara Ŷ con Y real", C.gold],
    ["Algoritmo Genético", "ajusta los 7 946 pesos", C.red],
    ["Validación con SIDER", "verificación externa", C.green],
  ];
  const PX = 0.7, PW = 6.0, PY0 = 1.35, PH = 0.85, PG = 0.13;
  for (let i = 0; i < steps.length; i++) {
    const [title, sub, color] = steps[i];
    const y = PY0 + i * (PH + PG);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: PX, y, w: PW, h: PH, rectRadius: 0.08,
      fill: { color: "FFFFFF" }, line: { color, width: 2 },
    });
    s.addShape(pres.shapes.OVAL, {
      x: PX + 0.15, y: y + (PH - 0.5)/2, w: 0.5, h: 0.5,
      fill: { color }, line: { color, width: 0 },
    });
    s.addText(String(i + 1), {
      x: PX + 0.15, y: y + (PH - 0.5)/2, w: 0.5, h: 0.5,
      fontSize: 16, fontFace: "Calibri", bold: true,
      color: "FFFFFF", align: "center", valign: "middle", margin: 0,
    });
    s.addText(title, {
      x: PX + 0.75, y: y + 0.05, w: PW - 0.85, h: 0.4,
      fontSize: 14, fontFace: "Calibri", bold: true, color: C.textDk,
      valign: "middle", margin: 0,
    });
    s.addText(sub, {
      x: PX + 0.75, y: y + 0.42, w: PW - 0.85, h: 0.38,
      fontSize: 11, fontFace: "Calibri", italic: true, color: C.textMd,
      valign: "middle", margin: 0,
    });
    if (i < steps.length - 1) {
      s.addShape(pres.shapes.DOWN_ARROW, {
        x: PX + PW/2 - 0.07, y: y + PH + 0.005, w: 0.14, h: PG - 0.01,
        fill: { color: C.textMd }, line: { color: C.textMd, width: 0 },
      });
    }
  }

  const RX = 7.1, RY = 1.35;
  card(s, RX, RY, W - RX - 0.5, 1.7, C.cardAlt);
  s.addText("GENOTIPO", { x: RX + 0.2, y: RY + 0.1, w: 5.5, h: 0.35,
    fontSize: 12, fontFace: "Calibri", bold: true, color: C.red, charSpacing: 4, margin: 0 });
  s.addText("Vector real de 7 946 pesos (no binario, Módulo 3)", {
    x: RX + 0.2, y: RY + 0.45, w: 5.5, h: 0.45,
    fontSize: 14, fontFace: "Calibri", bold: true, color: C.textDk, margin: 0 });
  s.addText("[W1 (228×24) | b1 (24) | W2 (24×98) | b2 (98)]", {
    x: RX + 0.2, y: RY + 0.9, w: 5.5, h: 0.7,
    fontSize: 12, fontFace: "Consolas", color: C.textMd, valign: "top" });

  card(s, RX, RY + 1.9, W - RX - 0.5, 1.7, C.card);
  s.addText("FENOTIPO", { x: RX + 0.2, y: RY + 2.0, w: 5.5, h: 0.35,
    fontSize: 12, fontFace: "Calibri", bold: true, color: C.blue, charSpacing: 4, margin: 0 });
  s.addText("Red neuronal de 1 capa oculta", {
    x: RX + 0.2, y: RY + 2.35, w: 5.5, h: 0.45,
    fontSize: 14, fontFace: "Calibri", bold: true, color: C.textDk, margin: 0 });
  s.addText("228  →  ReLU(24)  →  sigmoide(98)", {
    x: RX + 0.2, y: RY + 2.8, w: 5.5, h: 0.7,
    fontSize: 13, fontFace: "Consolas", color: C.textMd, valign: "top" });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: RX, y: 5.5, w: W - RX - 0.5, h: 1.7, rectRadius: 0.1,
    fill: { color: C.navy }, line: { color: C.navy, width: 0 },
  });
  s.addText("Frase clave", { x: RX + 0.2, y: 5.6, w: 5.5, h: 0.3,
    fontSize: 11, fontFace: "Calibri", bold: true, color: C.red, charSpacing: 4, margin: 0 });
  s.addText(
    "El cromosoma son los pesos de la red. El AG no usa gradiente: combina y muta pesos hasta que la red predice Ŷ parecidas a las Y reales.",
    { x: RX + 0.2, y: 5.95, w: 5.5, h: 1.2,
      fontSize: 13, fontFace: "Cambria", italic: true, color: C.textLt }
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — 4 · REPRESENTACIÓN: GENOTIPO Y FENOTIPO (Módulo 3 detallado)
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "4", "Representación — genotipo y fenotipo (Módulo 3)");

  s.addText("La decisión más importante del AG: cómo codificar el problema", {
    x: 0.5, y: 1.15, w: W - 1, h: 0.35,
    fontSize: 13, fontFace: "Calibri", italic: true, color: C.textMd });

  // Genotipo (izquierda)
  card(s, 0.5, 1.6, 6.2, 5.5, C.cardAlt);
  s.addText("GENOTIPO — representación real no binaria", {
    x: 0.7, y: 1.75, w: 5.8, h: 0.4,
    fontSize: 16, fontFace: "Calibri", bold: true, color: C.red, margin: 0 });

  s.addText("[ W1 (228×24)  |  b1 (24)  |  W2 (24×98)  |  b2 (98) ]", {
    x: 0.7, y: 2.2, w: 5.8, h: 0.5,
    fontSize: 13, fontFace: "Consolas", color: C.textDk,
    align: "center", valign: "middle" });

  s.addText("= 7 946 números reales por individuo", {
    x: 0.7, y: 2.7, w: 5.8, h: 0.4,
    fontSize: 13, fontFace: "Calibri", italic: true, color: C.textMd,
    align: "center" });

  s.addText([
    { text: "Por qué REAL y no binario:", options: { bold: true, color: C.navy, breakLine: true } },
    { text: "  Los pesos de una red son continuos.", options: { bullet: true, breakLine: true } },
    { text: "  Codificar en bits perdería precisión.", options: { bullet: true, breakLine: true } },
    { text: "  Operadores naturales (reales): cruce aritmético + mutación gaussiana.", options: { bullet: true, breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Propiedad clave: LOCALIDAD", options: { bold: true, color: C.navy, breakLine: true } },
    { text: "  Cada gen es un peso individual; mutaciones chicas → cambios chicos en el fenotipo. Cumple con \"perfect representation\" del material.", options: { bullet: true } },
  ], { x: 0.7, y: 3.2, w: 5.8, h: 3.8,
       fontSize: 12, fontFace: "Calibri", color: C.textDk, paraSpaceAfter: 4 });

  // Fenotipo (derecha)
  card(s, 6.9, 1.6, 5.9, 5.5, C.card);
  s.addText("FENOTIPO — red neuronal", {
    x: 7.1, y: 1.75, w: 5.5, h: 0.4,
    fontSize: 16, fontFace: "Calibri", bold: true, color: C.blue, margin: 0 });

  // Diagrama compacto de capas
  const layerY = 2.4;
  const layers = [
    ["Entrada", "228 features", C.blue],
    ["Oculta", "24 neuronas ReLU", C.navy],
    ["Salida", "98 sigmoides", C.red],
  ];
  for (let i = 0; i < layers.length; i++) {
    const [n, sub, col] = layers[i];
    const x = 7.2 + i * 1.85;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: layerY, w: 1.6, h: 1.0, rectRadius: 0.08,
      fill: { color: "FFFFFF" }, line: { color: col, width: 2 },
    });
    s.addText(n, { x, y: layerY + 0.1, w: 1.6, h: 0.4,
      fontSize: 12, fontFace: "Calibri", bold: true, color: col,
      align: "center", margin: 0 });
    s.addText(sub, { x, y: layerY + 0.5, w: 1.6, h: 0.4,
      fontSize: 10, fontFace: "Calibri", italic: true, color: C.textMd,
      align: "center", margin: 0 });
    if (i < layers.length - 1) {
      s.addShape(pres.shapes.RIGHT_ARROW, {
        x: x + 1.6, y: layerY + 0.4, w: 0.25, h: 0.2,
        fill: { color: C.textMd }, line: { color: C.textMd, width: 0 },
      });
    }
  }

  s.addText([
    { text: "Cómo se conectan genotipo y fenotipo:", options: { bold: true, color: C.navy, breakLine: true } },
    { text: "  set_genome(cromosoma) desempaqueta los 7 946", options: { bullet: true, breakLine: true } },
    { text: "  números en las matrices W1, b1, W2, b2.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Forward pass:", options: { bold: true, color: C.navy, breakLine: true } },
    { text: "  Ŷ = sigmoide( ReLU(X·W1 + b1) · W2 + b2 )", options: { bullet: true } },
  ], { x: 7.1, y: 3.7, w: 5.5, h: 3.3,
       fontSize: 12, fontFace: "Calibri", color: C.textDk, paraSpaceAfter: 4 });
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — 5 · OPERADORES EVOLUTIVOS (Módulo 2 detallado)
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "5", "Operadores evolutivos (Módulos 2 y 4)");

  s.addText("Cómo el AG transforma una generación en la siguiente", {
    x: 0.5, y: 1.15, w: W - 1, h: 0.35,
    fontSize: 13, fontFace: "Calibri", italic: true, color: C.textMd });

  // 4 operadores en grilla 2×2
  const ops = [
    {
      title: "Selección por RANK lineal",
      module: "Módulo 4",
      color: C.blue,
      cfg: "selection = rank, s = 1.7",
      desc: "Cada individuo recibe probabilidad según su posición en el ranking (no según fitness absoluto). Presión selectiva CONSTANTE = s = 1.7, independiente de la escala del fitness y del colapso de la población.",
    },
    {
      title: "Cruce uniforme",
      module: "Módulo 2",
      color: C.green,
      cfg: "p_crossover = 0.9",
      desc: "Para cada gen, hijo1 hereda del padre A o B con probabilidad 0.5. Operador primario — recombina building blocks de buenos individuos.",
    },
    {
      title: "Mutación gaussiana",
      module: "Módulo 2",
      color: C.gold,
      cfg: "p_mutation = 0.05",
      desc: "Cada gen tiene 5 % de probabilidad de recibir ruido N(0, σ). Operador secundario — reinyecta variación y evita perder alelos.",
    },
    {
      title: "Elitismo + control adaptativo",
      module: "Módulo 2/5",
      color: C.red,
      cfg: "elitism = 2, immigrants = 8",
      desc: "Los 2 mejores pasan intactos. Diversidad medida como std promedio del GENOTIPO: si cae bajo 0.09 se inyectan 8 inmigrantes aleatorios y σ se multiplica ×3 (boost) para explorar de nuevo.",
    },
  ];

  for (let i = 0; i < 4; i++) {
    const op = ops[i];
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.5 + col * 6.4;
    const y = 1.6 + row * 2.85;

    card(s, x, y, 6.1, 2.7);
    s.addText(op.title, { x: x + 0.2, y: y + 0.15, w: 4.3, h: 0.45,
      fontSize: 17, fontFace: "Calibri", bold: true, color: op.color, margin: 0 });
    s.addText(op.module, { x: x + 4.5, y: y + 0.15, w: 1.5, h: 0.45,
      fontSize: 11, fontFace: "Calibri", italic: true, color: C.textMd,
      align: "right", margin: 0 });

    s.addText(op.cfg, { x: x + 0.2, y: y + 0.6, w: 5.7, h: 0.4,
      fontSize: 12, fontFace: "Consolas", color: op.color, margin: 0 });

    s.addText(op.desc, { x: x + 0.2, y: y + 1.05, w: 5.7, h: 1.55,
      fontSize: 12, fontFace: "Calibri", color: C.textDk });
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — 6 · CÓMO EVOLUCIONÓ EL FITNESS
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "6", "Cómo evolucionó el fitness");

  s.addImage({
    path: path.join(FIG, "ga_convergence.png"),
    x: 0.5, y: 1.4, w: 8.5, h: 5.0, sizing: { type: "contain", w: 8.5, h: 5.0 },
  });

  const RX = 9.3, RY = 1.4;
  card(s, RX, RY, W - RX - 0.5, 5.0);
  s.addText("Lo que se ve", { x: RX + 0.2, y: RY + 0.1, w: 3.3, h: 0.4,
    fontSize: 16, fontFace: "Calibri", bold: true, color: C.navy, margin: 0 });

  s.addText([
    { text: "Mejor fitness 0.05 → 0.14", options: { bullet: true, bold: true, breakLine: true } },
    { text: "  en 300 generaciones", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Diversidad genotípica (naranja)", options: { bullet: true, bold: true, breakLine: true } },
    { text: "  std promedio del cromosoma", options: { breakLine: true } },
    { text: "  (independiente del fitness)", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "5 inyecciones de inmigrantes", options: { bullet: true, bold: true, breakLine: true } },
    { text: "  al cruzar el umbral 0.09", options: { breakLine: true } },
    { text: "  (control adaptativo, Módulo 5)", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Presión selectiva = 1.7", options: { bullet: true, bold: true, breakLine: true } },
    { text: "  rank lineal, constante", options: { breakLine: true } },
    { text: "  (independiente de la población, Módulo 4)", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "σ adaptativo (verde)", options: { bullet: true, bold: true, breakLine: true } },
    { text: "  reacciona a la diversidad", options: {} },
  ], { x: RX + 0.2, y: RY + 0.55, w: 3.3, h: 4.3,
       fontSize: 11, fontFace: "Calibri", color: C.textDk });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 6.55, w: W - 1, h: 0.7, rectRadius: 0.08,
    fill: { color: C.navy }, line: { color: C.navy, width: 0 },
  });
  s.addText(
    "3 métricas independientes: fitness (Módulo 6), diversidad genotípica y presión selectiva teórica (rank, Módulo 4). El ciclo cae → inyecta → recupera se ve 5 veces gracias a los inmigrantes adaptativos (Módulo 5).",
    { x: 0.7, y: 6.6, w: W - 1.4, h: 0.6,
      fontSize: 13, fontFace: "Cambria", italic: true, color: C.textLt,
      align: "center", valign: "middle" }
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — 7 · CRITERIO DE PARADA (experimento 120/300/500 gen)
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "7", "Criterio de parada — ¿cuántas generaciones bastan?");

  s.addText("Corrimos 3 experimentos con 120, 300 y 500 generaciones para encontrar el punto óptimo de parada.", {
    x: 0.5, y: 1.15, w: W - 1, h: 0.35,
    fontSize: 13, fontFace: "Calibri", italic: true, color: C.textMd });

  // Tabla resumen
  const stopTbl = [
    [
      { text: "Generaciones", options: { bold: true, fill: { color: C.navy }, color: C.textLt } },
      { text: "Fitness final", options: { bold: true, fill: { color: C.navy }, color: C.textLt, align: "right" } },
      { text: "F1 test", options: { bold: true, fill: { color: C.navy }, color: C.textLt, align: "right" } },
      { text: "Tiempo CPU", options: { bold: true, fill: { color: C.navy }, color: C.textLt, align: "right" } },
      { text: "Tasa mejora", options: { bold: true, fill: { color: C.navy }, color: C.textLt, align: "right" } },
    ],
    [{ text: "120" }, { text: "0.132", options: { align: "right" } }, { text: "0.0998", options: { align: "right" } }, { text: "3 min", options: { align: "right" } }, { text: "0.0011/gen", options: { align: "right" } }],
    [
      { text: "300  ★", options: { bold: true, color: C.green, fill: { color: C.cardGr } } },
      { text: "0.164", options: { align: "right", bold: true, color: C.green, fill: { color: C.cardGr } } },
      { text: "0.1091", options: { align: "right", bold: true, color: C.green, fill: { color: C.cardGr } } },
      { text: "7.5 min", options: { align: "right", bold: true, color: C.green, fill: { color: C.cardGr } } },
      { text: "0.00018/gen", options: { align: "right", bold: true, color: C.green, fill: { color: C.cardGr } } },
    ],
    [{ text: "500" }, { text: "0.166", options: { align: "right" } }, { text: "0.1089", options: { align: "right" } }, { text: "12.7 min", options: { align: "right" } }, { text: "0.00001/gen", options: { align: "right" } }],
  ];
  s.addTable(stopTbl, {
    x: 0.5, y: 1.6, w: W - 1, colW: [2.4, 2.6, 2.4, 2.4, 2.5],
    fontSize: 13, fontFace: "Calibri", color: C.textDk,
    border: { pt: 0.5, color: C.line },
    rowH: 0.5,
  });

  // 2 hallazgos clave abajo
  card(s, 0.5, 4.0, 6.1, 3.0, C.cardGr);
  s.addText("📈 Hallazgo 1: rendimiento marginal decreciente", {
    x: 0.7, y: 4.15, w: 5.7, h: 0.45,
    fontSize: 15, fontFace: "Calibri", bold: true, color: C.green, margin: 0 });
  s.addText([
    { text: "La tasa de mejora cae 110× entre el inicio y el final.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "120 → 300 gen: +0.014 (mejora útil).", options: { bullet: true, breakLine: true } },
    { text: "300 → 500 gen: solo +0.002 (casi cero).", options: { bullet: true } },
  ], { x: 0.7, y: 4.65, w: 5.7, h: 2.3, fontSize: 13, color: C.textDk, paraSpaceAfter: 4 });

  card(s, 6.7, 4.0, 6.1, 3.0, C.cardAlt);
  s.addText("⚠️ Hallazgo 2: overfitting al subset de fitness", {
    x: 6.9, y: 4.15, w: 5.7, h: 0.45,
    fontSize: 15, fontFace: "Calibri", bold: true, color: C.red, margin: 0 });
  s.addText([
    { text: "En 500 gen, el fitness sigue subiendo (0.164 → 0.166)", options: { bullet: true, breakLine: true } },
    { text: "pero el F1 de test BAJA (0.1091 → 0.1089).", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "El AG empieza a sobreajustarse al subset de fitness.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "300 generaciones = mejor F1 con menor tiempo.", options: { bold: true, color: C.green } },
  ], { x: 6.9, y: 4.65, w: 5.7, h: 2.3, fontSize: 13, color: C.textDk, paraSpaceAfter: 4 });
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — 8 · IMPACTO DEL AG (ABLACIÓN)
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "8", "Impacto del AG en el sistema (ablación)");

  s.addText(
    "Le sacamos al AG un componente a la vez. Mismo presupuesto: 2 040 evaluaciones de fitness en todas las configuraciones.",
    { x: 0.5, y: 1.15, w: W - 1, h: 0.35,
      fontSize: 13, fontFace: "Calibri", italic: true, color: C.textMd }
  );

  s.addImage({
    path: path.join(FIG, "ablation.png"),
    x: 0.5, y: 1.55, w: 7.5, h: 4.0, sizing: { type: "contain", w: 7.5, h: 4.0 },
  });

  const tbl = [
    [
      { text: "Configuración", options: { bold: true, fill: { color: C.navy }, color: C.textLt } },
      { text: "Fitness", options: { bold: true, fill: { color: C.navy }, color: C.textLt, align: "right" } },
      { text: "F1 test", options: { bold: true, fill: { color: C.navy }, color: C.textLt, align: "right" } },
    ],
    [{ text: "Random search (sin AG)", options: { color: C.textMd } }, { text: "0.061", options: { align: "right", color: C.textMd } }, { text: "0.051", options: { align: "right", color: C.textMd } }],
    [{ text: "Solo mutación" }, { text: "0.082", options: { align: "right" } }, { text: "0.059", options: { align: "right" } }],
    [{ text: "Solo cruce" }, { text: "0.096", options: { align: "right" } }, { text: "0.066", options: { align: "right" } }],
    [{ text: "Sin elitismo" }, { text: "0.102", options: { align: "right" } }, { text: "0.075", options: { align: "right" } }],
    [
      { text: "★ AG COMPLETO", options: { bold: true, color: C.red, fill: { color: C.cardAlt } } },
      { text: "0.100", options: { align: "right", bold: true, color: C.red, fill: { color: C.cardAlt } } },
      { text: "0.065", options: { align: "right", bold: true, color: C.red, fill: { color: C.cardAlt } } },
    ],
  ];
  s.addTable(tbl, {
    x: 8.3, y: 1.55, w: 4.5, colW: [2.4, 1.05, 1.05],
    fontSize: 12, fontFace: "Calibri", color: C.textDk,
    border: { pt: 0.5, color: C.line },
    rowH: 0.42,
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 5.7, w: W - 1, h: 1.6, rectRadius: 0.1,
    fill: { color: C.card }, line: { color: C.card, width: 0 },
  });
  s.addText([
    { text: "Random search → AG completo: F1 +27 %.", options: { bold: true, color: C.red, breakLine: true } },
    { text: "Con el mismo presupuesto de evaluaciones, la evolución hace algo más que muestrear al azar — la presión de selección + los operadores guían la búsqueda.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Cruce vs mutación.", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "Solo cruce (0.096) > solo mutación (0.082). Coincide con Goldberg (SGA, Módulo 2): cruce es primario, mutación es secundario.", options: { } },
  ], { x: 0.8, y: 5.8, w: W - 1.6, h: 1.4, fontSize: 12, fontFace: "Calibri", color: C.textDk });
}

// ═════════════════════════════════════════════════════════════════════════════
// HELPER para slides de experimentos (9a/9b/9c)
// ═════════════════════════════════════════════════════════════════════════════
function experimentSlide(num, title, imgPath, tbl, takeaway, takeawayColor) {
  const s = pres.addSlide();
  header(s, num, title);

  s.addImage({
    path: imgPath,
    x: 0.5, y: 1.4, w: 8.0, h: 4.6, sizing: { type: "contain", w: 8.0, h: 4.6 },
  });

  s.addTable(tbl, {
    x: 8.8, y: 1.55, w: 4.0, colW: [1.5, 1.25, 1.25],
    fontSize: 12, fontFace: "Calibri", color: C.textDk,
    border: { pt: 0.5, color: C.line },
    rowH: 0.4,
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 6.2, w: W - 1, h: 1.1, rectRadius: 0.1,
    fill: { color: takeawayColor }, line: { color: takeawayColor, width: 0 },
  });
  s.addText(takeaway, {
    x: 0.8, y: 6.3, w: W - 1.6, h: 0.9,
    fontSize: 13, fontFace: "Calibri", color: C.textDk, valign: "middle",
  });
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — 9a · EXPERIMENTO MUTACIÓN
// ═════════════════════════════════════════════════════════════════════════════
experimentSlide(
  "9a", "Experimento — probabilidad de mutación",
  path.join(FIG, "exp_mutation.png"),
  [
    [
      { text: "p_mutation", options: { bold: true, fill: { color: C.navy }, color: C.textLt } },
      { text: "Fitness", options: { bold: true, fill: { color: C.navy }, color: C.textLt, align: "right" } },
      { text: "F1 test", options: { bold: true, fill: { color: C.navy }, color: C.textLt, align: "right" } },
    ],
    [{ text: "0.01" }, { text: "0.092", options: { align: "right" } }, { text: "0.069", options: { align: "right" } }],
    [{ text: "0.03" }, { text: "0.094", options: { align: "right" } }, { text: "0.070", options: { align: "right" } }],
    [{ text: "0.05" }, { text: "0.100", options: { align: "right" } }, { text: "0.065", options: { align: "right" } }],
    [
      { text: "0.10  ★", options: { bold: true, color: C.red, fill: { color: C.cardAlt } } },
      { text: "0.103", options: { align: "right", bold: true, color: C.red, fill: { color: C.cardAlt } } },
      { text: "0.068", options: { align: "right", bold: true, color: C.red, fill: { color: C.cardAlt } } },
    ],
    [{ text: "0.20" }, { text: "0.093", options: { align: "right" } }, { text: "0.065", options: { align: "right" } }],
    [{ text: "0.30" }, { text: "0.092", options: { align: "right" } }, { text: "0.071", options: { align: "right" } }],
  ],
  [
    { text: "Patrón en U invertida — exactamente lo que predice la teoría:", options: { bold: true, color: C.red, breakLine: true } },
    { text: "Mutación muy baja (0.01) → no explora. Óptima (0.05-0.10) → balance exploración/explotación. Alta (0.20-0.30) → destruye soluciones buenas más rápido que las construye.", options: {} },
  ],
  C.cardAlt
);

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 11 — 9b · EXPERIMENTO PROBABILIDAD DE CRUCE
// ═════════════════════════════════════════════════════════════════════════════
experimentSlide(
  "9b", "Experimento — probabilidad de cruce",
  path.join(FIG, "exp_crossover_prob.png"),
  [
    [
      { text: "p_crossover", options: { bold: true, fill: { color: C.navy }, color: C.textLt } },
      { text: "Fitness", options: { bold: true, fill: { color: C.navy }, color: C.textLt, align: "right" } },
      { text: "F1 test", options: { bold: true, fill: { color: C.navy }, color: C.textLt, align: "right" } },
    ],
    [{ text: "0.30" }, { text: "0.086", options: { align: "right" } }, { text: "0.061", options: { align: "right" } }],
    [{ text: "0.60" }, { text: "0.095", options: { align: "right" } }, { text: "0.069", options: { align: "right" } }],
    [{ text: "0.90" }, { text: "0.100", options: { align: "right" } }, { text: "0.065", options: { align: "right" } }],
    [
      { text: "1.00  ★", options: { bold: true, color: C.red, fill: { color: C.cardAlt } } },
      { text: "0.110", options: { align: "right", bold: true, color: C.red, fill: { color: C.cardAlt } } },
      { text: "0.077", options: { align: "right", bold: true, color: C.red, fill: { color: C.cardAlt } } },
    ],
  ],
  [
    { text: "Patrón monótono creciente:", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "Cuanta más recombinación, mejor. Confirma el rol del cruce como operador primario en alta dimensionalidad (7 946 dims).", options: {} },
  ],
  C.card
);

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 12 — 9c · OPERADOR DE CRUCE
// ═════════════════════════════════════════════════════════════════════════════
experimentSlide(
  "9c", "Experimento — operador de cruce",
  path.join(FIG, "exp_crossover_kind.png"),
  [
    [
      { text: "Operador", options: { bold: true, fill: { color: C.navy }, color: C.textLt } },
      { text: "Fitness", options: { bold: true, fill: { color: C.navy }, color: C.textLt, align: "right" } },
      { text: "F1 test", options: { bold: true, fill: { color: C.navy }, color: C.textLt, align: "right" } },
    ],
    [
      { text: "uniform  ★", options: { bold: true, color: C.red, fill: { color: C.cardAlt } } },
      { text: "0.100", options: { align: "right", bold: true, color: C.red, fill: { color: C.cardAlt } } },
      { text: "0.065", options: { align: "right", bold: true, color: C.red, fill: { color: C.cardAlt } } },
    ],
    [{ text: "one_point" }, { text: "0.086", options: { align: "right" } }, { text: "0.060", options: { align: "right" } }],
    [{ text: "arithmetic" }, { text: "0.086", options: { align: "right" } }, { text: "0.064", options: { align: "right" } }],
  ],
  [
    { text: "Uniforme gana en alta dimensión:", options: { bold: true, color: C.red, breakLine: true } },
    { text: "Con 7 946 dimensiones, el corte de un punto rompe bloques útiles dispersos. El aritmético queda \"entre los padres\" → poca exploración.", options: { breakLine: true } },
    { text: "Configuración final elegida: p_mutation = 0.05, p_crossover = 0.9, operador = uniforme.", options: { bold: true, color: C.navy } },
  ],
  C.cardAlt
);

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 13 — 10 · NSGA-II (Módulo 7 — multi-objetivo)
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "10", "Multi-objetivo: NSGA-II (Módulo 7)");

  s.addText("Precisión y recall están en conflicto: un solo F1 esconde el compromiso.", {
    x: 0.5, y: 1.15, w: W - 1, h: 0.35,
    fontSize: 13, fontFace: "Calibri", italic: true, color: C.textMd });

  // Imagen del frente de Pareto
  s.addImage({
    path: path.join(FIG, "nsga2_pareto_front.png"),
    x: 0.5, y: 1.55, w: 7.5, h: 5.0, sizing: { type: "contain", w: 7.5, h: 5.0 },
  });

  // Panel derecho
  card(s, 8.3, 1.55, 4.5, 5.0, C.cardPu);
  s.addText("Qué cambia respecto al AG normal", {
    x: 8.5, y: 1.7, w: 4.2, h: 0.4,
    fontSize: 14, fontFace: "Calibri", bold: true, color: C.purple, margin: 0 });

  s.addText([
    { text: "Antes (AG mono-objetivo):", options: { bold: true, breakLine: true } },
    { text: "  un solo número (F1) por individuo.", options: { bullet: true, breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Ahora (NSGA-II):", options: { bold: true, breakLine: true } },
    { text: "  cada individuo tiene 2 fitness (precisión y recall).", options: { bullet: true, breakLine: true } },
    { text: "  selección por non-dominated sort.", options: { bullet: true, breakLine: true } },
    { text: "  diversidad mantenida por crowding distance.", options: { bullet: true, breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Resultado:", options: { bold: true, color: C.purple, breakLine: true } },
    { text: "  ~21 soluciones no dominadas — el frente de Pareto.", options: { bullet: true, breakLine: true } },
    { text: "  El usuario elige el punto según prefiera precisión o cobertura.", options: { bullet: true } },
  ], { x: 8.5, y: 2.15, w: 4.2, h: 4.3, fontSize: 11, fontFace: "Calibri", color: C.textDk, paraSpaceAfter: 4 });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 6.7, w: W - 1, h: 0.6, rectRadius: 0.08,
    fill: { color: C.navy }, line: { color: C.navy, width: 0 },
  });
  s.addText(
    "Demuestra que la evolución también puede explorar trade-offs en lugar de optimizar un único objetivo.",
    { x: 0.7, y: 6.7, w: W - 1.4, h: 0.6,
      fontSize: 13, fontFace: "Cambria", italic: true, color: C.textLt,
      align: "center", valign: "middle", margin: 0 }
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 14 — 11 · XAI EVOLUTIVO (Módulo 9)
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "11", "XAI — explicabilidad del individuo (Módulo 9)");

  s.addText("Como la red es chica, sus pesos son inspeccionables: derivamos importancia por feature.", {
    x: 0.5, y: 1.15, w: W - 1, h: 0.35,
    fontSize: 13, fontFace: "Calibri", italic: true, color: C.textMd });

  s.addImage({
    path: path.join(FIG, "ga_feature_importance.png"),
    x: 0.5, y: 1.55, w: 8.0, h: 5.0, sizing: { type: "contain", w: 8.0, h: 5.0 },
  });

  card(s, 8.8, 1.55, 4.0, 5.0, C.cardGr);
  s.addText("Cómo se calcula", {
    x: 9.0, y: 1.7, w: 3.7, h: 0.4,
    fontSize: 14, fontFace: "Calibri", bold: true, color: C.green, margin: 0 });

  s.addText([
    { text: "Propagamos las magnitudes:", options: { breakLine: true } },
    { text: "  | W1 |  ·  | W2 |", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Cada feature (cada columna de entrada) recibe un score según cuánto pesa en las decisiones finales.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Qué pesa más", options: { bold: true, color: C.green, breakLine: true } },
    { text: "Fármacos concretos e indicaciones aparecen primero — el modelo aprendió relaciones interpretables.", options: {} },
  ], { x: 9.0, y: 2.15, w: 3.7, h: 4.3, fontSize: 11, fontFace: "Calibri", color: C.textDk, paraSpaceAfter: 4 });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 6.7, w: W - 1, h: 0.6, rectRadius: 0.08,
    fill: { color: C.navy }, line: { color: C.navy, width: 0 },
  });
  s.addText(
    "El individuo evolucionado NO es una caja negra: la representación elegida lo hace interpretable.",
    { x: 0.7, y: 6.7, w: W - 1.4, h: 0.6,
      fontSize: 13, fontFace: "Cambria", italic: true, color: C.textLt,
      align: "center", valign: "middle", margin: 0 }
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 15 — 12 · RESULTADOS FINALES
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "12", "Resultados finales");

  const stats = [
    ["F1-macro test", "0.109", "modelo evolucionado", C.red],
    ["Precision macro", "0.125", "del individuo ganador", C.blue],
    ["F1 vs SIDER 4.1", "0.32", "validación externa", C.green],
    ["Tiempo / generaciones", "7.5 min", "300 generaciones en CPU", C.gold],
  ];
  for (let i = 0; i < 4; i++) {
    const [label, val, sub, color] = stats[i];
    const x = 0.5 + i * 3.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 1.3, w: 3.0, h: 1.7, rectRadius: 0.1,
      fill: { color: "FFFFFF" }, line: { color, width: 2 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.08 },
    });
    s.addText(label, { x: x + 0.15, y: 1.4, w: 2.7, h: 0.35,
      fontSize: 11, fontFace: "Calibri", bold: true, color, charSpacing: 3,
      align: "center", margin: 0 });
    s.addText(val, { x: x + 0.15, y: 1.75, w: 2.7, h: 0.9,
      fontSize: 38, fontFace: "Cambria", bold: true, color: C.textDk,
      align: "center", valign: "middle", margin: 0 });
    s.addText(sub, { x: x + 0.15, y: 2.65, w: 2.7, h: 0.3,
      fontSize: 10, fontFace: "Calibri", italic: true, color: C.textMd,
      align: "center", margin: 0 });
  }

  s.addImage({
    path: path.join(FIG, "sider_validation.png"),
    x: 0.5, y: 3.25, w: 7.5, h: 3.7, sizing: { type: "contain", w: 7.5, h: 3.7 },
  });

  card(s, 8.3, 3.25, 4.5, 3.7);
  s.addText("Extra: selección de features con cromosoma binario", {
    x: 8.55, y: 3.4, w: 4.0, h: 0.6,
    fontSize: 14, fontFace: "Calibri", bold: true, color: C.purple, margin: 0 });

  s.addText([
    { text: "Otro genotipo, otro AG:", options: { bold: true, breakLine: true } },
    { text: "  cromosoma binario {0,1}^154", options: { bullet: true, breakLine: true } },
    { text: "  operadores canónicos: bit-flip + cruce uniforme", options: { bullet: true, breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Resultado:", options: { bold: true, color: C.purple, breakLine: true } },
    { text: "  el AG eligió 59 de 154 features", options: { bullet: true, breakLine: true } },
    { text: "  F1 mejoró 0.124 → 0.133", options: { bullet: true, breakLine: true } },
    { text: "  menos features y mejor desempeño", options: { bullet: true, breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Cubre Módulo 2 y 3 (genotipo binario).", options: { italic: true, color: C.textMd } },
  ], { x: 8.55, y: 4.0, w: 4.0, h: 2.9, fontSize: 11, fontFace: "Calibri", color: C.textDk });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 7.05, w: W - 1, h: 0.3, rectRadius: 0.06,
    fill: { color: C.navy }, line: { color: C.navy, width: 0 },
  });
  s.addText(
    "El modelo predice efectos que ya están documentados por farmacólogos en una base independiente — el AG aprendió relaciones reales.",
    { x: 0.7, y: 7.05, w: W - 1.4, h: 0.3,
      fontSize: 11, fontFace: "Cambria", italic: true, color: C.textLt,
      align: "center", valign: "middle", margin: 0 }
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// SLIDE 16 — CIERRE: mapeo a módulos
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addShape(pres.shapes.OVAL, {
    x: W - 4, y: -2, w: 6, h: 6,
    fill: { color: C.red, transparency: 88 }, line: { color: C.red, width: 0 },
  });

  s.addText("7 de los 9 módulos del programa cubiertos", {
    x: 0.7, y: 0.5, w: W - 1.4, h: 0.7,
    fontSize: 30, fontFace: "Calibri", bold: true,
    color: C.textLt, align: "left", valign: "middle", margin: 0,
  });
  s.addText("en un mismo proyecto coherente y verificable", {
    x: 0.7, y: 1.15, w: W - 1.4, h: 0.5,
    fontSize: 18, fontFace: "Calibri", italic: true,
    color: "CADCFC", align: "left", valign: "middle", margin: 0,
  });

  const modTbl = [
    [
      { text: "Módulo", options: { bold: true, fill: { color: C.red }, color: C.textLt } },
      { text: "Cómo se cubre", options: { bold: true, fill: { color: C.red }, color: C.textLt } },
    ],
    [{ text: "2 — Algoritmos Genéticos" }, { text: "Bucle completo SGA + elitismo" }],
    [{ text: "3 — Genotipos no binarios" }, { text: "Genotipo real (pesos) + binario (selección de features)" }],
    [{ text: "4 — Selección, fitness, constraints" }, { text: "Torneo, F1-macro con umbral por etiqueta, exclusión de no-ADRs" }],
    [{ text: "5 — Parameter tuning and control" }, { text: "σ determinístico + inmigrantes adaptativos + sweep de hiperparámetros" }],
    [{ text: "6 — Performance metrics (single-obj)" }, { text: "Best/mean fitness, diversidad, presión de selección" }],
    [{ text: "7 — Multi-objective EAs" }, { text: "NSGA-II para precisión vs recall (frente de Pareto)" }],
    [{ text: "9 — XAI + EC" }, { text: "Importancia de features derivada del individuo evolucionado" }],
  ];
  s.addTable(modTbl, {
    x: 0.7, y: 1.95, w: 12, colW: [4.5, 7.5],
    fontSize: 13, fontFace: "Calibri", color: C.textLt,
    fill: { color: "21295C" },
    border: { pt: 0.5, color: "3C4A6B" },
    rowH: 0.42,
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.7, y: 5.85, w: 12, h: 1.3, rectRadius: 0.1,
    fill: { color: "FFFFFF" }, line: { color: "FFFFFF", width: 0 },
  });
  s.addText(
    "Demostramos que un Algoritmo Genético puede entrenar un clasificador real multi-label sin gradiente, validado contra una base externa, con resultados reproducibles.",
    { x: 0.95, y: 5.95, w: 11.5, h: 1.1,
      fontSize: 16, fontFace: "Cambria", italic: true, color: C.navy,
      align: "center", valign: "middle" }
  );

  s.addText("¡Gracias!", {
    x: 0.7, y: 7.2, w: 12, h: 0.3,
    fontSize: 14, fontFace: "Calibri", bold: true,
    color: "CADCFC", align: "center", margin: 0,
  });
}

pres.writeFile({ fileName: path.join(ROOT, "PRESENTACION.pptx") })
  .then(f => console.log("Generado:", f));
