"""
XAI sobre el modelo evolucionado por el AG  (extra — Modulo 9: XAI + EC).

Una ventaja de optimizar una red CHICA con un AG es que el modelo resultante es
inspeccionable: sus pesos son pocos y estan a la vista. Aca derivamos la
IMPORTANCIA GLOBAL de cada feature de entrada propagando las magnitudes de los
pesos a traves de la red (metodo de pesos de conexion, estilo Garson/Olden):

    importancia(feature_i) = sum_h |W1[i,h]| * sum_o |W2[h,o]|

es decir, cuanto "empuja" cada feature de entrada hacia las salidas a traves de
las neuronas ocultas. Sirve para explicar QUE terminos del texto del paciente y
que variables demograficas guian las predicciones.

Salida: outputs/ga_feature_importance.csv + outputs/figures/ga_feature_importance.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUTS_DIR
from ga_model import load_ga_model

model, featurizer, labels, thresholds = load_ga_model()
names = featurizer.feature_names                     # longitud == input_dim

W1 = np.abs(model.W1)                                # [d, h]
W2 = np.abs(model.W2)                                # [h, o]
hidden_influence = W2.sum(axis=1)                    # [h] influencia de cada oculta en la salida
importance = W1 @ hidden_influence                   # [d] importancia por feature de entrada

imp = pd.DataFrame({"feature": names, "importance": importance})
imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
out_csv = OUTPUTS_DIR / "ga_feature_importance.csv"
imp.to_csv(out_csv, index=False)

print("=== XAI — Importancia global de features (modelo AG) ===")
print(imp.head(20).to_string(index=False))

top = imp.head(20).iloc[::-1]
plt.figure(figsize=(9, 7))
plt.barh(top["feature"], top["importance"], color="#9b59b6")
plt.title("Importancia de features — modelo evolucionado por AG (XAI)")
plt.xlabel("Importancia  ( |W1| propagada por |W2| )")
plt.tight_layout()
out_png = OUTPUTS_DIR / "figures" / "ga_feature_importance.png"
out_png.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png, dpi=150)

print(f"\nGuardado: {out_csv}")
print(f"Figura:   {out_png}")
