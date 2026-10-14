"""Ejemplo documentado: cómo decide XGBoost en UN partido real del set de prueba.

Exporta (en la carpeta indicada) ejemplo_xgboost.json con:
  - el partido, sus cuotas y el resultado real
  - las 19 variables que vio el modelo
  - el camino real del partido por los árboles de las rondas 1 y 2
  - cómo cambian las probabilidades ronda a ronda
  - la contribución SHAP de cada variable
  - la decisión final (valor esperado y Kelly)

Usa el modelo de EVALUACIÓN (entrenado sin ver la prueba), así el ejemplo es honesto.
Uso:  python scripts/export_ejemplo.py [carpeta_salida]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import betting, config, data, explain, features, model  # noqa: E402
from src.features import FEATURE_LABELS, FEATURES  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
OUT.mkdir(parents=True, exist_ok=True)
r = lambda x, n=4: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)  # noqa: E731

hist = data.load_history(cache_dir=os.environ.get("DATA_DIR", "data/raw"), verbose=True)
ds, _ = features.build_dataset(hist)
train, valid, test = model.temporal_split(ds)

# 1) Mismo modelo que el de evaluación, re-entrenado con exactamente las rondas elegidas
m0 = model.train_model(train, valid)
best = model.best_n_trees(m0)
params = {k: v for k, v in config.XGB_PARAMS.items() if k != "early_stopping_rounds"}
params["n_estimators"] = best
m = xgb.XGBClassifier(**params).fit(train[FEATURES], train["y"], verbose=False)
booster = m.get_booster()

# 2) Elegir un partido conocido de la prueba (el más reciente de la lista)
PAIRS = [("Real Madrid", "Barcelona"), ("Barcelona", "Real Madrid"), ("Liverpool", "Man City"),
         ("Man City", "Liverpool"), ("Arsenal", "Man City"), ("Man City", "Arsenal"), ("Man United", "Liverpool"),
         ("Liverpool", "Man United"), ("Inter", "Milan"), ("Milan", "Inter"), ("Bayern Munich", "Dortmund"),
         ("Dortmund", "Bayern Munich"), ("Paris SG", "Marseille"), ("Marseille", "Paris SG")]
mask = [(h, a) in PAIRS for h, a in zip(test["HomeTeam"], test["AwayTeam"])]
cands = test[mask]
cands = cands[np.all(np.isfinite(cands[betting.ODDS].to_numpy(dtype=float)), axis=1)]
row = cands.iloc[-1] if len(cands) else test.iloc[-1]
X = test.loc[[row.name], FEATURES]
dm = xgb.DMatrix(X, feature_names=FEATURES)

# 3) Camino real por los árboles de las rondas 1 y 2 (3 árboles por ronda: Local, Empate, Visitante)
tdf = booster.trees_to_dataframe()

# Árboles completos de la ronda 1 (para dibujarlos enteros en la presentación)
arboles_r1 = {}
for k, cls in enumerate(config.CLASS_NAMES):
    nodes = []
    for n in tdf[tdf["Tree"] == k].itertuples():
        leaf = n.Feature == "Leaf"
        nodes.append({"id": n.ID, "variable": None if leaf else FEATURE_LABELS.get(n.Feature, n.Feature),
                      "feature": None if leaf else n.Feature, "corte": None if leaf else r(n.Split, 3),
                      "si": None if leaf else n.Yes, "no": None if leaf else n.No,
                      "hoja": r(n.Gain, 4) if leaf else None, "cover": r(n.Cover, 1)})
    arboles_r1[cls] = nodes


def path(tree_id):
    t = tdf[tdf["Tree"] == tree_id].set_index("ID")
    node, steps = f"{tree_id}-0", []
    while True:
        n = t.loc[node]
        if n["Feature"] == "Leaf":
            return steps, float(n["Gain"])
        f, s = n["Feature"], float(n["Split"])
        v = float(X.iloc[0][f])
        if np.isnan(v):
            nxt, ans = n["Missing"], "sin dato"
        elif v < s:
            nxt, ans = n["Yes"], "sí"
        else:
            nxt, ans = n["No"], "no"
        steps.append({"variable": FEATURE_LABELS.get(f, f), "feature": f, "corte": r(s, 3), "valor": r(v, 3), "respuesta": ans})
        node = nxt


rondas = []
for rnd in [0, 1]:
    arboles = []
    for k, cls in enumerate(config.CLASS_NAMES):
        steps, leaf = path(rnd * 3 + k)
        arboles.append({"clase": cls, "pasos": steps, "hoja": r(leaf, 4)})
    rondas.append({"ronda": rnd + 1, "arboles": arboles})

# 4) Probabilidades y puntajes ronda a ronda
def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


traj = []
for k in sorted(set([1, 2, 3, 5, 10, 20, 50, 100, 150, best])):
    if k > best:
        continue
    mg = booster.predict(dm, output_margin=True, iteration_range=(0, k))[0]
    traj.append({"ronda": k, "puntajes": [r(x) for x in mg], "probs": [r(x) for x in softmax(mg)]})
mg1 = booster.predict(dm, output_margin=True, iteration_range=(0, 1))[0]
base_margin = [r(mg1[k] - rondas[0]["arboles"][k]["hoja"]) for k in range(3)]

# 5) SHAP del partido (con el mismo modelo)
expl = explain.make_explainer(m)
sv = explain.shap_values(expl, X)[0]
ev_base = np.ravel(expl.expected_value)
p = m.predict_proba(X)[0]
shap_out = {}
for k, cls in enumerate(config.CLASS_NAMES):
    order = np.argsort(-np.abs(sv[:, k]))
    shap_out[cls] = {"base": r(ev_base[k]), "suma": r(sv[:, k].sum()),
                     "top": [{"variable": FEATURE_LABELS[FEATURES[j]], "valor": r(float(X.iloc[0][FEATURES[j]]), 3),
                              "impacto": r(sv[j, k])} for j in order[:7]]}

# 6) Decisión
odds = row[betting.ODDS].to_numpy(dtype=float)
d = betting.decide(p, odds)

res = {
    "partido": {"fecha": f"{row['Date']:%Y-%m-%d}", "liga": config.LEAGUES[row["Div"]].split(" (")[0],
                "local": row["HomeTeam"], "visitante": row["AwayTeam"], "goles": [int(row["FTHG"]), int(row["FTAG"])],
                "resultado": {"H": "Local", "D": "Empate", "A": "Visitante"}[row["FTR"]]},
    "rondas_modelo": best,
    "variables": [{"feature": f, "variable": FEATURE_LABELS[f], "valor": r(float(X.iloc[0][f]), 3)} for f in FEATURES],
    "rondas": rondas,
    "arboles_ronda1": arboles_r1,
    "puntaje_inicial": base_margin,
    "trayectoria": traj,
    "prob_final": [r(x) for x in p],
    "shap": shap_out,
    "cuotas": [r(x, 2) for x in odds],
    "prob_mercado": [r(x) for x in d["p_fair"]],
    "margen": r(d["overround"]),
    "ev": [r(x) for x in d["ev"]],
    "kelly_completo": [r(x) for x in d["kelly_full"]],
    "apuesta": None if d["bet"] is None else config.CLASS_NAMES[d["bet"]],
    "stake": r(d["stake"]),
    "razon": d["reason"],
}
(OUT / "ejemplo_xgboost.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res["partido"], ensure_ascii=False), res["prob_final"], res["apuesta"])
