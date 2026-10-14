"""Corre el pipeline con datos reales y exporta resultados para el equipo.

Salida en la carpeta indicada (por defecto results/):
  - resultados.json  : métricas, backtest, importancia, ejemplo real, próximos partidos
  - bankroll.png     : curvas de capital del backtest
  - importancia.png  : variables más importantes (SHAP)
  - calibracion.png  : calibración por clase

Uso:  python scripts/export_results.py [carpeta_salida]
GitHub Actions lo corre solo (.github/workflows/run_model.yml) y publica en la rama "resultados".
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import betting, config, explain, model  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
OUT.mkdir(parents=True, exist_ok=True)
INK, BLUE, ORANGE, GREY, BG = "#14261F", "#2457B8", "#C26A12", "#8A948F", "#F6F4EE"


def r(x, n=4):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, n)


B = run_pipeline(cache_dir=os.environ.get("DATA_DIR", "data/raw"), verbose=True)
hist, test, ds = B.history, B.test, B.dataset
train, valid, _ = model.temporal_split(ds)

res = {
    "generado": f"{np.datetime64('now', 's')}Z",
    "datos": {
        "partidos": int(len(hist)),
        "desde": f"{hist['Date'].min():%Y-%m-%d}",
        "hasta": f"{hist['Date'].max():%Y-%m-%d}",
        "temporadas": sorted(hist["Season"].unique().tolist()),
        "por_liga": {config.LEAGUES[d].split(" (")[0]: {
            "partidos": int(len(g)),
            "local": r((g["FTR"] == "H").mean()), "empate": r((g["FTR"] == "D").mean()),
            "visitante": r((g["FTR"] == "A").mean()),
            "margen_bet365": r(np.nanmean(betting.overround(g[betting.ODDS].dropna().to_numpy(dtype=float)))),
        } for d, g in hist.groupby("Div")},
    },
    "split": {name: {"temporadas": [d["Season"].min(), d["Season"].max()], "partidos": int(len(d))}
              for name, d in [("entrenamiento", train), ("validacion", valid), ("prueba", test)]},
    "arboles": B.n_trees,
    "metricas": B.metrics.to_dict(orient="records"),
    "backtest": B.bt_summary.to_dict(orient="records"),
    "backtest_aciertos": r(B.bt_summary.attrs.get("hit_rate")),
    "backtest_apuestas_valor": B.bt_summary.attrs.get("n_value_bets"),
    "importancia": [{"variable": row.Variable, "total": r(row.Total)} for row in B.importance.head(10).itertuples()],
}

# Curva de aprendizaje: error (log loss) en entrenamiento y validación según el número de árboles
from xgboost import XGBClassifier  # noqa: E402
from src.features import FEATURES  # noqa: E402
lc_params = {k: v for k, v in config.XGB_PARAMS.items() if k != "early_stopping_rounds"}
lc = XGBClassifier(**lc_params)
lc.fit(train[FEATURES], train["y"], eval_set=[(train[FEATURES], train["y"]), (valid[FEATURES], valid["y"])], verbose=False)
er = lc.evals_result()
tr, va = er["validation_0"]["mlogloss"], er["validation_1"]["mlogloss"]
res["curva_aprendizaje"] = {"arboles": list(range(1, len(tr) + 1, 5)), "entrenamiento": [r(x) for x in tr[::5]],
                            "validacion": [r(x) for x in va[::5]], "arboles_elegidos": B.n_trees}

# Sensibilidad al umbral: ROI plano ($1 por apuesta, sin bankroll, para comparar umbrales de forma limpia)
_o = test[betting.ODDS].to_numpy(dtype=float)
_ok = np.all(np.isfinite(_o) & (_o > 1), axis=1)
_ev = np.where(_o <= config.MAX_ODDS, B.test_proba * _o - 1, -np.inf)
_best = _ev.argmax(1); _bev = _ev[np.arange(len(test)), _best]
_won = _best == test["y"].to_numpy(); _ob = _o[np.arange(len(test)), _best]
sens = []
for t in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    mk = _ok & (_bev >= t)
    n = int(mk.sum())
    pnl = np.where(_won[mk], _ob[mk] - 1, -1.0)
    sens.append({"umbral": t, "apuestas": n, "roi": r(pnl.mean() * 100 if n else np.nan, 2),
                 "aciertos": r(_won[mk].mean() if n else np.nan), "cuota_media": r(_ob[mk].mean() if n else np.nan, 2)})
res["sensibilidad_umbral"] = sens

# ¿Sirve la cuota máxima? ROI plano por tramo de cuota, SIN tope de cuota (EV >= 5%)
_evn = B.test_proba * _o - 1
_bn = _evn.argmax(1); _bevn = _evn[np.arange(len(test)), _bn]
_wn = _bn == test["y"].to_numpy(); _obn = _o[np.arange(len(test)), _bn]
_pm = B.test_proba[np.arange(len(test)), _bn]
_fair = np.full_like(B.test_proba, np.nan)
_fair[_ok] = betting.fair_probs(_o[_ok])
_pf = _fair[np.arange(len(test)), _bn]
tramos = []
for lo, hi, lab in [(1.0, 2.0, "1-2"), (2.0, 3.0, "2-3"), (3.0, 5.0, "3-5"), (5.0, 10.0, "5-10"), (10.0, 1e9, "10+")]:
    mk = _ok & (_bevn >= config.MIN_EV) & (_obn >= lo) & (_obn < hi)
    n = int(mk.sum())
    pnl = np.where(_wn[mk], _obn[mk] - 1, -1.0)
    tramos.append({"tramo": lab, "apuestas": n, "roi": r(pnl.mean() * 100 if n else np.nan, 2),
                   "aciertos": r(_wn[mk].mean() if n else np.nan), "prob_modelo": r(_pm[mk].mean() if n else np.nan),
                   "prob_mercado": r(np.nanmean(_pf[mk]) if n else np.nan)})
res["roi_por_cuota"] = tramos

# Ejemplo real del set de prueba: el partido más reciente con valor y EV moderado (5%–15%)
odds = test[betting.ODDS].to_numpy(dtype=float)
ev = np.where(odds <= config.MAX_ODDS, B.test_proba * odds - 1, -np.inf)
best = ev.max(1)
cand = np.where(np.isfinite(best) & (best >= config.MIN_EV) & (best <= 0.15))[0]
if len(cand):
    i = int(cand[-1])
    row = test.iloc[i]
    d = betting.decide(B.test_proba[i], odds[i])
    sv = explain.shap_values(B.explainer, test.iloc[[i]])[0]
    cls = d["bet"] if d["bet"] is not None else int(np.argmax(B.test_proba[i]))
    contrib = explain.explain_one(sv, row, cls, top=5)
    base = np.ravel(B.explainer.expected_value)
    res["ejemplo"] = {
        "fecha": f"{row['Date']:%Y-%m-%d}", "liga": config.LEAGUES[row["Div"]].split(" (")[0],
        "local": row["HomeTeam"], "visitante": row["AwayTeam"], "resultado_real": row["FTR"],
        "goles": [int(row["FTHG"]), int(row["FTAG"])],
        "cuotas": [r(x, 2) for x in odds[i]], "p_modelo": [r(x) for x in d["p_model"]],
        "p_mercado": [r(x) for x in d["p_fair"]], "ev": [r(x) for x in d["ev"]],
        "kelly_completo": [r(x) for x in d["kelly_full"]],
        "margen": r(d["overround"]), "apuesta": None if d["bet"] is None else config.CLASS_NAMES[d["bet"]],
        "stake": r(d["stake"]), "razon": d["reason"],
        "shap": [{"variable": c.Variable, "valor": r(c.Valor, 3), "impacto": r(c.Impacto)} for c in contrib.itertuples()],
        "shap_base": [r(x) for x in base], "shap_clase": config.CLASS_NAMES[cls],
        "shap_total": r(float(sv[:, cls].sum())),
    }

fx = []
for i, row in B.fixtures.iterrows():
    d = betting.decide(B.fixtures_proba[i], row[betting.ODDS].to_numpy(dtype=float))
    fx.append({"fecha": f"{row['Date']:%Y-%m-%d}", "liga": config.LEAGUES[row["Div"]].split(" (")[0],
               "local": row["HomeTeam"], "visitante": row["AwayTeam"],
               "p": [r(x, 3) for x in B.fixtures_proba[i]],
               "apuesta": None if d["bet"] is None else config.CLASS_NAMES[d["bet"]],
               "ev": None if d["bet"] is None else r(d["ev"][d["bet"]])})
res["proximos"] = fx

(OUT / "resultados.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

# ------------------------------------------------------------------------- gráficos
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.facecolor": BG, "figure.facecolor": BG})
curves = B.curves.groupby(level=0).last()
fig, ax = plt.subplots(figsize=(12, 5.2))
colors = {"Apuesta fija 1%": GREY, "Kelly completo": ORANGE, "Kelly 1/2": "#6C8FD6", "Kelly 1/4": BLUE}
for c in curves.columns:
    ax.plot(curves.index, curves[c], label=c, color=colors.get(c, INK), lw=2.4 if c == "Kelly 1/4" else 1.6)
ax.axhline(1000, color=INK, ls="--", lw=0.8)
ax.set_ylabel("Capital ($)")
ax.legend(frameon=False, ncol=4, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "bankroll.png", dpi=160); plt.close(fig)

imp = B.importance.head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.barh(imp["Variable"], imp["Total"], color=BLUE)
ax.set_xlabel("Importancia promedio |SHAP|")
fig.tight_layout(); fig.savefig(OUT / "importancia.png", dpi=160); plt.close(fig)

cal = model.calibration_table(test["y"].to_numpy(), B.test_proba)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
for ax, (cls, g) in zip(axes, cal.groupby("Clase", sort=False)):
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    ax.plot(g["Prob. predicha"], g["Frecuencia real"], "o-", color=BLUE)
    ax.set_title(cls); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Predicha")
axes[0].set_ylabel("Real")
fig.tight_layout(); fig.savefig(OUT / "calibracion.png", dpi=160); plt.close(fig)

print(json.dumps({k: res[k] for k in ["metricas", "backtest"]}, ensure_ascii=False, indent=1))
