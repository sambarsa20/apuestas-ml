"""Explicabilidad con SHAP: ¿por qué el modelo dio esa probabilidad?

Feynman: imagina que el modelo arranca con una probabilidad "promedio" para
cualquier partido. Cada variable empuja esa probabilidad hacia arriba o hacia
abajo. SHAP mide cuánto empujó cada una. La suma de todos los empujones lleva
del promedio a la predicción final de ESTE partido.

Nota técnica: en XGBoost multiclase los empujones se miden en escala log-odds
(antes del softmax), no en puntos porcentuales; el signo y el tamaño relativo
son lo que interpretamos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from . import config
from .features import FEATURE_LABELS, FEATURES


def make_explainer(model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """Devuelve un arreglo (partidos, variables, clases) sin importar la versión de shap."""
    sv = explainer.shap_values(X[FEATURES])
    if isinstance(sv, list):                       # versiones antiguas: lista por clase
        sv = np.stack(sv, axis=-1)
    if sv.ndim == 2:
        sv = sv[..., None]
    return sv


def global_importance(sv: np.ndarray) -> pd.DataFrame:
    """Importancia promedio |SHAP| por variable y clase."""
    imp = np.abs(sv).mean(axis=0)                  # (variables, clases)
    df = pd.DataFrame(imp, index=FEATURES, columns=config.CLASS_NAMES)
    df["Total"] = df.sum(axis=1)
    df.insert(0, "Variable", [FEATURE_LABELS[f] for f in FEATURES])
    return df.sort_values("Total", ascending=False)


def explain_one(sv_row: np.ndarray, x_row: pd.Series, cls: int, top: int = 8) -> pd.DataFrame:
    """Contribuciones de cada variable hacia la clase `cls` para un partido."""
    df = pd.DataFrame({
        "feature": FEATURES,
        "Variable": [FEATURE_LABELS[f] for f in FEATURES],
        "Valor": [x_row[f] for f in FEATURES],
        "Impacto": sv_row[:, cls],
    })
    df = df[~df["feature"].str.startswith("div_") | (df["Impacto"].abs() > 1e-3)]
    df["abs"] = df["Impacto"].abs()
    return df.sort_values("abs", ascending=False).head(top).drop(columns="abs").reset_index(drop=True)


def _fmt(feature: str, v: float) -> str:
    if pd.isna(v):
        return "sin dato aún"
    if feature.startswith("div_"):
        return "sí" if v > 0.5 else "no"
    if feature == "elo_exp_home":
        return f"{v:.0%}"
    if feature in ("elo_diff", "home_elo", "away_elo"):
        return f"{v:+.0f}" if feature == "elo_diff" else f"{v:.0f}"
    return f"{v:.2f}"


def narrative(contrib: pd.DataFrame, cls: int, n: int = 3) -> list[str]:
    """Frases en español con las razones principales a favor y en contra."""
    name = config.CLASS_NAMES[cls]
    pro = contrib[contrib["Impacto"] > 0].head(n)
    con = contrib[contrib["Impacto"] < 0].head(2)
    lines = [f"**A favor de '{name}':**"]
    lines += [f"- {r.Variable}: {_fmt(r.feature, r.Valor)}" for r in pro.itertuples()] or ["- (ninguna variable fuerte)"]
    if len(con):
        lines.append(f"**En contra de '{name}':**")
        lines += [f"- {r.Variable}: {_fmt(r.feature, r.Valor)}" for r in con.itertuples()]
    return lines
