"""Entrenamiento y evaluación del XGBoost multiclase (Local / Empate / Visitante)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from . import config
from .betting import ODDS, fair_probs
from .features import FEATURES


def temporal_split(data: pd.DataFrame, valid_seasons: int = 1, test_seasons: int = 2,
                   burn_in_seasons: int = 1):
    """Divide por TEMPORADAS, nunca al azar.

    [burn-in descartado] [ entrenamiento ] [ validación ] [ prueba ]
    El burn-in se descarta porque al inicio el Elo y la forma aún no "aprendieron".
    """
    seasons = sorted(data["Season"].unique())
    if len(seasons) < burn_in_seasons + valid_seasons + test_seasons + 1:
        raise ValueError(f"Muy pocas temporadas ({len(seasons)}) para este split.")
    test_s = seasons[-test_seasons:]
    valid_s = seasons[-(test_seasons + valid_seasons):-test_seasons]
    train_s = seasons[burn_in_seasons:-(test_seasons + valid_seasons)]
    pick = lambda s: data[data["Season"].isin(s)].reset_index(drop=True)  # noqa: E731
    return pick(train_s), pick(valid_s), pick(test_s)


def train_model(train: pd.DataFrame, valid: pd.DataFrame, params: dict | None = None) -> XGBClassifier:
    """Entrena con early stopping: para cuando la validación deja de mejorar."""
    params = {**config.XGB_PARAMS, **(params or {})}
    model = XGBClassifier(**params)
    model.fit(train[FEATURES], train["y"],
              eval_set=[(valid[FEATURES], valid["y"])], verbose=False)
    return model


def train_final(data: pd.DataFrame, n_estimators: int, params: dict | None = None,
                burn_in_seasons: int = 1) -> XGBClassifier:
    """Re-entrena con TODO el histórico usando el nº de árboles encontrado."""
    seasons = sorted(data["Season"].unique())
    d = data[data["Season"].isin(seasons[burn_in_seasons:])]
    params = {**config.XGB_PARAMS, **(params or {}), "n_estimators": int(n_estimators)}
    params.pop("early_stopping_rounds", None)
    model = XGBClassifier(**params)
    model.fit(d[FEATURES], d["y"], verbose=False)
    return model


def best_n_trees(model: XGBClassifier) -> int:
    bi = getattr(model, "best_iteration", None)
    return int(bi) + 1 if bi is not None else int(model.get_params()["n_estimators"])


def predict_proba(model: XGBClassifier, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES])


def brier_multiclass(y: np.ndarray, p: np.ndarray) -> float:
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def evaluate(df: pd.DataFrame, proba: np.ndarray, train_freq: np.ndarray | None = None) -> pd.DataFrame:
    """Compara el modelo contra dos referencias:
    - Ingenuo: siempre predice la frecuencia histórica (≈45% L, 27% E, 28% V).
    - Mercado: probabilidades justas de Bet365 (la vara más difícil).
    Métricas: log loss y Brier (menor = mejor, miden calidad de PROBABILIDADES)
    y accuracy (aciertos del resultado más probable).
    """
    y = df["y"].to_numpy()
    rows = []
    has_odds = np.all(np.isfinite(df[ODDS].to_numpy(dtype=float)), axis=1)

    def add(name, p, mask):
        rows.append({
            "Modelo": name,
            "Partidos": int(mask.sum()),
            "Log loss": round(log_loss(y[mask], p[mask], labels=[0, 1, 2]), 4),
            "Brier": round(brier_multiclass(y[mask], p[mask]), 4),
            "Accuracy": round(accuracy_score(y[mask], p[mask].argmax(1)), 4),
        })

    if train_freq is not None:
        add("Ingenuo (frecuencias)", np.tile(train_freq, (len(df), 1)), has_odds)
    add("XGBoost (nuestro)", proba, has_odds)
    mkt = np.full_like(proba, np.nan)
    mkt[has_odds] = fair_probs(df.loc[has_odds, ODDS].to_numpy(dtype=float))
    add("Mercado Bet365 (sin margen)", mkt, has_odds)
    return pd.DataFrame(rows)


def calibration_table(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Si el modelo dice 30%, ¿pasa ~30% de las veces? (por clase)."""
    out = []
    bins = np.linspace(0, 1, n_bins + 1)
    for k, name in enumerate(config.CLASS_NAMES):
        p, hit = proba[:, k], (y == k).astype(float)
        idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            m = idx == b
            if m.sum() >= 20:
                out.append({"Clase": name, "Prob. predicha": p[m].mean(),
                            "Frecuencia real": hit[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(out)
