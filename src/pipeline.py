"""Pipeline completo en una sola función (lo usa la app).

El notebook ejecuta los mismos pasos uno por uno para poder explicarlos.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import betting, data, explain, features, model


@dataclass
class Bundle:
    history: pd.DataFrame
    dataset: pd.DataFrame
    state: features.TeamState
    test: pd.DataFrame
    test_proba: np.ndarray
    metrics: pd.DataFrame
    curves: pd.DataFrame
    bt_summary: pd.DataFrame
    final_model: object
    explainer: object
    importance: pd.DataFrame
    fixtures: pd.DataFrame
    fixtures_proba: np.ndarray
    n_trees: int


def run_pipeline(cache_dir: str = "data/raw", verbose: bool = False) -> Bundle:
    # 1) Datos
    hist = data.load_history(cache_dir=cache_dir, verbose=verbose)
    # 2) Features sin fuga
    ds, state = features.build_dataset(hist)
    # 3) Split temporal + entrenamiento con early stopping
    train, valid, test = model.temporal_split(ds)
    m = model.train_model(train, valid)
    n_trees = model.best_n_trees(m)
    # 4) Evaluación honesta en temporadas que el modelo nunca vio
    test_proba = model.predict_proba(m, test)
    freq = train["y"].value_counts(normalize=True).sort_index().to_numpy()
    metrics = model.evaluate(test, test_proba, freq)
    curves, bt = betting.backtest(test, test_proba)
    # 5) Modelo final con todo el histórico (para partidos de hoy)
    final = model.train_final(ds, n_trees)
    expl = explain.make_explainer(final)
    imp = explain.global_importance(explain.shap_values(expl, test))
    # 6) Partidos próximos
    fx = features.features_for_fixtures(data.load_fixtures(cache_dir=cache_dir), state)
    fx_proba = model.predict_proba(final, fx) if len(fx) else np.empty((0, 3))
    return Bundle(hist, ds, state, test, test_proba, metrics, curves, bt,
                  final, expl, imp, fx, fx_proba, n_trees)
