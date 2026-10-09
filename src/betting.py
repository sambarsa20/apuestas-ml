"""Del modelo a la decisión: probabilidad justa, valor esperado y Kelly.

Feynman en tres líneas:
  1. Una cuota es un PRECIO. Cuota 2.50 = el mercado cree que pasa ~40% (1/2.50).
  2. Las tres probabilidades implícitas suman >100%: ese exceso es la comisión
     de la casa (overround). La quitamos para tener la probabilidad "justa".
  3. Si NUESTRA probabilidad x cuota > 1, la apuesta tiene valor esperado positivo.
     Kelly dice qué % del capital arriesgar para crecer más rápido sin quebrar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

OUTCOMES = ["H", "D", "A"]
ODDS = ["B365H", "B365D", "B365A"]


def implied_probs(odds: np.ndarray) -> np.ndarray:
    """Probabilidad implícita bruta (incluye la comisión): 1 / cuota."""
    return 1.0 / np.asarray(odds, dtype=float)


def fair_probs(odds: np.ndarray) -> np.ndarray:
    """Quita el margen normalizando para que sumen 100% (método proporcional)."""
    imp = implied_probs(odds)
    return imp / imp.sum(axis=-1, keepdims=True)


def overround(odds: np.ndarray) -> np.ndarray:
    """Margen de la casa: suma de implícitas − 1 (ej. 0.05 = 5%)."""
    return implied_probs(odds).sum(axis=-1) - 1.0


def expected_value(p: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Ganancia esperada por cada $1 apostado: p·cuota − 1."""
    return np.asarray(p) * np.asarray(odds) - 1.0


def kelly_fraction(p: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """f* = (p·cuota − 1) / (cuota − 1). Si es negativa, no se apuesta (0)."""
    odds = np.asarray(odds, dtype=float)
    f = (np.asarray(p) * odds - 1.0) / (odds - 1.0)
    return np.clip(f, 0.0, None)


def decide(p_model: np.ndarray, odds: np.ndarray,
           min_ev: float = config.MIN_EV,
           kelly_mult: float = config.KELLY_FRACTION,
           max_stake: float = config.MAX_STAKE,
           max_odds: float = config.MAX_ODDS) -> dict:
    """Decisión para UN partido.

    Regla: entre los 3 resultados con cuota <= max_odds, elegir el de mayor valor
    esperado; apostar solo si EV >= min_ev. Monto = Kelly × fracción, con tope max_stake.
    """
    p_model = np.asarray(p_model, dtype=float)
    odds = np.asarray(odds, dtype=float)
    out = {
        "p_model": p_model, "p_fair": None, "ev": None, "kelly_full": None,
        "bet": None, "stake": 0.0, "reason": "",
    }
    if np.any(~np.isfinite(odds)) or np.any(odds <= 1.0):
        out["reason"] = "No hay cuotas válidas para este partido."
        return out

    p_fair = fair_probs(odds)
    ev = expected_value(p_model, odds)
    kf = kelly_fraction(p_model, odds)
    ev_ok = np.where(odds <= max_odds, ev, -np.inf)   # descarta batacazos
    best = int(np.argmax(ev_ok))
    out.update(p_fair=p_fair, ev=ev, kelly_full=kf, overround=float(overround(odds)))

    if ev_ok[best] >= min_ev:
        out["bet"] = best
        out["stake"] = float(min(kf[best] * kelly_mult, max_stake))
        out["reason"] = (
            f"El modelo estima {p_model[best]:.1%} para '{config.CLASS_NAMES[best]}' y el mercado "
            f"(sin comisión) {p_fair[best]:.1%}. Con cuota {odds[best]:.2f}, el valor esperado es "
            f"{ev[best]:+.1%} por cada $1, sobre el umbral de {min_ev:.0%}."
        )
    else:
        out["reason"] = (
            f"Ningún resultado supera el umbral de valor esperado ({min_ev:.0%}). El mejor es "
            f"'{config.CLASS_NAMES[best]}' con {ev[best]:+.1%}: el mercado ya lo paga a un precio justo "
            f"o caro. Decisión: NO APOSTAR."
        )
        skipped = [i for i in range(3) if odds[i] > max_odds and ev[i] >= min_ev]
        if skipped:
            out["reason"] += (
                f" (Hay EV aparente en '{config.CLASS_NAMES[skipped[0]]}' a cuota {odds[skipped[0]]:.2f}, "
                f"pero se descarta por superar la cuota máxima de {max_odds:.1f}: en batacazos el modelo "
                f"es poco confiable y la casa suele cobrar más margen.)"
            )
    return out


def backtest(df: pd.DataFrame, proba: np.ndarray,
             min_ev: float = config.MIN_EV,
             strategies: dict | None = None,
             max_stake: float = config.MAX_STAKE,
             max_odds: float = config.MAX_ODDS,
             start_bankroll: float = 1000.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simula apostar partido a partido en orden cronológico.

    Estrategias: apuesta fija (1% del bankroll inicial) y Kelly con distintas
    fracciones. Devuelve (curvas de bankroll, resumen de métricas).
    """
    strategies = strategies or {"Apuesta fija 1%": None, "Kelly completo": 1.0,
                                "Kelly 1/2": 0.5, "Kelly 1/4": 0.25}
    odds = df[ODDS].to_numpy(dtype=float)
    valid = np.all(np.isfinite(odds) & (odds > 1.0), axis=1)
    y = df["y"].to_numpy()

    ev = proba * odds - 1.0
    ev_ok = np.where(odds <= max_odds, ev, -np.inf)    # descarta batacazos
    best = np.argmax(ev_ok, axis=1)
    best_ev = ev_ok[np.arange(len(df)), best]
    bet_mask = valid & (best_ev >= min_ev)
    o_best = odds[np.arange(len(df)), best]
    p_best = proba[np.arange(len(df)), best]
    won = (best == y)

    curves, summary = {}, []
    for name, mult in strategies.items():
        bank, path, staked, n = start_bankroll, [], 0.0, 0
        peak, max_dd = start_bankroll, 0.0
        for i in range(len(df)):
            if bet_mask[i] and bank > 0:
                if mult is None:
                    stake = 0.01 * start_bankroll
                else:
                    f = (p_best[i] * o_best[i] - 1) / (o_best[i] - 1)
                    # Kelly completo SIN tope (para mostrar su riesgo); fraccional con tope
                    f = f * mult if mult == 1.0 else min(f * mult, max_stake)
                    stake = bank * max(f, 0.0)
                stake = min(stake, bank)
                bank += stake * (o_best[i] - 1) if won[i] else -stake
                staked += stake
                n += 1
            peak = max(peak, bank)
            max_dd = max(max_dd, (peak - bank) / peak if peak > 0 else 0)
            path.append(bank)
        curves[name] = path
        summary.append({
            "Estrategia": name,
            "Apuestas": n,
            "Bankroll final": round(bank, 2),
            "Ganancia %": round((bank / start_bankroll - 1) * 100, 1),
            "ROI sobre lo apostado %": round(((bank - start_bankroll) / staked * 100) if staked else 0.0, 1),
            "Máx. caída %": round(max_dd * 100, 1),
        })
    curves = pd.DataFrame(curves, index=df["Date"].values)
    hit = won[bet_mask].mean() if bet_mask.any() else np.nan
    summary = pd.DataFrame(summary)
    summary.attrs["hit_rate"] = hit
    summary.attrs["n_value_bets"] = int(bet_mask.sum())
    return curves, summary
