"""Ingeniería de variables SIN fuga de información (leakage).

Idea clave (Feynman): recorremos los partidos en orden cronológico como si
viviéramos cada fecha. Antes de cada partido "tomamos una foto" del estado de
los dos equipos (Elo, forma, goles) y DESPUÉS actualizamos ese estado con el
resultado. Así el modelo nunca ve información del futuro.

La misma clase `TeamState` sirve para:
  1. construir el dataset histórico (entrenamiento / backtest), y
  2. calcular las variables de un partido próximo en la app.
Por eso lo que explicamos en el notebook es exactamente lo que corre en vivo.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

# Nombre técnico -> explicación en español (se usa en SHAP y en la app)
FEATURE_LABELS = {
    "elo_diff": "Diferencia de Elo (local + ventaja de casa − visitante)",
    "elo_exp_home": "Prob. de victoria local según Elo",
    "home_elo": "Elo del local",
    "away_elo": "Elo del visitante",
    "home_pts5": "Puntos por partido del local (últimos 5)",
    "away_pts5": "Puntos por partido del visitante (últimos 5)",
    "home_gf5": "Goles a favor del local (prom. últimos 5)",
    "home_ga5": "Goles en contra del local (prom. últimos 5)",
    "away_gf5": "Goles a favor del visitante (prom. últimos 5)",
    "away_ga5": "Goles en contra del visitante (prom. últimos 5)",
    "home_pts5_home": "Puntos por partido del local jugando EN CASA (últimos 5)",
    "away_pts5_away": "Puntos por partido del visitante jugando FUERA (últimos 5)",
    "form_diff": "Diferencia de forma (puntos últimos 5)",
    "gd_diff": "Diferencia de gol reciente (local − visitante)",
}
for _div, _name in config.LEAGUES.items():
    FEATURE_LABELS[f"div_{_div}"] = f"Liga: {_name.split(' (')[0]}"

FEATURES = list(FEATURE_LABELS.keys())


def season_of(date: pd.Timestamp) -> str:
    start = date.year if date.month >= 7 else date.year - 1
    return config.season_code(start)


def _mean(d: deque) -> float:
    return float(np.mean(d)) if len(d) else np.nan


@dataclass
class TeamState:
    """Memoria de cada equipo: Elo + últimos partidos."""
    elo: dict = field(default_factory=lambda: defaultdict(lambda: float(config.ELO_START)))
    last_season: dict = field(default_factory=dict)
    pts: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=config.FORM_WINDOW)))
    gf: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=config.FORM_WINDOW)))
    ga: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=config.FORM_WINDOW)))
    pts_home: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=config.FORM_WINDOW)))
    pts_away: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=config.FORM_WINDOW)))
    team_league: dict = field(default_factory=dict)
    first_season: str | None = None

    # ---- Elo con regresión a la media al cambiar de temporada -------------------
    def _elo_now(self, team: str, season: str) -> float:
        prev = self.last_season.get(team)
        if prev is None:
            # Equipo nunca visto: en la 1ra temporada todos arrancan en 1500;
            # después, un equipo nuevo suele ser un ascendido -> arranca más bajo.
            if self.first_season is None or season == self.first_season:
                return float(config.ELO_START)
            return float(config.ELO_START - config.ELO_PROMOTED_GAP)
        e = self.elo[team]
        if prev is not None and prev != season:
            e = e + config.ELO_SEASON_REGRESS * (config.ELO_START - e)
        return e

    # ---- Foto ANTES del partido ------------------------------------------------
    def features_for(self, home: str, away: str, div: str, season: str) -> dict:
        he, ae = self._elo_now(home, season), self._elo_now(away, season)
        diff = he + config.ELO_HOME_ADV - ae
        f = {
            "elo_diff": diff,
            "elo_exp_home": 1.0 / (1.0 + 10 ** (-diff / 400.0)),
            "home_elo": he,
            "away_elo": ae,
            "home_pts5": _mean(self.pts[home]),
            "away_pts5": _mean(self.pts[away]),
            "home_gf5": _mean(self.gf[home]),
            "home_ga5": _mean(self.ga[home]),
            "away_gf5": _mean(self.gf[away]),
            "away_ga5": _mean(self.ga[away]),
            "home_pts5_home": _mean(self.pts_home[home]),
            "away_pts5_away": _mean(self.pts_away[away]),
        }
        f["form_diff"] = f["home_pts5"] - f["away_pts5"]
        f["gd_diff"] = (f["home_gf5"] - f["home_ga5"]) - (f["away_gf5"] - f["away_ga5"])
        for d in config.LEAGUES:
            f[f"div_{d}"] = 1.0 if div == d else 0.0
        return f

    # ---- Actualización DESPUÉS del partido -------------------------------------
    def update(self, home: str, away: str, div: str, season: str, hg: int, ag: int) -> None:
        if self.first_season is None:
            self.first_season = season
        he, ae = self._elo_now(home, season), self._elo_now(away, season)
        exp_home = 1.0 / (1.0 + 10 ** (-(he + config.ELO_HOME_ADV - ae) / 400.0))
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = config.ELO_K * (score - exp_home)
        self.elo[home], self.elo[away] = he + delta, ae - delta
        self.last_season[home] = self.last_season[away] = season

        hp, ap = (3, 0) if hg > ag else (1, 1) if hg == ag else (0, 3)
        self.pts[home].append(hp); self.pts[away].append(ap)
        self.gf[home].append(hg); self.ga[home].append(ag)
        self.gf[away].append(ag); self.ga[away].append(hg)
        self.pts_home[home].append(hp); self.pts_away[away].append(ap)
        self.team_league[home] = self.team_league[away] = div

    def teams_by_league(self) -> dict[str, list[str]]:
        out = defaultdict(list)
        for t, d in self.team_league.items():
            out[d].append(t)
        return {d: sorted(v) for d, v in out.items()}


def build_dataset(history: pd.DataFrame) -> tuple[pd.DataFrame, TeamState]:
    """Recorre el histórico en orden y devuelve (dataset con features, estado final)."""
    state = TeamState()
    rows = []
    for r in history.itertuples(index=False):
        feats = state.features_for(r.HomeTeam, r.AwayTeam, r.Div, r.Season)  # 1) foto
        rows.append(feats)
        state.update(r.HomeTeam, r.AwayTeam, r.Div, r.Season, int(r.FTHG), int(r.FTAG))  # 2) actualizar
    X = pd.DataFrame(rows, index=history.index)[FEATURES]
    data = pd.concat([history.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    data["y"] = data["FTR"].map(config.TARGET_MAP)
    return data, state


def features_for_fixtures(fixtures: pd.DataFrame, state: TeamState) -> pd.DataFrame:
    """Variables de partidos próximos usando el estado al día de hoy."""
    if fixtures.empty:
        return fixtures.assign(**{f: [] for f in FEATURES})
    rows = [state.features_for(r.HomeTeam, r.AwayTeam, r.Div, season_of(r.Date))
            for r in fixtures.itertuples(index=False)]
    X = pd.DataFrame(rows)[FEATURES]
    return pd.concat([fixtures.reset_index(drop=True), X], axis=1)
