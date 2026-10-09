"""Genera datos SIMULADOS con el formato exacto de Football-Data.co.uk.

Solo sirve para probar el código sin internet. Los resultados del modelo con
estos datos NO significan nada: el análisis real se hace con datos descargados.

Uso:  python tests/make_synthetic_data.py <carpeta_destino>
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

rng = np.random.default_rng(7)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
out.mkdir(parents=True, exist_ok=True)

TODAY = date(2026, 10, 9)
pool = {d: [f"{d}_Team{i:02d}" for i in range(26)] for d in config.LEAGUES}
strength = {t: rng.normal(0, 0.35) for d in pool for t in pool[d]}


def odds_from(p, margin=0.05):
    p = np.clip(p * np.exp(rng.normal(0, 0.06, 3)), 0.02, None)
    p = p / p.sum() * (1 + margin)
    return np.round(1 / p, 2)


def match(h, a):
    lam_h = np.exp(0.30 + strength[h] - strength[a])
    lam_a = np.exp(0.05 + strength[a] - strength[h])
    hg, ag = rng.poisson(lam_h), rng.poisson(lam_a)
    # probabilidad "verdadera" aprox. vía simulación Poisson
    sims_h, sims_a = rng.poisson(lam_h, 4000), rng.poisson(lam_a, 4000)
    p = np.array([(sims_h > sims_a).mean(), (sims_h == sims_a).mean(), (sims_h < sims_a).mean()])
    return hg, ag, odds_from(p)


fixtures = []
for y in range(config.FIRST_SEASON_START, config.current_season_start(TODAY) + 1):
    code = config.season_code(y)
    for div, teams in pool.items():
        for t in teams:                        # pequeña deriva de fuerza entre temporadas
            strength[t] += rng.normal(0, 0.08)
        league_teams = list(rng.choice(teams, 20, replace=False))
        start = date(y, 8, 15)
        rows = []
        pairs = [(h, a) for h in league_teams for a in league_teams if h != a]
        rng.shuffle(pairs)
        for i, (h, a) in enumerate(pairs):
            d = start + timedelta(days=int(i // 10) * 7 // 1 % 280)
            hg, ag, o = match(h, a)
            row = {"Div": div, "Date": d.strftime("%d/%m/%Y"), "Time": "15:00",
                   "HomeTeam": h, "AwayTeam": a, "FTHG": hg, "FTAG": ag,
                   "FTR": "H" if hg > ag else "D" if hg == ag else "A",
                   "B365H": o[0], "B365D": o[1], "B365A": o[2]}
            if d >= TODAY:
                if d <= TODAY + timedelta(days=10):
                    fixtures.append({k: row[k] for k in ["Div", "Date", "Time", "HomeTeam", "AwayTeam",
                                                         "B365H", "B365D", "B365A"]})
                continue
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / f"{code}_{div}.csv", index=False)

pd.DataFrame(fixtures).to_csv(out / "fixtures.csv", index=False, encoding="utf-8-sig")
print(f"Datos simulados en {out} | próximos partidos: {len(fixtures)}")
