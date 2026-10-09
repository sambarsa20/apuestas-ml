"""Configuración central del proyecto.

Todo parámetro que el equipo quiera discutir o cambiar vive aquí,
para que el notebook y la app usen exactamente los mismos valores.
"""
from datetime import date

# --- Ligas (códigos de Football-Data.co.uk) ---------------------------------
LEAGUES = {
    "E0": "Premier League (Inglaterra)",
    "SP1": "LaLiga (España)",
    "I1": "Serie A (Italia)",
    "D1": "Bundesliga (Alemania)",
    "F1": "Ligue 1 (Francia)",
}

# --- Temporadas ---------------------------------------------------------------
FIRST_SEASON_START = 2011          # primera temporada descargada: 2011-12


def season_code(start_year: int) -> str:
    """2025 -> '2526' (formato de las URLs de Football-Data)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: date | None = None) -> int:
    """La temporada europea arranca en julio/agosto."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def all_season_codes(today: date | None = None) -> list[str]:
    last = current_season_start(today)
    return [season_code(y) for y in range(FIRST_SEASON_START, last + 1)]


# --- URLs ---------------------------------------------------------------------
HISTORY_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# --- Features -----------------------------------------------------------------
ELO_START = 1500        # rating de un equipo nuevo
ELO_K = 20              # cuánto se mueve el Elo tras cada partido
ELO_HOME_ADV = 60       # puntos Elo de ventaja por jugar en casa
ELO_SEASON_REGRESS = 0.2  # al iniciar temporada, el Elo vuelve 20% hacia 1500
ELO_PROMOTED_GAP = 80   # un equipo recién ascendido arranca en 1500 - 80
FORM_WINDOW = 5         # partidos para medir "forma reciente"

# --- Modelo -------------------------------------------------------------------
TARGET_MAP = {"H": 0, "D": 1, "A": 2}
CLASS_NAMES = ["Local", "Empate", "Visitante"]
RANDOM_STATE = 42

XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=3,
    n_estimators=600,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    eval_metric="mlogloss",
    early_stopping_rounds=50,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# --- Decisión de apuesta --------------------------------------------------------
MIN_EV = 0.05           # solo apostar si el valor esperado es >= 5%
KELLY_FRACTION = 0.25   # Kelly fraccional (1/4) por prudencia
MAX_STAKE = 0.05        # nunca más del 5% del bankroll en un partido
MAX_ODDS = 5.0          # no apostar a cuotas > 5 (batacazos): ahí el modelo es menos confiable
