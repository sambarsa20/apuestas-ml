"""Descarga y limpieza de datos de Football-Data.co.uk.

- Histórico: un CSV por liga y temporada (resultados + cuotas Bet365).
- Próximos partidos: fixtures.csv (se actualiza martes y viernes).

Los CSV descargados se guardan en `data/raw/` para no volver a bajarlos.
La temporada en curso se vuelve a descargar si el archivo tiene más de
`refresh_hours` horas, porque cada fin de semana suma resultados nuevos.
"""
from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path

import pandas as pd

from . import config

KEEP_COLS = ["Div", "Date", "Time", "HomeTeam", "AwayTeam",
             "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]
ODDS_COLS = ["B365H", "B365D", "B365A"]


def _download(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (USFQ academic project)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Lee un CSV de Football-Data tolerando BOM, acentos y filas rotas."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip")
            break
        except UnicodeDecodeError:
            continue
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df


def _parse_dates(s: pd.Series) -> pd.Series:
    # Football-Data usa dd/mm/yy (antiguo) o dd/mm/yyyy (reciente)
    return pd.to_datetime(s, dayfirst=True, format="mixed", errors="coerce")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
    df["Date"] = _parse_dates(df["Date"])
    df = df.dropna(subset=["Date"])
    for c in ["FTHG", "FTAG"] + ODDS_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    return df[KEEP_COLS]


def load_history(leagues: dict | None = None,
                 seasons: list[str] | None = None,
                 cache_dir: str | Path = "data/raw",
                 refresh_hours: float = 6.0,
                 verbose: bool = True) -> pd.DataFrame:
    """Devuelve todos los partidos jugados, ordenados por fecha."""
    leagues = leagues or config.LEAGUES
    seasons = seasons or config.all_season_codes()
    current = seasons[-1]
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    frames, missing = [], []
    for season in seasons:
        for div in leagues:
            path = cache / f"{season}_{div}.csv"
            stale = (season == current and path.exists()
                     and (time.time() - path.stat().st_mtime) > refresh_hours * 3600)
            if not path.exists() or stale:
                try:
                    path.write_bytes(_download(config.HISTORY_URL.format(season=season, div=div)))
                except Exception as e:  # noqa: BLE001
                    if not path.exists():
                        missing.append(f"{season}/{div} ({e.__class__.__name__})")
                        continue
            df = _clean(_read_csv_bytes(path.read_bytes()))
            df["Season"] = season
            frames.append(df)

    if not frames:
        raise RuntimeError("No se pudo cargar ningún archivo. ¿Hay conexión a internet?")
    if missing and verbose:
        print(f"Aviso: no se pudieron descargar {len(missing)} archivos: {missing[:5]}")

    hist = pd.concat(frames, ignore_index=True)
    hist = hist.dropna(subset=["FTHG", "FTAG", "FTR"])          # solo partidos jugados
    hist = hist[hist["FTR"].isin(["H", "D", "A"])]
    hist = hist.sort_values(["Date", "Div", "HomeTeam"]).reset_index(drop=True)
    if verbose:
        print(f"Partidos cargados: {len(hist):,} | {hist['Date'].min():%d/%m/%Y} → {hist['Date'].max():%d/%m/%Y}")
    return hist


def load_fixtures(leagues: dict | None = None,
                  cache_dir: str | Path = "data/raw",
                  refresh_hours: float = 3.0) -> pd.DataFrame:
    """Próximos partidos de las ligas elegidas, con cuotas Bet365."""
    leagues = leagues or config.LEAGUES
    path = Path(cache_dir) / "fixtures.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    stale = not path.exists() or (time.time() - path.stat().st_mtime) > refresh_hours * 3600
    if stale:
        try:
            path.write_bytes(_download(config.FIXTURES_URL))
        except Exception as e:  # noqa: BLE001
            if not path.exists():
                print(f"Aviso: no se pudo descargar fixtures.csv ({e.__class__.__name__})")
                return pd.DataFrame(columns=KEEP_COLS)
    fx = _clean(_read_csv_bytes(path.read_bytes()))
    fx = fx[fx["Div"].isin(leagues)].sort_values(["Date", "Div"]).reset_index(drop=True)
    return fx
