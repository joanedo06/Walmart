"""Variables directes: cap venda dels 28 dies a predir entra a les variables.

Per a un objectiu al dia t, tots els retards són >=28. En un bloc que
comença a l'índex c, t <= c+27 i, per tant, t-28 <= c-1.
La mateixa propietat s'aplica a les finestres mòbils acabades a t-28.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from . import HORIZON

LAGS = (28, 35, 42, 49, 56, 364, 371)
WARMUP = max(LAGS)


def feature_row(y: np.ndarray, calendar: list[dict[str, str]], t: int) -> dict[str, float]:
    if t < WARMUP:
        raise ValueError(f"Calen almenys {WARMUP} dies previs")
    if t - HORIZON >= len(y):
        raise ValueError("Falten vendes observades per calcular les variables")
    row = calendar[t]
    day = date.fromisoformat(row["date"])
    events = {row.get("event_name_1", ""), row.get("event_name_2", "")} - {""}
    event_types = {row.get("event_type_1", ""), row.get("event_type_2", "")} - {""}
    features = {
        "trend_years": t / 365.25,
        "day_of_month": float(day.day),
        "month_start": float(day.day == 1),
        "month_end": float(t + 1 == len(calendar) or calendar[t + 1]["month"] != row["month"]),
        "snap_CA": float(row["snap_CA"]),
        "has_event": float(bool(events)),
        "christmas": float("Christmas" in events),
        "thanksgiving": float("Thanksgiving" in events),
        "easter": float("Easter" in events or "OrthodoxEaster" in events),
        "super_bowl": float("SuperBowl" in events),
    }
    for weekday in range(7):
        features[f"weekday_{weekday}"] = float(day.weekday() == weekday)
    for kind in ("Cultural", "National", "Religious", "Sporting"):
        features[f"event_{kind}"] = float(kind in event_types)
    for harmonic in (1, 2, 3):
        phase = 2 * np.pi * harmonic * (day.timetuple().tm_yday - 1) / 365.25
        features[f"year_sin_{harmonic}"] = float(np.sin(phase))
        features[f"year_cos_{harmonic}"] = float(np.cos(phase))
    for lag in LAGS:
        features[f"lag_{lag}"] = float(y[t - lag])
    # stop és exclusiu: l'última observació utilitzada és y[t-28].
    stop = t - HORIZON + 1
    for window in (7, 28, 56):
        history = y[stop - window:stop]
        features[f"mean_{window}_shift28"] = float(np.mean(history))
        features[f"std_{window}_shift28"] = float(np.std(history))
    features["same_weekday_mean_4_shift28"] = float(np.mean([y[t - lag] for lag in (28, 35, 42, 49)]))
    return features


def make_features(y: np.ndarray, calendar: list[dict[str, str]], indices) -> tuple[np.ndarray, list[str]]:
    rows = [feature_row(y, calendar, int(t)) for t in indices]
    if not rows:
        raise ValueError("No hi ha prou dades per construir variables")
    names = list(rows[0])
    matrix = np.asarray([[r[name] for name in names] for r in rows], dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Variables amb valors buits o infinits")
    return matrix, names
