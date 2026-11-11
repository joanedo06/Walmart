"""Models i avaluació de blocs complets de 28 dies sense actualitzar retards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import HORIZON
from .features import WARMUP, make_features


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    parameter: float = 0


CANDIDATES = [
    Candidate("setmana_anterior", "baseline"),
    Candidate("mitjana_4_setmanes", "baseline"),
    *[Candidate(f"ridge_{a}", "ridge", a) for a in (1, 10, 100)],
    *[Candidate(f"random_forest_leaf{leaf}", "random_forest", leaf) for leaf in (3, 10)],
    *[Candidate(f"boosting_leaves{leaves}", "boosting", leaves) for leaves in (7, 15)],
]


def build_model(candidate: Candidate):
    if candidate.family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=candidate.parameter))
    if candidate.family == "random_forest":
        return RandomForestRegressor(
            n_estimators=150, min_samples_leaf=int(candidate.parameter),
            max_features=0.8, random_state=42, n_jobs=-1,
        )
    if candidate.family == "boosting":
        return HistGradientBoostingRegressor(
            max_iter=250, max_leaf_nodes=int(candidate.parameter),
            min_samples_leaf=10, learning_rate=0.05,
            l2_regularization=10, early_stopping=False, random_state=42,
        )
    raise ValueError(f"El model {candidate.name} és una referència sense entrenament")


def baseline_predict(name: str, history: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    if len(history) < HORIZON:
        raise ValueError("La referència necessita almenys 28 dies d'històric")
    if name == "setmana_anterior":
        return np.tile(history[-7:], int(np.ceil(horizon / 7)))[:horizon].astype(float)
    if name == "mitjana_4_setmanes":
        return np.tile(history[-28:].reshape(4, 7).mean(axis=0), int(np.ceil(horizon / 7)))[:horizon]
    raise ValueError(name)


def fit_predict(candidate: Candidate, history: np.ndarray, calendar: list[dict[str, str]]):
    cutoff = len(history)
    if candidate.family == "baseline":
        return baseline_predict(candidate.name, history), None, []
    train_x, names = make_features(history, calendar, range(WARMUP, cutoff))
    future_x, _ = make_features(history, calendar, range(cutoff, cutoff + HORIZON))
    model = build_model(candidate)
    model.fit(train_x, history[WARMUP:])
    return np.maximum(model.predict(future_x), 0), model, names


def metrics(actual: np.ndarray, predicted: np.ndarray, history: np.ndarray) -> dict[str, float]:
    error = np.asarray(actual) - np.asarray(predicted)
    scale = np.mean(np.abs(history[7:] - history[:-7]))
    total = np.sum(np.abs(actual))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "wape_pct": float(100 * np.sum(np.abs(error)) / total) if total else float("nan"),
        "mase7": float(np.mean(np.abs(error)) / scale) if scale else float("nan"),
        "bias": float(np.mean(predicted - actual)),
    }


def rolling_origins(training_length: int, folds: int = 12) -> list[int]:
    if folds < 1:
        raise ValueError("Cal almenys un bloc de validació")
    origins = [training_length - HORIZON * k for k in range(folds, 0, -1)]
    if origins[0] <= WARMUP + 365:
        raise ValueError("Massa blocs: calen almenys dos anys per al primer entrenament")
    return origins
