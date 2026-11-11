"""Models globals amb prediccions, errors i importàncies locals per producte."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASELINES = ("setmana_anterior", "mitjana_4_setmanes")
LEARNED_MODELS = ("ridge_global", "random_forest_global", "boosting_global")
PRODUCT_MODELS = BASELINES + LEARNED_MODELS


def product_model(name: str):
    if name == "ridge_global":
        return make_pipeline(StandardScaler(), Ridge(alpha=100))
    if name == "random_forest_global":
        return RandomForestRegressor(n_estimators=64, max_depth=14, min_samples_leaf=15,
                                     max_features=0.8, n_jobs=4, random_state=42)
    if name == "boosting_global":
        return HistGradientBoostingRegressor(loss="poisson", learning_rate=0.07, max_iter=160,
                                            max_leaf_nodes=15, min_samples_leaf=30,
                                            l2_regularization=10, early_stopping=False, random_state=42)
    raise ValueError(name)


def product_baselines(history: np.ndarray) -> dict[str, np.ndarray]:
    if history.shape[1] < 28:
        raise ValueError("Calen 28 dies observats")
    n = history.shape[0]
    return {
        "setmana_anterior": np.tile(history[:, -7:], (1, 4)).astype(float),
        "mitjana_4_setmanes": np.tile(history[:, -28:].reshape(n, 4, 7).mean(axis=1), (1, 4)),
    }


def per_product_metrics(actual: np.ndarray, predicted: np.ndarray, history: np.ndarray):
    error = predicted - actual
    mae = np.abs(error).mean(axis=1)
    volume = np.abs(actual).sum(axis=1)
    scale = np.abs(history[:, 7:] - history[:, :-7]).mean(axis=1)
    return {
        "mae": mae,
        "rmse": np.sqrt((error ** 2).mean(axis=1)),
        "wape_pct": np.divide(100 * np.abs(error).sum(axis=1), volume, out=np.full_like(mae, np.nan), where=volume != 0),
        "mase7": np.divide(mae, scale, out=np.full_like(mae, np.nan), where=scale != 0),
        "bias": error.mean(axis=1),
    }


def within_product_importance(model, x: np.ndarray, actual: np.ndarray, repeats: int = 3, seed: int = 42, progress=None):
    """Permuta cada columna ENTRE DIES del mateix producte, mai entre productes.

    El resultat és l'augment de MAE en unitats/dia de cada producte.
    Variables constants dins dels 28 dies tenen importància local zero; això
    no mesura la seva importància global per distingir productes.
    """
    n, horizon = actual.shape
    if x.shape[0] != n * horizon or repeats < 1:
        raise ValueError("Dimensions o repeticions invàlides")
    baseline = np.abs(np.maximum(model.predict(x), 0).reshape(n, horizon) - actual).mean(axis=1)
    means = np.zeros((n, x.shape[1]))
    deviations = np.zeros_like(means)
    rng = np.random.default_rng(seed)
    changed = x.copy()
    product_index = np.arange(n)[:, None]
    for feature in range(x.shape[1]):
        values = x[:, feature].reshape(n, horizon)
        if np.all(values == values[:, :1]):
            continue
        effects = []
        for _ in range(repeats):
            permutation = np.argsort(rng.random((n, horizon)), axis=1)
            changed[:, feature] = values[product_index, permutation].ravel()
            prediction = np.maximum(model.predict(changed), 0).reshape(n, horizon)
            effects.append(np.abs(prediction - actual).mean(axis=1) - baseline)
        changed[:, feature] = x[:, feature]
        means[:, feature] = np.mean(effects, axis=0)
        deviations[:, feature] = np.std(effects, axis=0)
        if progress and (feature + 1) % 10 == 0:
            progress(feature + 1, x.shape[1])
    return means, deviations
