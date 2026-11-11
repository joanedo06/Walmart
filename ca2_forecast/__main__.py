"""Executa la pràctica: python -m ca2_forecast."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np
from sklearn.inspection import permutation_importance

from . import HORIZON, STORE
from .data import load_dataset
from .features import LAGS, WARMUP, make_features
from .modeling import CANDIDATES, fit_predict, metrics, rolling_origins
from .reporting import create_figures, write_csv, write_report

ROOT = Path(__file__).resolve().parent.parent


def prediction_rows(calendar, cutoff, actual, predicted, name, fold):
    return [
        {"fold": fold, "model": name, "origin_date": calendar[cutoff - 1]["date"],
         "date": calendar[cutoff + i]["date"], "horizon": i + 1,
         "actual": int(actual[i]), "prediction": float(predicted[i])}
        for i in range(HORIZON)
    ]


def run(data_dir: Path, output_dir: Path, folds: int = 12, include_products: bool = True):
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Llegint CA_2 i comprovant l'històric producte a producte…", flush=True)
    data = load_dataset(data_dir)
    n = data.validation_length
    training_y = data.sales.totals[:n]
    origins = rolling_origins(n, folds)
    fold_metrics, cv_predictions = [], []
    for fold, cutoff in enumerate(origins, 1):
        print(f"Validació {fold}/{folds}: {data.calendar[cutoff]['date']} — {data.calendar[cutoff+27]['date']}", flush=True)
        # Només aquest prefix es passa a fit_predict: no pot consultar vendes futures.
        history = training_y[:cutoff]
        actual = training_y[cutoff:cutoff + HORIZON]
        for candidate in CANDIDATES:
            predicted, _, _ = fit_predict(candidate, history, data.calendar)
            fold_metrics.append({
                "fold": fold, "model": candidate.name, "train_end": data.calendar[cutoff - 1]["date"],
                "test_start": data.calendar[cutoff]["date"], "test_end": data.calendar[cutoff + 27]["date"],
                **metrics(actual, predicted, history),
            })
            cv_predictions.extend(prediction_rows(data.calendar, cutoff, actual, predicted, candidate.name, fold))
    summary = []
    for candidate in CANDIDATES:
        rows = [r for r in fold_metrics if r["model"] == candidate.name]
        record = {"model": candidate.name, "family": candidate.family}
        for metric in ("mae", "rmse", "wape_pct", "mase7", "bias"):
            record[f"{metric}_mean"] = float(np.mean([r[metric] for r in rows]))
        record["mae_std"] = float(np.std([r["mae"] for r in rows], ddof=1)) if folds > 1 else 0.0
        summary.append(record)
    summary.sort(key=lambda row: row["mae_mean"])
    winner = next(c for c in CANDIDATES if c.name == summary[0]["model"])
    # Es desa la selecció ABANS de llegir els errors de test.
    selection = {"criterion": "minimum_mean_validation_mae", "winner": asdict(winner), "candidates": [asdict(c) for c in CANDIDATES]}
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Model seleccionat per validació: {winner.name} (MAE {summary[0]['mae_mean']:.2f})", flush=True)

    family_winners = {}
    for row in summary:
        if row["family"] != "baseline":
            family_winners.setdefault(row["family"], row["model"])
    test_names = set(family_winners.values()) | {"setmana_anterior", "mitjana_4_setmanes"}
    test_metrics, test_predictions = [], []
    actual_test = data.sales.totals[n:n + HORIZON]
    for candidate in CANDIDATES:
        if candidate.name not in test_names:
            continue
        predicted, _, _ = fit_predict(candidate, training_y, data.calendar)
        test_metrics.append({"model": candidate.name, "selected": candidate.name == winner.name, **metrics(actual_test, predicted, training_y)})
        test_predictions.extend(prediction_rows(data.calendar, n, actual_test, predicted, candidate.name, "test"))

    importance_rows = []
    if winner.family != "baseline":
        cutoff = origins[-1]
        _, validation_model, names = fit_predict(winner, training_y[:cutoff], data.calendar)
        x, _ = make_features(training_y[:cutoff], data.calendar, range(cutoff, cutoff + HORIZON))
        importance = permutation_importance(validation_model, x, training_y[cutoff:cutoff + HORIZON], scoring="neg_mean_absolute_error", n_repeats=10, random_state=42)
        importance_rows = sorted([
            {"feature": name, "mae_increase": float(mean), "std": float(std)}
            for name, mean, std in zip(names, importance.importances_mean, importance.importances_std)
        ], key=lambda r: r["mae_increase"], reverse=True)

    all_y = data.sales.totals
    predicted, final_model, names = fit_predict(winner, all_y, data.calendar)
    cutoff = len(all_y)
    forecast = [
        {"store_id": STORE, "date": data.calendar[cutoff + i]["date"], "d": data.calendar[cutoff + i]["d"],
         "horizon": i + 1, "model": winner.name, "prediction": float(value)}
        for i, value in enumerate(predicted)
    ]
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump({"model": final_model, "candidate": asdict(winner), "feature_names": names,
                 "history": all_y, "calendar": data.calendar, "forecast_origin": data.calendar[cutoff - 1]["date"]}, model_dir / "ca2_model.joblib")
    daily_rows = []
    for t, value in enumerate(all_y):
        daily_rows.append({"date": data.calendar[t]["date"], "d": f"d_{t+1}", "store_id": STORE,
                           "units": int(value), "split": "development" if t < n else "test",
                           **{cat: int(values[t]) for cat, values in data.sales.categories.items()}})

    manifest = {
        "store": STORE, "horizon": HORIZON, "folds": folds, "seed": 42,
        "feature_lags": LAGS, "supervised_warmup": WARMUP,
        "selection": selection, "quality": data.quality,
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "matplotlib", "marimo", "scipy", "joblib")},
        "input_sha256": {name: sha256((data_dir / name).read_bytes()).hexdigest() for name in ("sales_train_validation.csv", "sales_train_evaluation.csv", "calendar.csv")},
    }
    for filename, rows in {
        "serie_diaria_ca2.csv": daily_rows, "validacio_blocs.csv": fold_metrics,
        "comparacio_models.csv": summary, "prediccions_validacio.csv": cv_predictions,
        "metriques_test.csv": test_metrics, "prediccions_test.csv": test_predictions,
        "previsio_28_dies.csv": forecast, "importancia_variables.csv": importance_rows,
    }.items():
        write_csv(output_dir / filename, rows)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    create_figures(data, summary, cv_predictions, test_predictions, forecast, winner.name, output_dir)
    write_report(data, summary, fold_metrics, test_metrics, forecast, winner, manifest, output_dir)
    selected_test = next(row for row in test_metrics if row["model"] == winner.name)
    print(f"Test: MAE={selected_test['mae']:.2f}; WAPE={selected_test['wape_pct']:.2f}%", flush=True)
    print(f"Resultats desats a {output_dir.resolve()}", flush=True)
    if include_products:
        from .products import run_products
        run_products(data, output_dir, folds=folds)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "m5-forecasting-accuracy")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--folds", type=int, default=12)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--products-only", action="store_true", help="Executa només l'experiment per productes")
    scope.add_argument("--total-only", action="store_true", help="Executa només l'experiment agregat original")
    args = parser.parse_args()
    if args.data_dir.resolve() == args.output_dir.resolve():
        parser.error("La carpeta de resultats ha de ser diferent de la de dades")
    if args.products_only:
        from .products import run_products
        run_products(load_dataset(args.data_dir), args.output_dir, folds=args.folds)
    else:
        run(args.data_dir, args.output_dir, args.folds, include_products=not args.total_only)


if __name__ == "__main__":
    main()
