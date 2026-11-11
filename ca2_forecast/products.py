"""Entrenament compartit i comparació separada dels 3.049 productes."""

from __future__ import annotations

import csv
import gzip
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from . import HORIZON
from .modeling import rolling_origins
from .product_features import ProductFeatures
from .product_models import LEARNED_MODELS, PRODUCT_MODELS, per_product_metrics, product_baselines, product_model, within_product_importance
from .reporting import write_csv


def finite(value):
    return float(value) if np.isfinite(value) else None


def archive_product_details(output_dir: Path) -> Path:
    """Comprimeix els resultats indexats per distribuir el quadern amb Git."""
    archive = output_dir / "detalls.sqlite.gz"
    with (output_dir / "detalls.sqlite").open("rb") as source, archive.open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
            shutil.copyfileobj(source, compressed)
    return archive


def restore_product_details(output_dir: Path) -> Path:
    """Descomprimeix la còpia distribuïda només quan falta la base local."""
    database = output_dir / "detalls.sqlite"
    archive = output_dir / "detalls.sqlite.gz"
    if not database.exists() and archive.exists():
        temporary = output_dir / "detalls.sqlite.restoring"
        try:
            with gzip.open(archive, "rb") as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(database)
        finally:
            temporary.unlink(missing_ok=True)
    return database


def read_product_detail(output_dir: Path, item_id: str) -> dict:
    """Carrega només el producte seleccionat al quadern, sense llegir-los tots."""
    database = restore_product_details(output_dir)
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT detail FROM products WHERE item_id = ?", (item_id,)).fetchone()
    if row is None:
        raise KeyError(item_id)
    return json.loads(row[0])


def metric_records(item_ids, model, values, **extra):
    return [{"item_id": item, "model": model, **extra,
             **{key: finite(array[i]) for key, array in values.items()}}
            for i, item in enumerate(item_ids)]


def run_products(dataset, output_dir: Path, folds: int = 12, samples_per_product: int = 32, importance_repeats: int = 3):
    with threadpool_limits(limits=4):
        return _run_products(dataset, output_dir, folds, samples_per_product, importance_repeats)


def _run_products(dataset, output_dir, folds, samples_per_product, importance_repeats):
    out = output_dir / "productes"
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    items = sorted(dataset.sales.items)
    sales = np.stack([dataset.sales.items[item] for item in items]).astype(np.float32)
    departments = [dataset.sales.item_departments[item] for item in items]
    builder = ProductFeatures(sales, dataset.calendar, departments)
    n_items, observed = sales.shape
    n = dataset.validation_length
    origins = rolling_origins(n, folds)
    cv_mae = {name: np.zeros(n_items) for name in PRODUCT_MODELS}
    cv_absolute = {name: 0.0 for name in PRODUCT_MODELS}
    cv_volume = 0.0
    last_models, last_x, last_actual = {}, None, None
    print(f"Productes: {n_items}; {folds} blocs; {samples_per_product*n_items:,} files d'entrenament mostrejades per bloc.", flush=True)

    with (out / "validacio_blocs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for fold, cutoff in enumerate(origins, 1):
            print(f"Productes — validació {fold}/{folds}: {dataset.calendar[cutoff]['date']}", flush=True)
            x_train, y_train = builder.training_sample(cutoff, samples_per_product, 42 + fold)
            x_future = builder.forecast_matrix(cutoff)
            history, actual = sales[:, :cutoff], sales[:, cutoff:cutoff + HORIZON]
            predictions = product_baselines(history)
            for name in LEARNED_MODELS:
                model = product_model(name)
                model.fit(x_train, y_train)
                predictions[name] = np.maximum(model.predict(x_future), 0).reshape(n_items, HORIZON)
                if fold == folds:
                    last_models[name] = model
            cv_volume += float(actual.sum())
            for name in PRODUCT_MODELS:
                values = per_product_metrics(actual, predictions[name], history)
                cv_mae[name] += values["mae"] / folds
                cv_absolute[name] += float(np.abs(predictions[name] - actual).sum())
                rows = metric_records(items, name, values, fold=fold,
                                      train_end=dataset.calendar[cutoff-1]["date"],
                                      test_start=dataset.calendar[cutoff]["date"], test_end=dataset.calendar[cutoff+27]["date"])
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
            if fold == folds:
                last_x, last_actual = x_future, actual

    score_matrix = np.column_stack([cv_mae[name] for name in PRODUCT_MODELS])
    selected_indices = np.argmin(score_matrix, axis=1)
    selected = [PRODUCT_MODELS[i] for i in selected_indices]
    learned_indices = np.argmin(np.column_stack([cv_mae[name] for name in LEARNED_MODELS]), axis=1)
    selection = {"criterion": "minimum_mean_product_mae_in_validation", "models": list(PRODUCT_MODELS),
                 "per_product": dict(zip(items, selected))}
    # Congelació de la selecció abans de calcular mètriques del període final.
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    write_csv(out / "comparacio_models.csv", [
        {"model": name, "cv_mae_mean": float(cv_mae[name].mean()),
         "cv_mae_median": float(np.median(cv_mae[name])), "cv_wape_pct": 100 * cv_absolute[name] / cv_volume,
         "selected_products": selected.count(name)} for name in PRODUCT_MODELS
    ])

    importance, importance_std, importance_base = {}, {}, {}
    print("Productes — importàncies locals en l'últim bloc de validació (no en test).", flush=True)
    for name in LEARNED_MODELS:
        print(f"Importàncies: {name}", flush=True)
        def progress(done, total, name=name):
            print(f"  {name}: {done}/{total} variables", flush=True)
        importance[name], importance_std[name] = within_product_importance(
            last_models[name], last_x, last_actual, repeats=importance_repeats, progress=progress,
        )
        importance_base[name] = np.abs(np.maximum(last_models[name].predict(last_x), 0).reshape(n_items, HORIZON) - last_actual).mean(axis=1)
    with (out / "importancia_variables.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item_id", "model", "feature", "mae_increase", "std", "validation_mae", "mae_increase_pct"])
        writer.writeheader()
        for name in LEARNED_MODELS:
            for i, item in enumerate(items):
                writer.writerows({"item_id": item, "model": name, "feature": feature,
                                  "mae_increase": float(importance[name][i, j]), "std": float(importance_std[name][i, j]),
                                  "validation_mae": float(importance_base[name][i]),
                                  "mae_increase_pct": float(100*importance[name][i,j]/importance_base[name][i]) if importance_base[name][i] else None}
                                 for j, feature in enumerate(builder.names))
    del last_models, last_x

    print("Productes — avaluant el període final amb la selecció congelada.", flush=True)
    x_train, y_train = builder.training_sample(n, samples_per_product, 42)
    x_test = builder.forecast_matrix(n)
    actual_test = sales[:, n:n + HORIZON]
    test_predictions = product_baselines(sales[:, :n])
    for name in LEARNED_MODELS:
        model = product_model(name)
        model.fit(x_train, y_train)
        test_predictions[name] = np.maximum(model.predict(x_test), 0).reshape(n_items, HORIZON)
    test_values = {name: per_product_metrics(actual_test, prediction, sales[:, :n]) for name, prediction in test_predictions.items()}
    test_rows = [row for name in PRODUCT_MODELS for row in metric_records(items, name, test_values[name])]
    write_csv(out / "metriques_test.csv", test_rows)
    with (out / "prediccions_test.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item_id", "date", "horizon", "model", "actual", "prediction"])
        writer.writeheader()
        for name in PRODUCT_MODELS:
            for i, item in enumerate(items):
                writer.writerows({"item_id": item, "date": dataset.calendar[n+h]["date"], "horizon": h+1,
                                  "model": name, "actual": int(actual_test[i, h]), "prediction": float(test_predictions[name][i, h])}
                                 for h in range(HORIZON))
    selected_test = np.stack([test_predictions[name][i] for i, name in enumerate(selected)])
    selected_values = per_product_metrics(actual_test, selected_test, sales[:, :n])
    levels = []
    for name, prediction in {**test_predictions, "seleccio_per_producte": selected_test}.items():
        error = prediction - actual_test
        store_error = error.sum(axis=0)
        levels.append({"model": name, "product_mae_mean": float(np.abs(error).mean()),
                       "product_wape_pct": float(100 * np.abs(error).sum() / actual_test.sum()),
                       "store_mae": float(np.abs(store_error).mean()),
                       "store_wape_pct": float(100 * np.abs(store_error).sum() / actual_test.sum())})
    aggregate_metrics = output_dir / "metriques_test.csv"
    if aggregate_metrics.exists():
        with aggregate_metrics.open(newline="") as stream:
            for row in csv.DictReader(stream):
                levels.append({"model": f"total_directe:{row['model']}", "product_mae_mean": None,
                               "product_wape_pct": None, "store_mae": float(row["mae"]), "store_wape_pct": float(row["wape_pct"])})
    write_csv(out / "comparacio_nivells.csv", levels)

    print("Productes — reentrenant i preveient els 28 dies següents per a cada producte.", flush=True)
    x_train, y_train = builder.training_sample(observed, samples_per_product, 42)
    x_forecast = builder.forecast_matrix(observed)
    forecast_predictions = product_baselines(sales)
    model_dir = output_dir / "models" / "productes"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in LEARNED_MODELS:
        model = product_model(name)
        model.fit(x_train, y_train)
        forecast_predictions[name] = np.maximum(model.predict(x_forecast), 0).reshape(n_items, HORIZON)
        joblib.dump({"model": model, "feature_names": builder.names, "item_ids": items,
                     "observed_until": observed, "samples_per_product": samples_per_product}, model_dir / f"{name}.joblib")
    selected_forecast = np.stack([forecast_predictions[name][i] for i, name in enumerate(selected)])
    with (out / "previsio_28_dies.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item_id", "date", "horizon", "model", "prediction"])
        writer.writeheader()
        for i, item in enumerate(items):
            writer.writerows({"item_id": item, "date": dataset.calendar[observed+h]["date"], "horizon": h+1,
                              "model": selected[i], "prediction": float(selected_forecast[i, h])} for h in range(HORIZON))
    write_csv(out / "previsio_total_suma_productes.csv", [
        {"date": dataset.calendar[observed+h]["date"], "prediction": float(selected_forecast[:, h].sum())} for h in range(HORIZON)
    ])

    summary = []
    database = out / "detalls.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS products (item_id TEXT PRIMARY KEY, detail TEXT NOT NULL)")
        # Una execució completa substitueix tots els detalls, dins d'una transacció.
        connection.execute("DELETE FROM products")
        for i, item in enumerate(items):
            learned = LEARNED_MODELS[learned_indices[i]]
            top = int(np.argmax(importance[learned][i]))
            item_row = {"item_id": item, "category": dataset.sales.item_categories[item], "department": departments[i],
                        "mean_units_development": float(sales[i, :n].mean()),
                        "zero_fraction_development": float((sales[i, :n] == 0).mean()),
                        "selected_model": selected[i], "best_learned_model": learned,
                        "cv_mae_selected": float(score_matrix[i, selected_indices[i]]),
                        "test_mae_selected": float(selected_values["mae"][i]),
                        "test_wape_pct_selected": finite(selected_values["wape_pct"][i]),
                        "top_feature_learned": builder.names[top] if importance[learned][i, top] > 0 else "cap augment positiu",
                        **{f"cv_mae_{name}": float(cv_mae[name][i]) for name in PRODUCT_MODELS}}
            summary.append(item_row)
            detail = {
                "summary": item_row,
                "test_dates": [dataset.calendar[n+h]["date"] for h in range(HORIZON)],
                "test_actual": actual_test[i].tolist(),
                "test_predictions": {name: prediction[i].tolist() for name, prediction in test_predictions.items()},
                "test_metrics": [{"model": name, **{metric: finite(values[i]) for metric, values in test_values[name].items()}} for name in PRODUCT_MODELS],
                "cv_metrics": [{"model": name, "mae": float(cv_mae[name][i])} for name in PRODUCT_MODELS],
                "importance": {name: [
                    {"feature": feature, "mae_increase": float(importance[name][i,j]), "std": float(importance_std[name][i,j]),
                     "mae_increase_pct": float(100*importance[name][i,j]/importance_base[name][i]) if importance_base[name][i] else None}
                    for j, feature in enumerate(builder.names)] for name in LEARNED_MODELS},
                "forecast_dates": [dataset.calendar[observed+h]["date"] for h in range(HORIZON)],
                "forecast_predictions": {name: prediction[i].tolist() for name, prediction in forecast_predictions.items()},
            }
            connection.execute("INSERT INTO products VALUES (?, ?)", (item, json.dumps(detail, allow_nan=False)))
    archive_product_details(out)
    write_csv(out / "resum_productes.csv", summary)
    manifest = {
        "products": n_items, "horizon": HORIZON, "folds": folds, "seed": 42,
        "samples_per_product_per_fold": samples_per_product, "training_sample_size": samples_per_product * n_items,
        "training_sampling": "uniform_with_replacement_per_product_over_development_days_after_warmup",
        "feature_names": builder.names, "shared_models": True,
        "model_parameters": {name: product_model(name).get_params(deep=False) if name != "ridge_global" else {"alpha":100, "standardized":True} for name in LEARNED_MODELS},
        "importance_repeats": importance_repeats, "importance_scope": "last_validation_block_within_product_permutation",
        "importance_dates": [dataset.calendar[origins[-1]]["date"], dataset.calendar[origins[-1]+27]["date"]],
        "test_dates": [dataset.calendar[n]["date"], dataset.calendar[n+27]["date"]],
        "forecast_dates": [dataset.calendar[observed]["date"], dataset.calendar[observed+27]["date"]],
        "quality": dataset.quality,
        "selection_counts": {name: selected.count(name) for name in PRODUCT_MODELS},
        "elapsed_seconds": time.monotonic() - started,
        "input_sales_sha256": sha256(sales.tobytes()).hexdigest(),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "scipy", "joblib", "threadpoolctl")},
        "source_sha256": {name: sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ("data.py", "features.py", "modeling.py", "product_features.py", "product_models.py", "products.py")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    from .product_reporting import write_product_report
    write_product_report(out, summary, levels, manifest)
    print(f"Productes completats en {time.monotonic()-started:.1f}s: {out.resolve()}", flush=True)
    return manifest
