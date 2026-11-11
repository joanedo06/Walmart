"""Lectura dels CSV M5 i controls de qualitat, sense modificar els originals."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path

import numpy as np

from . import HORIZON, STORE


@dataclass
class SalesData:
    totals: np.ndarray
    categories: dict[str, np.ndarray]
    item_ids: set[str]
    prefix_hashes: dict[str, str]
    items: dict[str, np.ndarray]
    item_categories: dict[str, str]
    item_departments: dict[str, str]


@dataclass
class Dataset:
    calendar: list[dict[str, str]]
    sales: SalesData
    validation_length: int
    quality: dict


def read_sales(path: Path, prefix_length: int | None = None) -> SalesData:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        day_start = header.index("d_1")
        days = header[day_start:]
        if days != [f"d_{i}" for i in range(1, len(days) + 1)]:
            raise ValueError(f"Dies absents o desordenats a {path.name}")
        totals = np.zeros(len(days), dtype=np.int64)
        categories: dict[str, np.ndarray] = {}
        ids: set[str] = set()
        hashes = {}
        items, item_categories, item_departments = {}, {}, {}
        store_col, item_col, cat_col = map(header.index, ("store_id", "item_id", "cat_id"))
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Fila incompleta a {path.name}:{line}")
            if row[store_col] != STORE:
                raise ValueError(f"{path.name}:{line} conté una botiga diferent de {STORE}")
            item = row[item_col]
            if item in ids:
                raise ValueError(f"Producte duplicat: {item}")
            ids.add(item)
            values = np.array(row[day_start:], dtype=np.int64)
            if (values < 0).any():
                raise ValueError(f"Vendes negatives: {item}")
            totals += values
            category = row[cat_col]
            items[item] = values
            item_categories[item] = category
            item_departments[item] = row[header.index("dept_id")] if "dept_id" in header else category
            if category not in categories:
                categories[category] = np.zeros(len(days), dtype=np.int64)
            categories[category] += values
            hashes[item] = sha256(values[:prefix_length].astype("<i8").tobytes()).hexdigest()
    if not ids:
        raise ValueError(f"No hi ha productes a {path}")
    return SalesData(totals, categories, ids, hashes, items, item_categories, item_departments)


def load_dataset(data_dir: Path) -> Dataset:
    with (data_dir / "calendar.csv").open(newline="", encoding="utf-8") as stream:
        calendar = list(csv.DictReader(stream))
    for i, row in enumerate(calendar):
        if row["d"] != f"d_{i + 1}":
            raise ValueError("El calendari no té els identificadors diaris consecutius")
        if i and date.fromisoformat(row["date"]) != date.fromisoformat(calendar[i - 1]["date"]) + timedelta(days=1):
            raise ValueError("El calendari no és diari i consecutiu")
    validation = read_sales(data_dir / "sales_train_validation.csv")
    n = len(validation.totals)
    evaluation = read_sales(data_dir / "sales_train_evaluation.csv", prefix_length=n)
    if validation.prefix_hashes != evaluation.prefix_hashes:
        raise ValueError("Validation i evaluation no coincideixen producte a producte en el període compartit")
    if len(evaluation.totals) != n + HORIZON:
        raise ValueError("S'esperaven exactament 28 dies addicionals a evaluation")
    if len(calendar) < len(evaluation.totals) + HORIZON:
        raise ValueError("Falta calendari per generar els pròxims 28 dies")
    quality = {
        "store": STORE,
        "products": len(evaluation.item_ids),
        "validation_days": n,
        "observed_days": len(evaluation.totals),
        "first_date": calendar[0]["date"],
        "validation_end": calendar[n - 1]["date"],
        "observed_end": calendar[len(evaluation.totals) - 1]["date"],
        "zero_total_days": int((evaluation.totals == 0).sum()),
        "matching_product_histories": True,
        "consecutive_days": True,
        "missing_sales_cells": 0,
    }
    return Dataset(calendar, evaluation, n, quality)
