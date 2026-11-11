"""Comprovacions dels riscos principals: filtracions, horitzó i mètriques."""

import csv
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ca2_forecast.data import read_sales
from ca2_forecast.features import LAGS, feature_row, make_features
from ca2_forecast.modeling import baseline_predict, metrics, rolling_origins


def synthetic_calendar(n):
    start = date(2011, 1, 29)
    return [{"date": (start + timedelta(days=i)).isoformat(),
             "month": str((start + timedelta(days=i)).month), "snap_CA": "0",
             "event_name_1": "", "event_name_2": "", "event_type_1": "", "event_type_2": ""}
            for i in range(n)]


class ForecastingTests(unittest.TestCase):
    def test_entire_forecast_block_ignores_unobserved_sales(self):
        cutoff = 800
        y = np.arange(900, dtype=float)
        cal = synthetic_calendar(900)
        indices = range(cutoff, cutoff + 28)
        expected, names = make_features(y[:cutoff], cal, indices)
        corrupted = y.copy()
        corrupted[cutoff:] = 1e12
        actual, _ = make_features(corrupted, cal, indices)
        np.testing.assert_array_equal(expected, actual)
        self.assertGreater(len(names), 20)

    def test_lags_and_rolling_windows_are_aligned(self):
        y = np.arange(800, dtype=float)
        row = feature_row(y, synthetic_calendar(800), 500)
        for lag in LAGS:
            self.assertEqual(row[f"lag_{lag}"], 500 - lag)
        self.assertEqual(row["mean_7_shift28"], np.mean(y[466:473]))
        self.assertEqual(row["mean_28_shift28"], np.mean(y[445:473]))

    def test_features_reject_a_horizon_longer_than_available_history(self):
        with self.assertRaises(ValueError):
            feature_row(np.arange(800), synthetic_calendar(900), 828)

    def test_weekly_baselines_do_not_use_future_observations(self):
        y = np.arange(28)
        np.testing.assert_array_equal(baseline_predict("setmana_anterior", y), np.tile(np.arange(21, 28), 4))
        np.testing.assert_array_equal(baseline_predict("mitjana_4_setmanes", y), np.tile(np.arange(7) + 10.5, 4))

    def test_metrics_have_interpretable_units_and_scaling(self):
        actual = np.array([10, 20])
        predicted = np.array([17, 27])
        result = metrics(actual, predicted, np.arange(50))
        self.assertEqual(result["mae"], 7)
        self.assertEqual(result["rmse"], 7)
        self.assertEqual(result["mase7"], 1)
        self.assertEqual(result["bias"], 7)
        self.assertAlmostEqual(result["wape_pct"], 100 * 14 / 30)

    def test_temporal_blocks_cover_last_year_without_overlap(self):
        origins = rolling_origins(1913, 12)
        self.assertEqual(origins[0], 1577)
        self.assertEqual(origins[-1] + 28, 1913)
        self.assertTrue(all(b - a == 28 for a, b in zip(origins, origins[1:])))
        with self.assertRaises(ValueError):
            rolling_origins(1913, 0)

    def test_csv_loader_rejects_other_stores_and_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sales.csv"
            for store, value in [("CA_1", "2"), ("CA_2", "")]:
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["store_id", "item_id", "cat_id", "d_1"])
                    writer.writerow([store, "item", "FOODS", value])
                with self.assertRaises(ValueError):
                    read_sales(path)


if __name__ == "__main__":
    unittest.main()
