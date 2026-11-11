"""Independència entre productes i ús exclusiu d'informació passada."""

import unittest

import numpy as np

from ca2_forecast.features import feature_row
from ca2_forecast.product_features import ProductFeatures
from ca2_forecast.product_models import per_product_metrics, product_baselines, within_product_importance
from test_forecasting import synthetic_calendar


class ProductTests(unittest.TestCase):
    def test_vectorized_features_match_single_series(self):
        sales = np.stack([np.arange(900), 2 * np.arange(900)]).astype(float)
        calendar = synthetic_calendar(930)
        builder = ProductFeatures(sales, calendar, ["FOODS_1", "HOUSEHOLD_1"])
        matrix = builder.matrix([0, 1], [800, 827], 800)
        for i, (product, day) in enumerate([(0, 800), (1, 827)]):
            expected = feature_row(sales[product], calendar, day)
            for name, value in expected.items():
                self.assertAlmostEqual(float(matrix[i, builder.names.index(name)]), value, places=3)
        self.assertEqual(matrix[0, builder.names.index("dept_FOODS_1")], 1)
        self.assertEqual(matrix[1, builder.names.index("dept_FOODS_1")], 0)

    def test_future_changes_cannot_affect_product_features(self):
        rng = np.random.default_rng(42)
        sales = rng.integers(0, 15, size=(2, 900))
        calendar = synthetic_calendar(930)
        first = ProductFeatures(sales, calendar, ["A", "B"]).forecast_matrix(800)
        corrupted = sales.copy()
        corrupted[:, 800:] = 100000
        second = ProductFeatures(corrupted, calendar, ["A", "B"]).forecast_matrix(800)
        np.testing.assert_array_equal(first, second)

    def test_training_sample_never_uses_future_targets(self):
        sales = np.zeros((2, 900))
        sales[:, 800:] = 999999
        builder = ProductFeatures(sales, synthetic_calendar(930), ["A", "B"])
        _, labels = builder.training_sample(800, 32, 42)
        self.assertEqual(len(labels), 64)
        self.assertTrue(np.all(labels == 0))
        with self.assertRaises(ValueError):
            builder.matrix([0], [828], 800)

    def test_product_baselines_preserve_weekdays_and_product_identity(self):
        sales = np.stack([np.arange(28), np.arange(28) + 100])
        prediction = product_baselines(sales)
        np.testing.assert_array_equal(prediction["setmana_anterior"][1], np.tile(np.arange(21, 28) + 100, 4))
        np.testing.assert_array_equal(prediction["mitjana_4_setmanes"][0], np.tile(np.arange(7) + 10.5, 4))

    def test_zero_volume_metrics_remain_undefined(self):
        values = per_product_metrics(np.zeros((2, 28)), np.ones((2, 28)), np.zeros((2, 800)))
        np.testing.assert_array_equal(values["mae"], [1, 1])
        self.assertTrue(np.isnan(values["wape_pct"]).all())
        self.assertTrue(np.isnan(values["mase7"]).all())

    def test_local_importance_does_not_shuffle_between_products(self):
        class SignalModel:
            def predict(self, x):
                np.testing.assert_array_equal(x[:28, 1], 0)
                np.testing.assert_array_equal(x[28:, 1], 100)
                return x[:, 0]
        actual = np.stack([np.arange(28), 100 + np.arange(28)])
        x = np.column_stack([actual.ravel(), np.repeat([0, 100], 28)])
        means, stds = within_product_importance(SignalModel(), x, actual, repeats=3)
        self.assertTrue(np.all(means[:, 0] > 0))
        np.testing.assert_array_equal(means[:, 1], 0)
        self.assertTrue(np.isfinite(stds).all())


if __name__ == "__main__":
    unittest.main()
