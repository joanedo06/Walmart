"""Variables vectoritzades per a parelles (producte, dia), sense vendes futures."""

from __future__ import annotations

import numpy as np

from .features import LAGS, WARMUP, feature_row


class ProductFeatures:
    """Comparteix el calendari i conserva historials independents per producte."""

    def __init__(self, sales: np.ndarray, calendar, departments: list[str]):
        self.sales = np.asarray(sales, dtype=np.float32)
        self.calendar = calendar
        self.departments = np.asarray(departments)
        self.department_names = sorted(set(departments))
        zeros = np.zeros(len(calendar))
        sample = feature_row(zeros, calendar, WARMUP)
        self.calendar_names = [name for name in sample if not (
            name.startswith(("lag_", "mean_", "std_", "same_weekday_"))
        )]
        self.calendar_matrix = np.zeros((len(calendar), len(self.calendar_names)), dtype=np.float32)
        # El calendari no depèn de les vendes. Els primers dies són warmup.
        for t in range(WARMUP, len(calendar)):
            values = feature_row(zeros, calendar, t)
            self.calendar_matrix[t] = [values[name] for name in self.calendar_names]
        self.sums = np.pad(np.cumsum(self.sales, axis=1, dtype=np.float64), ((0, 0), (1, 0)))
        self.squares = np.pad(np.cumsum(self.sales.astype(np.float64) ** 2, axis=1), ((0, 0), (1, 0)))
        self.nonzero = np.pad(np.cumsum(self.sales > 0, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
        self.names = (
            self.calendar_names + [f"lag_{lag}" for lag in LAGS]
            + [name for w in (7, 28, 56) for name in (f"mean_{w}_shift28", f"std_{w}_shift28")]
            + ["same_weekday_mean_4_shift28", "zero_fraction_28_shift28", "zero_fraction_56_shift28"]
            + [f"dept_{dept}" for dept in self.department_names]
        )

    def matrix(self, products, days, observed_until: int) -> np.ndarray:
        products = np.asarray(products, dtype=int)
        days = np.asarray(days, dtype=int)
        if products.shape != days.shape or products.ndim != 1:
            raise ValueError("Cal una parella producte/dia per fila")
        if np.any(days < WARMUP) or np.any(days - 28 >= observed_until):
            raise ValueError("Les variables consultarien vendes no observades o sense warmup")
        if observed_until > self.sales.shape[1]:
            raise ValueError("L'origen supera l'històric disponible")
        cols = [self.calendar_matrix[days]]
        lag_values = np.column_stack([self.sales[products, days - lag] for lag in LAGS])
        cols.append(lag_values)
        stop = days - 28 + 1
        for window in (7, 28, 56):
            means = (self.sums[products, stop] - self.sums[products, stop - window]) / window
            second = (self.squares[products, stop] - self.squares[products, stop - window]) / window
            cols.append(np.column_stack((means, np.sqrt(np.maximum(second - means ** 2, 0)))))
        cols.append(lag_values[:, :4].mean(axis=1, keepdims=True))
        for window in (28, 56):
            fraction = 1 - (self.nonzero[products, stop] - self.nonzero[products, stop - window]) / window
            cols.append(fraction[:, None])
        cols.append(np.column_stack([self.departments[products] == dept for dept in self.department_names]))
        matrix = np.column_stack(cols).astype(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("Variables no finites")
        return matrix

    def forecast_matrix(self, cutoff: int) -> np.ndarray:
        n = self.sales.shape[0]
        return self.matrix(np.repeat(np.arange(n), 28), np.tile(np.arange(cutoff, cutoff + 28), n), cutoff)

    def training_sample(self, cutoff: int, samples_per_product: int, seed: int):
        if samples_per_product < 1 or cutoff <= WARMUP:
            raise ValueError("Mostra o històric insuficient")
        rng = np.random.default_rng(seed)
        products = np.repeat(np.arange(self.sales.shape[0]), samples_per_product)
        days = rng.integers(WARMUP, cutoff, size=len(products))
        x = self.matrix(products, days, cutoff)
        return x, self.sales[products, days]
