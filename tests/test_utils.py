import numpy as np
import pytest

from EBP_sweep.utils import (
    find_nearest,
    format_ranges,
    get_oversampling_factor,
    outlier_clipping,
    robust_std,
    select_best_index,
)


def test_robust_std_matches_normal_std_for_clean_gaussian_data():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=0.0, scale=2.0, size=20000)
    assert robust_std(data) == pytest.approx(2.0, rel=0.05)


def test_robust_std_is_resistant_to_outliers():
    data = np.concatenate([np.zeros(1000), [1000.0]])  # one huge outlier
    assert robust_std(data) == pytest.approx(0.0, abs=1e-9)


def test_find_nearest_returns_closest_index_time_and_flux():
    time_array = np.array([0.0, 1.0, 2.0, 3.0])
    flux_array = np.array([1.0, 0.5, 0.2, 0.8])
    idx, time, flux = find_nearest(time_array, flux_array, value=0.25)
    assert idx == 2
    assert time == 2.0
    assert flux == 0.2


def test_format_ranges_collapses_consecutive_runs():
    assert format_ranges([1, 2, 3, 5, 7, 8, 9]) == "1-3, 5, 7-9"


def test_format_ranges_single_value():
    assert format_ranges([42]) == "42"


def test_select_best_index_skips_values_below_floor():
    stds = [1e-5, 1e-5, 0.5, 0.2]
    # index 3 has the smallest std above the floor
    assert select_best_index(stds, floor=1e-4) == 3


def test_select_best_index_falls_back_when_nothing_clears_floor():
    stds = [1e-6, 1e-7, 1e-8]
    assert select_best_index(stds, floor=1e-4, fallback_index=1) == 1


def test_get_oversampling_factor_is_positive_int():
    time_array = np.arange(0, 10, 1 / (24 * 60 * 2))  # 30s cadence, in days
    factor = get_oversampling_factor(time_array)
    assert isinstance(factor, int)
    assert factor >= 1


def test_outlier_clipping_removes_a_single_spike():
    rng = np.random.default_rng(1)
    x = np.arange(101, dtype=float)
    y = 1.0 + rng.normal(scale=0.001, size=101)
    y[50] = 5.0  # obvious spike
    x_clean, y_clean = outlier_clipping(x, y, clip=5, width=15, verbose=False)
    assert 50.0 not in x_clean
    assert len(x_clean) == 100
