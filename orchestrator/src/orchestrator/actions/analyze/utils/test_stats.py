import math

import pytest

from orchestrator.actions.analyze.utils.stats import compute_group_stats


def test_compute_group_stats_known_fixture() -> None:
    result = compute_group_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result.n == 5
    assert result.median == 3.0
    assert result.mean == 3.0
    assert math.isclose(result.std, 1.5811388300841898, rel_tol=1e-9)
    assert result.min == 1.0
    assert result.max == 5.0
    assert result.p25 == 2.0
    assert result.p75 == 4.0
    assert result.iqr == 2.0


def test_compute_group_stats_n1_has_nan_std_and_zero_iqr() -> None:
    result = compute_group_stats([7.5])
    assert result.n == 1
    assert result.median == 7.5
    assert result.mean == 7.5
    assert result.min == 7.5
    assert result.max == 7.5
    assert math.isnan(result.std)
    assert result.iqr == 0.0


def test_compute_group_stats_n2_iqr_finite() -> None:
    result = compute_group_stats([10.0, 20.0])
    assert result.n == 2
    assert result.median == 15.0
    assert math.isclose(result.std, 7.0710678118654755, rel_tol=1e-9)
    assert math.isfinite(result.iqr)


def test_compute_group_stats_empty_raises() -> None:
    with pytest.raises(ValueError):
        compute_group_stats([])
