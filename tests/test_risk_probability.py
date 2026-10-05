import pytest

from responsible_gaming_ml.risk_probability import (
    calculate_risk_probability,
    calculate_risk_score,
    normalize_feature,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0.0), (5.0, 0.5), (10.0, 1.0)],
)
def test_normalize_feature(value: float, expected: float) -> None:
    assert normalize_feature(value, 0.0, 10.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("feature_name", "maximum", "expected_weight"),
    [
        ("losses_7d", 1000.0, 0.35),
        ("session_hours", 40.0, 0.30),
        ("deposit_frequency", 20.0, 0.20),
        ("night_sessions", 10.0, 0.10),
        ("deposits_7d", 2000.0, 0.05),
    ],
)
def test_calculate_risk_score_applies_feature_weights(
    feature_name: str, maximum: float, expected_weight: float
) -> None:
    feature_values = {
        "deposits_7d": 0.0,
        "losses_7d": 0.0,
        "night_sessions": 0.0,
        "session_hours": 0.0,
        "deposit_frequency": 0.0,
    }
    feature_values[feature_name] = maximum

    assert calculate_risk_score(**feature_values) == pytest.approx(expected_weight)


@pytest.mark.parametrize(
    ("feature_name", "out_of_bounds_value"),
    [
        ("deposits_7d", 2001.0),
        ("losses_7d", 1001.0),
        ("night_sessions", 11.0),
        ("session_hours", 41.0),
        ("deposit_frequency", 21.0),
    ],
)
def test_calculate_risk_score_rejects_values_outside_bounds(
    feature_name: str, out_of_bounds_value: float
) -> None:
    feature_values = {
        "deposits_7d": 0.0,
        "losses_7d": 0.0,
        "night_sessions": 0.0,
        "session_hours": 0.0,
        "deposit_frequency": 0.0,
    }
    feature_values[feature_name] = out_of_bounds_value

    with pytest.raises(ValueError, match="value must be between"):
        calculate_risk_score(**feature_values)


def test_normalize_feature_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="maximum must be greater"):
        normalize_feature(5.0, 10.0, 0.0)


def test_calculate_risk_probability_is_midpoint_and_monotonic() -> None:
    lower_probability = calculate_risk_probability(0, 0, 0, 0, 0)
    midpoint_probability = calculate_risk_probability(1000, 500, 5, 20, 10)
    upper_probability = calculate_risk_probability(2000, 1000, 10, 40, 20)

    assert midpoint_probability == pytest.approx(0.5)
    assert 0.0 < lower_probability < midpoint_probability < upper_probability < 1.0