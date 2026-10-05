"""Educational probability calculation for synthetic data only."""

import math

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "deposits_7d": (0.0, 2000.0),
    "losses_7d": (0.0, 1000.0),
    "night_sessions": (0.0, 10.0),
    "session_hours": (0.0, 40.0),
    "deposit_frequency": (0.0, 20.0),
}

FEATURE_WEIGHTS: dict[str, float] = {
    "losses_7d": 0.35,
    "session_hours": 0.30,
    "deposit_frequency": 0.20,
    "night_sessions": 0.10,
    "deposits_7d": 0.05,
}


def normalize_feature(value: float, minimum: float, maximum: float) -> float:
    if minimum >= maximum:
        raise ValueError("maximum must be greater than minimum")
    if not minimum <= value <= maximum:
        raise ValueError(f"value must be between {minimum} and {maximum}")
    return (value - minimum) / (maximum - minimum)


def calculate_risk_score(
    deposits_7d: float,
    losses_7d: float,
    night_sessions: float,
    session_hours: float,
    deposit_frequency: float,
) -> float:
    feature_values = {
        "deposits_7d": deposits_7d,
        "losses_7d": losses_7d,
        "night_sessions": night_sessions,
        "session_hours": session_hours,
        "deposit_frequency": deposit_frequency,
    }
    normalized_values = {
        name: normalize_feature(value, *FEATURE_BOUNDS[name])
        for name, value in feature_values.items()
    }
    return sum(
        normalized_values[name] * weight
        for name, weight in FEATURE_WEIGHTS.items()
    )


def calculate_risk_probability(
    deposits_7d: float,
    losses_7d: float,
    night_sessions: float,
    session_hours: float,
    deposit_frequency: float,
) -> float:
    risk_score = calculate_risk_score(
        deposits_7d=deposits_7d,
        losses_7d=losses_7d,
        night_sessions=night_sessions,
        session_hours=session_hours,
        deposit_frequency=deposit_frequency,
    )
    z = (risk_score - 0.5) * 10
    return 1 / (1 + math.exp(-z))