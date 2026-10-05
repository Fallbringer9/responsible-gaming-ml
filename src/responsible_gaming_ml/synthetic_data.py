"""Generate synthetic features and labels for educational use only."""

import numpy as np
import pandas as pd

from responsible_gaming_ml.risk_probability import (
    FEATURE_BOUNDS,
    calculate_risk_probability,
)


def generate_features(n_players: int = 5_000, seed: int = 42) -> pd.DataFrame:
    if n_players <= 0:
        raise ValueError("n_players must be greater than zero")

    rng = np.random.default_rng(seed)

    frequency_min, frequency_max = FEATURE_BOUNDS["deposit_frequency"]
    deposit_frequency = np.rint(
        rng.triangular(frequency_min, 3.0, frequency_max, size=n_players)
    ).astype(int)

    average_deposit_amount = rng.uniform(35.0, 75.0, size=n_players)
    deposits_7d = deposit_frequency * average_deposit_amount

    hours_min, hours_max = FEATURE_BOUNDS["session_hours"]
    session_hours = hours_min + rng.beta(2.0, 5.0, size=n_players) * (
        hours_max - hours_min
    )

    night_min, night_max = FEATURE_BOUNDS["night_sessions"]
    night_probability = 0.5 * (session_hours - hours_min) / (hours_max - hours_min)
    night_sessions = rng.binomial(
        n=int(night_max - night_min), p=night_probability, size=n_players
    )

    losses_7d = (0.4 * deposits_7d + 4.0 * session_hours) * rng.uniform(
        0.8, 1.2, size=n_players
    )

    return pd.DataFrame(
        {
            "deposits_7d": deposits_7d,
            "losses_7d": losses_7d,
            "night_sessions": night_sessions,
            "session_hours": session_hours,
            "deposit_frequency": deposit_frequency,
        }
    )


def generate_target(features: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Sample educational labels without modifying the input features."""
    rng = np.random.default_rng(seed)
    labels = []

    for player in features.itertuples(index=False):
        probability = calculate_risk_probability(
            deposits_7d=player.deposits_7d,
            losses_7d=player.losses_7d,
            night_sessions=player.night_sessions,
            session_hours=player.session_hours,
            deposit_frequency=player.deposit_frequency,
        )
        draw = rng.uniform(0.0, 1.0)
        labels.append("RISK" if draw < probability else "NO_RISK")

    return pd.Series(labels, index=features.index, name="target", dtype="str")


def generate_dataset(n_players: int = 5_000, seed: int = 42) -> pd.DataFrame:
    """Return only the five features and the synthetic classification target."""
    features = generate_features(n_players=n_players, seed=seed)
    dataset = features.copy()
    dataset["target"] = generate_target(features, seed=seed)
    return dataset
