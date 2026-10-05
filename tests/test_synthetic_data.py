import numpy as np
import pandas as pd
import pytest

from responsible_gaming_ml.risk_probability import FEATURE_BOUNDS
from responsible_gaming_ml.synthetic_data import (
    generate_dataset,
    generate_features,
    generate_target,
)


def test_generate_features_returns_expected_columns_and_bounds() -> None:
    features = generate_features(n_players=100, seed=7)

    assert list(features.columns) == [
        "deposits_7d",
        "losses_7d",
        "night_sessions",
        "session_hours",
        "deposit_frequency",
    ]
    assert len(features) == 100

    for feature_name, (minimum, maximum) in FEATURE_BOUNDS.items():
        assert features[feature_name].between(minimum, maximum).all()

    assert pd.api.types.is_integer_dtype(features["night_sessions"])
    assert pd.api.types.is_integer_dtype(features["deposit_frequency"])


def test_generate_features_is_reproducible_for_same_seed() -> None:
    first = generate_features(n_players=100, seed=42)
    second = generate_features(n_players=100, seed=42)

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("n_players", [0, -1])
def test_generate_features_rejects_non_positive_player_count(
    n_players: int,
) -> None:
    with pytest.raises(ValueError, match="n_players must be greater than zero"):
        generate_features(n_players=n_players)


def test_generate_features_has_expected_positive_relationships() -> None:
    features = generate_features(n_players=5_000, seed=42)
    correlations = features.corr()

    assert correlations.loc["deposit_frequency", "deposits_7d"] > 0.5
    assert correlations.loc["session_hours", "night_sessions"] > 0.2
    assert correlations.loc["deposits_7d", "losses_7d"] > 0.5
    assert correlations.loc["session_hours", "losses_7d"] > 0.05


def test_generate_target_is_reproducible_and_preserves_features() -> None:
    features = generate_features(n_players=100, seed=7)
    features.index = pd.RangeIndex(100, 200)
    original = features.copy()

    first = generate_target(features, seed=42)
    second = generate_target(features, seed=42)

    pd.testing.assert_series_equal(first, second)
    pd.testing.assert_frame_equal(features, original)
    pd.testing.assert_index_equal(first.index, features.index)
    assert len(first) == len(features)
    assert first.isin(["RISK", "NO_RISK"]).all()
    assert first.name == "target"


def test_generate_target_samples_using_probability() -> None:
    # At the midpoint of all bounds, the synthetic RISK probability is 0.5.
    features = pd.DataFrame(
        {
            name: [(minimum + maximum) / 2] * 20
            for name, (minimum, maximum) in FEATURE_BOUNDS.items()
        }
    )
    draws = np.random.default_rng(7).uniform(0.0, 1.0, size=20)
    expected = pd.Series(
        np.where(draws < 0.5, "RISK", "NO_RISK"), name="target", dtype="str"
    )

    pd.testing.assert_series_equal(generate_target(features, seed=7), expected)
    assert set(expected) == {"RISK", "NO_RISK"}


def test_generate_dataset_has_only_features_and_target() -> None:
    dataset = generate_dataset(n_players=100, seed=7)
    features = generate_features(n_players=100, seed=7)

    assert list(dataset.columns) == [*features.columns, "target"]
    assert "risk_score" not in dataset.columns
    assert "risk_probability" not in dataset.columns
    assert len(dataset) == 100
    assert dataset["target"].isin(["RISK", "NO_RISK"]).all()
    pd.testing.assert_frame_equal(dataset.drop(columns="target"), features)
    pd.testing.assert_series_equal(dataset["target"], generate_target(features, seed=7))


def test_generate_dataset_is_reproducible_for_same_seed() -> None:
    first = generate_dataset(n_players=100, seed=42)
    second = generate_dataset(n_players=100, seed=42)

    pd.testing.assert_frame_equal(first, second)
