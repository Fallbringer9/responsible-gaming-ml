import pandas as pd
import pytest

from responsible_gaming_ml.data_preparation import (
    prepare_train_test_data,
    prepare_train_validation_test_data,
)
from responsible_gaming_ml.synthetic_data import generate_dataset


def test_split_sizes_features_and_target_alignment() -> None:
    X_train, X_test, y_train, y_test = prepare_train_test_data()
    dataset = generate_dataset()
    expected_features = [
        "deposits_7d",
        "losses_7d",
        "night_sessions",
        "session_hours",
        "deposit_frequency",
    ]

    assert X_train.shape == (4000, 5)
    assert X_test.shape == (1000, 5)
    assert y_train.shape == (4000,)
    assert y_test.shape == (1000,)

    for X, y in [(X_train, y_train), (X_test, y_test)]:
        assert list(X.columns) == expected_features
        assert y.name == "target"
        pd.testing.assert_index_equal(X.index, y.index)
        pd.testing.assert_frame_equal(X, dataset.loc[X.index, expected_features])
        pd.testing.assert_series_equal(y, dataset.loc[y.index, "target"])

    assert X_train.index.intersection(X_test.index).empty
    combined_index = X_train.index.union(X_test.index)
    pd.testing.assert_index_equal(combined_index, dataset.index)


def test_split_is_reproducible() -> None:
    first = prepare_train_test_data()
    second = prepare_train_test_data()

    for first_features, second_features in zip(first[:2], second[:2]):
        pd.testing.assert_frame_equal(first_features, second_features)
    for first_target, second_target in zip(first[2:], second[2:]):
        pd.testing.assert_series_equal(first_target, second_target)


def test_split_preserves_class_proportions() -> None:
    _, _, y_train, y_test = prepare_train_test_data()
    original_proportions = generate_dataset()["target"].value_counts(normalize=True)

    for y in [y_train, y_test]:
        assert set(y) == {"RISK", "NO_RISK"}
        proportions = y.value_counts(normalize=True)
        for label in ["RISK", "NO_RISK"]:
            # Allow at most one observation of rounding in each subset.
            assert proportions[label] == pytest.approx(
                original_proportions[label], abs=1 / len(y)
            )


def test_three_way_split_sizes_alignment_and_complete_partition() -> None:
    X_train, X_validation, X_test, y_train, y_validation, y_test = (
        prepare_train_validation_test_data()
    )
    dataset = generate_dataset()
    features = dataset.drop(columns="target")
    subsets = [(X_train, y_train), (X_validation, y_validation), (X_test, y_test)]

    for (X, y), expected_size in zip(subsets, [3000, 1000, 1000]):
        assert len(X) == len(y) == expected_size
        assert X.index.is_unique
        pd.testing.assert_index_equal(X.index, y.index)
        pd.testing.assert_frame_equal(X, features.loc[X.index])
        pd.testing.assert_series_equal(y, dataset.loc[y.index, "target"])

    assert X_train.index.intersection(X_validation.index).empty
    assert X_train.index.intersection(X_test.index).empty
    assert X_validation.index.intersection(X_test.index).empty
    combined_index = X_train.index.union(X_validation.index).union(X_test.index)
    pd.testing.assert_index_equal(combined_index, dataset.index)


def test_three_way_split_is_reproducible_and_keeps_existing_test_set() -> None:
    first = prepare_train_validation_test_data()
    second = prepare_train_validation_test_data()
    for first_features, second_features in zip(first[:3], second[:3]):
        pd.testing.assert_frame_equal(first_features, second_features)
    for first_target, second_target in zip(first[3:], second[3:]):
        pd.testing.assert_series_equal(first_target, second_target)

    _, X_test, _, y_test = prepare_train_test_data()
    pd.testing.assert_frame_equal(first[2], X_test)
    pd.testing.assert_series_equal(first[5], y_test)


def test_three_way_split_preserves_class_proportions() -> None:
    original_proportions = generate_dataset()["target"].value_counts(normalize=True)
    for y in prepare_train_validation_test_data()[3:]:
        assert set(y) == {"RISK", "NO_RISK"}
        proportions = y.value_counts(normalize=True)
        for label in ["RISK", "NO_RISK"]:
            # Allow at most one observation of rounding in each subset.
            assert proportions[label] == pytest.approx(
                original_proportions[label], abs=1 / len(y)
            )
