import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from responsible_gaming_ml.data_preparation import prepare_train_validation_test_data
from responsible_gaming_ml.random_forest import (
    compare_random_forest_and_logistic_regression,
    select_best_validation_model,
)


def test_models_share_data_and_report_correct_metrics() -> None:
    X_train, X_validation, _, y_train, y_validation, _ = (
        prepare_train_validation_test_data()
    )
    results, models = compare_random_forest_and_logistic_regression(
        X_train, X_validation, y_train, y_validation
    )
    assert len(models) == len(results) == 4
    forests = [model for model in models.values() if isinstance(model, RandomForestClassifier)]
    assert [model.max_depth for model in forests] == [3, 5, None]
    for model in forests:
        assert not isinstance(model, Pipeline)
        assert model.n_estimators == 100
        assert model.random_state == 42
        assert model.class_weight is None
        # A deterministic refit on raw train data checks rows, labels and no scaling.
        expected = RandomForestClassifier(
            n_estimators=100, max_depth=model.max_depth, random_state=42
        ).fit(X_train, y_train)
        for tree, expected_tree in zip(model.estimators_, expected.estimators_):
            np.testing.assert_array_equal(tree.tree_.threshold, expected_tree.tree_.threshold)
            np.testing.assert_array_equal(tree.tree_.value, expected_tree.tree_.value)

    reference = models["Logistic Regression C=100"]
    assert isinstance(reference, Pipeline)
    assert isinstance(reference.named_steps["scaler"], StandardScaler)
    scaler = reference.named_steps["scaler"]
    assert scaler.n_samples_seen_ == len(X_train) == 3000
    np.testing.assert_allclose(scaler.mean_, X_train.mean())
    assert reference.named_steps["model"].get_params() == LogisticRegression(C=100).get_params()
    expected_lr = LogisticRegression(C=100).fit(scaler.transform(X_train), y_train)
    np.testing.assert_allclose(reference.named_steps["model"].coef_, expected_lr.coef_)

    for name, model in models.items():
        row = results.loc[results["Model"] == name].iloc[0]
        for X, y, subset in [
            (X_train, y_train, "Train"),
            (X_validation, y_validation, "Validation"),
        ]:
            predictions = model.predict(X)
            _, fp, fn, tp = confusion_matrix(
                y, predictions, labels=["NO_RISK", "RISK"]
            ).ravel()
            assert row[f"F1 RISK {subset}"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
            if subset == "Validation":
                precision = tp / (tp + fp) if tp + fp else 0.0
                assert row["Precision RISK Validation"] == pytest.approx(precision)
                assert row["Recall RISK Validation"] == pytest.approx(tp / (tp + fn))
                assert row["RISK predictions Validation"] == tp + fp
    expected_best = results.loc[results["F1 RISK Validation"].idxmax(), "Model"]
    assert select_best_validation_model(results) == expected_best


def test_selection_uses_validation_f1_and_documented_tie_priority() -> None:
    results = pd.DataFrame({
        "Model": [
            "Random Forest depth=None", "Random Forest depth=5",
            "Random Forest depth=3", "Logistic Regression C=100",
        ],
        "F1 RISK Validation": [0.8, 0.8, 0.8, 0.8],
        "F1 RISK Train": [1.0, 0.9, 0.8, 0.7],
    })
    assert select_best_validation_model(results) == "Logistic Regression C=100"
    assert select_best_validation_model(results.iloc[:3]) == "Random Forest depth=3"
    assert select_best_validation_model(results.iloc[:2]) == "Random Forest depth=5"
    results.loc[0, "F1 RISK Validation"] += 0.00000001
    assert select_best_validation_model(results) == "Random Forest depth=None"
    assert select_best_validation_model(results.iloc[::-1]) == "Random Forest depth=None"
