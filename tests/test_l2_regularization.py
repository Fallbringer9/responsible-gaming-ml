import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from responsible_gaming_ml.data_preparation import prepare_train_validation_test_data
from responsible_gaming_ml.l2_regularization import (
    compare_l2_regularization,
    select_best_l2_C,
)


def test_l2_pipelines_share_training_data_and_report_correct_metrics() -> None:
    X_train, X_validation, _, y_train, y_validation, _ = (
        prepare_train_validation_test_data()
    )
    results, pipelines = compare_l2_regularization(
        X_train, X_validation, y_train, y_validation
    )

    expected_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    assert results["C"].tolist() == expected_values
    assert list(pipelines) == expected_values
    for C, pipeline in pipelines.items():
        assert list(pipeline.named_steps) == ["scaler", "model"]
        scaler = pipeline.named_steps["scaler"]
        assert isinstance(scaler, StandardScaler)
        assert scaler.n_samples_seen_ == 3000
        np.testing.assert_allclose(scaler.mean_, X_train.mean().to_numpy())
        model = pipeline.named_steps["model"]
        expected_model = LogisticRegression(C=C)
        assert model.l1_ratio == 0.0
        assert model.solver == "lbfgs"
        assert model.get_params() == expected_model.get_params()
        # Independently refit on the exact same rows and labels.
        expected_model.fit(scaler.transform(X_train), y_train)
        np.testing.assert_allclose(model.coef_, expected_model.coef_)
        np.testing.assert_allclose(model.intercept_, expected_model.intercept_)

        row = results.loc[results["C"] == C].iloc[0]
        f1_values = []
        for X, y, column in [
            (X_train, y_train, "F1 RISK Train"),
            (X_validation, y_validation, "F1 RISK Validation"),
        ]:
            predictions = pipeline.predict(X)
            subset = "Train" if column == "F1 RISK Train" else "Validation"
            assert row[f"RISK predictions {subset}"] == (predictions == "RISK").sum()
            _, fp, fn, tp = confusion_matrix(
                y, predictions, labels=["NO_RISK", "RISK"]
            ).ravel()
            f1 = 2 * tp / (2 * tp + fp + fn)
            assert row[column] == pytest.approx(f1)
            f1_values.append(f1)
            if column == "F1 RISK Validation":
                # Match sklearn's default: undefined precision is reported as zero.
                precision = tp / (tp + fp) if tp + fp else 0.0
                assert row["Precision RISK Validation"] == pytest.approx(precision)
                assert row["Recall RISK Validation"] == pytest.approx(tp / (tp + fn))
        assert row["Train - Validation"] == pytest.approx(f1_values[0] - f1_values[1])


def test_selection_uses_only_validation_f1_and_smallest_C_for_exact_ties() -> None:
    results = pd.DataFrame({
        "C": [100.0, 1.0, 0.01],
        "F1 RISK Validation": [0.6, 0.6, 0.5],
        "F1 RISK Train": [1.0, 0.1, 0.9],
        "Precision RISK Validation": [1.0, 0.1, 0.9],
        "Recall RISK Validation": [1.0, 0.1, 0.9],
    })
    assert select_best_l2_C(results) == 1.0
    assert select_best_l2_C(results.iloc[::-1]) == 1.0
    results.loc[0, "F1 RISK Validation"] += 0.00000001
    assert select_best_l2_C(results) == 100.0
