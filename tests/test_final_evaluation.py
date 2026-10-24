import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from responsible_gaming_ml.data_preparation import prepare_train_validation_test_data
from responsible_gaming_ml.final_evaluation import (
    combine_train_validation,
    evaluate_final_pipeline,
    train_final_pipeline,
)


def test_final_training_uses_aligned_train_validation_only() -> None:
    X_train, X_validation, X_test, y_train, y_validation, _ = (
        prepare_train_validation_test_data()
    )
    X_fit, y_fit = combine_train_validation(
        X_train, X_validation, y_train, y_validation
    )
    assert X_fit.shape == (4000, 5)
    assert y_fit.shape == (4000,)
    pd.testing.assert_index_equal(X_fit.index, y_fit.index)
    pd.testing.assert_frame_equal(X_fit.iloc[:3000], X_train)
    pd.testing.assert_frame_equal(X_fit.iloc[3000:], X_validation)
    pd.testing.assert_series_equal(y_fit.iloc[:3000], y_train)
    pd.testing.assert_series_equal(y_fit.iloc[3000:], y_validation)
    assert X_fit.index.is_unique
    assert X_fit.index.intersection(X_test.index).empty
    assert len(X_fit.index.union(X_test.index)) == 5000

    pipeline = train_final_pipeline(X_fit, y_fit)
    assert list(pipeline.named_steps) == ["scaler", "model"]
    scaler = pipeline.named_steps["scaler"]
    assert isinstance(scaler, StandardScaler)
    assert scaler.n_samples_seen_ == 4000
    np.testing.assert_allclose(scaler.mean_, X_fit.mean())
    np.testing.assert_allclose(scaler.scale_, X_fit.std(ddof=0))
    model = pipeline.named_steps["model"]
    assert model.get_params() == LogisticRegression(C=100).get_params()
    assert model.l1_ratio == 0


def test_final_metrics_on_fictitious_predictions() -> None:
    # A small stand-in avoids repeating evaluation on the actual held-out test set.
    class FixedPipeline:
        classes_ = np.array(["RISK", "NO_RISK"])

        def __init__(self) -> None:
            self.predict_calls = 0
            self.probability_calls = 0

        def predict(self, X):
            self.predict_calls += 1
            return np.array(["NO_RISK", "NO_RISK", "NO_RISK", "RISK"])

        def predict_proba(self, X):
            self.probability_calls += 1
            return np.array([[0.1, 0.9], [0.4, 0.6], [0.35, 0.65], [0.8, 0.2]])

    pipeline = FixedPipeline()
    metrics, confusion = evaluate_final_pipeline(
        pipeline,
        pd.DataFrame({"feature": [1, 2, 3, 4]}),
        pd.Series(["NO_RISK", "NO_RISK", "RISK", "RISK"]),
    )
    assert metrics.to_dict() == pytest.approx({
        "Accuracy": 3 / 4,
        "Precision RISK": 1,
        "Recall RISK": 1 / 2,
        "F1 RISK": 2 / 3,
        "ROC-AUC": 3 / 4,
        "Average Precision": (1 + 2 / 3) / 2,
        "RISK predictions": 1,
    })
    pd.testing.assert_frame_equal(confusion, pd.DataFrame(
        [[2, 0], [1, 1]],
        index=["Actual NO_RISK", "Actual RISK"],
        columns=["Predicted NO_RISK", "Predicted RISK"],
    ))
    assert pipeline.predict_calls == pipeline.probability_calls == 1
