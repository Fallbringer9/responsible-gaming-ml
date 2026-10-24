import pandas as pd
import pytest

from responsible_gaming_ml.baseline import evaluate_baseline, predict_no_risk
from responsible_gaming_ml.data_preparation import prepare_train_test_data


def test_baseline_predicts_no_risk_for_every_test_observation() -> None:
    _, X_test, _, _ = prepare_train_test_data()

    predictions = predict_no_risk(X_test)

    assert len(predictions) == len(X_test)
    assert predictions.eq("NO_RISK").all()
    pd.testing.assert_index_equal(predictions.index, X_test.index)


def test_baseline_predictions_do_not_depend_on_feature_values() -> None:
    features = pd.DataFrame({"deposits_7d": [0.0, 2000.0]}, index=[7, 9])
    changed_features = features * 0

    pd.testing.assert_series_equal(
        predict_no_risk(features), predict_no_risk(changed_features)
    )


def test_baseline_metrics_match_constant_predictions() -> None:
    features = pd.DataFrame({"deposits_7d": [0.0, 100.0, 200.0, 300.0]})
    predictions = predict_no_risk(features)
    target = pd.Series(["NO_RISK", "RISK", "NO_RISK", "NO_RISK"])

    accuracy, risk_recall = evaluate_baseline(target, predictions)

    assert accuracy == pytest.approx(0.75)
    assert risk_recall == pytest.approx(0.0)
