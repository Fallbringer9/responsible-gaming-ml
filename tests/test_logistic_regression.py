import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from responsible_gaming_ml.data_preparation import prepare_train_test_data
from responsible_gaming_ml.logistic_regression import (
    calculate_balanced_class_weights,
    evaluate_logistic_regression,
    evaluate_risk_scores,
    predict_with_threshold,
    scale_train_test_features,
    select_risk_probabilities,
    train_balanced_logistic_regression_pipeline,
    train_logistic_regression,
    train_logistic_regression_pipeline,
)


def test_risk_scores_match_known_roc_auc_and_average_precision() -> None:
    target = pd.Series(["NO_RISK", "NO_RISK", "RISK", "RISK"])
    # Descending scores rank RISK, NO_RISK, RISK, NO_RISK.
    scores = np.array([0.1, 0.4, 0.35, 0.8])

    roc_auc, average_precision = evaluate_risk_scores(target, scores)

    assert roc_auc == pytest.approx(3 / 4)
    assert average_precision == pytest.approx((1 + 2 / 3) / 2)


def test_risk_is_positive_class_for_score_metrics() -> None:
    target = pd.Series(["RISK", "NO_RISK", "NO_RISK", "NO_RISK"])

    roc_auc, average_precision = evaluate_risk_scores(target, np.array([0.9, 0.3, 0.2, 0.1]))

    assert roc_auc == pytest.approx(1)
    assert average_precision == pytest.approx(1)


@pytest.mark.parametrize(
    "train_pipeline",
    [train_logistic_regression_pipeline, train_balanced_logistic_regression_pipeline],
)
def test_pipeline_risk_score_metrics_are_bounded(train_pipeline) -> None:
    X_train, X_test, y_train, y_test = prepare_train_test_data()
    pipeline = train_pipeline(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)
    scores = select_risk_probabilities(probabilities, pipeline.classes_)

    roc_auc, average_precision = evaluate_risk_scores(y_test, scores)

    risk_column = list(pipeline.classes_).index("RISK")
    np.testing.assert_array_equal(scores, probabilities[:, risk_column])
    assert 0 <= roc_auc <= 1
    assert 0 <= average_precision <= 1


def test_balanced_and_reference_pipelines_use_same_split_and_default_parameters() -> None:
    X_train, X_test, y_train, y_test = prepare_train_test_data()
    reference = train_logistic_regression_pipeline(X_train, y_train)
    balanced = train_balanced_logistic_regression_pipeline(X_train, y_train)

    assert reference.named_steps["model"].class_weight is None
    assert balanced.named_steps["model"].class_weight == "balanced"
    for pipeline in [reference, balanced]:
        assert list(pipeline.named_steps) == ["scaler", "model"]
        scaler = pipeline.named_steps["scaler"]
        assert isinstance(scaler, StandardScaler)
        assert scaler.get_params() == StandardScaler().get_params()
        assert scaler.n_samples_seen_ == len(X_train)
        np.testing.assert_allclose(scaler.mean_, X_train.mean().to_numpy())
        np.testing.assert_allclose(scaler.scale_, X_train.std(ddof=0).to_numpy())
        probabilities = select_risk_probabilities(
            pipeline.predict_proba(X_test), pipeline.classes_
        )
        predictions = predict_with_threshold(probabilities, 0.5)
        assert predictions.shape == (len(y_test),)
        assert set(predictions).issubset({"NO_RISK", "RISK"})
    expected_parameters = LogisticRegression().get_params()
    expected_parameters["class_weight"] = "balanced"
    assert balanced.named_steps["model"].get_params() == expected_parameters


def test_balanced_weights_use_training_label_frequencies_only() -> None:
    y_train = pd.Series(["NO_RISK"] * 6 + ["RISK"] * 2)
    y_test = pd.Series(["NO_RISK"] + ["RISK"] * 7)

    weights = calculate_balanced_class_weights(y_train)

    assert weights == pytest.approx({"NO_RISK": 8 / 12, "RISK": 8 / 4})
    assert weights != calculate_balanced_class_weights(pd.concat([y_train, y_test]))
    classes = np.array(["NO_RISK", "RISK"])
    sklearn_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    np.testing.assert_allclose([weights[label] for label in classes], sklearn_weights)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.29, "NO_RISK"), (0.31, "RISK"), (0.3, "RISK")],
)
def test_threshold_comparison(probability: float, expected: str) -> None:
    predictions = predict_with_threshold(np.array([probability]), threshold=0.3)

    np.testing.assert_array_equal(predictions, [expected])


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_threshold_rejects_invalid_values(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold must be between 0 and 1 inclusive"):
        predict_with_threshold(np.array([0.5]), threshold)


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(0.0, ["RISK", "RISK", "RISK"]), (1.0, ["NO_RISK", "NO_RISK", "RISK"])],
)
def test_threshold_accepts_endpoints(threshold: float, expected: list[str]) -> None:
    predictions = predict_with_threshold(np.array([0.0, 0.5, 1.0]), threshold)

    np.testing.assert_array_equal(predictions, expected)


@pytest.mark.parametrize("classes", [["NO_RISK", "RISK"], ["RISK", "NO_RISK"]])
def test_risk_probability_column_follows_class_order(classes: list[str]) -> None:
    probabilities_by_class = {"NO_RISK": [0.8, 0.1], "RISK": [0.2, 0.9]}
    probabilities = np.column_stack([probabilities_by_class[label] for label in classes])

    risk_probabilities = select_risk_probabilities(probabilities, np.array(classes))

    np.testing.assert_array_equal(risk_probabilities, [0.2, 0.9])


def test_threshold_half_matches_pipeline_predict_on_existing_test_data() -> None:
    X_train, X_test, y_train, _ = prepare_train_test_data()
    pipeline = train_logistic_regression_pipeline(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)
    risk_probabilities = select_risk_probabilities(probabilities, pipeline.classes_)
    predictions = predict_with_threshold(risk_probabilities, threshold=0.5)

    np.testing.assert_array_equal(predictions, pipeline.predict(X_test))


def test_pipeline_has_scaler_then_default_logistic_regression() -> None:
    X_train, _, y_train, _ = prepare_train_test_data()

    pipeline = train_logistic_regression_pipeline(X_train, y_train)

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scaler", "model"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["model"], LogisticRegression)
    assert pipeline.named_steps["scaler"].get_params() == StandardScaler().get_params()
    assert pipeline.named_steps["model"].get_params() == LogisticRegression().get_params()


def test_pipeline_scaler_learns_only_from_train() -> None:
    X_train = pd.DataFrame({"hours": [1.0, 3.0, 5.0, 7.0]})
    X_test = pd.DataFrame({"hours": [1000.0]})
    y_train = pd.Series(["NO_RISK", "NO_RISK", "RISK", "RISK"])
    pipeline = train_logistic_regression_pipeline(X_train, y_train)

    pipeline.predict(X_test)

    scaler = pipeline.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, X_train.mean().to_numpy())
    np.testing.assert_allclose(scaler.scale_, X_train.std(ddof=0).to_numpy())
    assert scaler.n_samples_seen_ == len(X_train)


def test_pipeline_matches_manual_scaling_and_preserves_original_data() -> None:
    X_train, X_test, y_train, y_test = prepare_train_test_data()
    original_train = X_train.copy(deep=True)
    original_test = X_test.copy(deep=True)
    original_train_target = y_train.copy(deep=True)
    original_test_target = y_test.copy(deep=True)
    _, train_scaled, test_scaled = scale_train_test_features(X_train, X_test)
    manual_model = train_logistic_regression(train_scaled, y_train)
    manual_predictions = manual_model.predict(test_scaled)

    pipeline = train_logistic_regression_pipeline(X_train, y_train)
    predictions = pipeline.predict(X_test)

    np.testing.assert_array_equal(predictions, manual_predictions)
    metrics = evaluate_logistic_regression(y_test, predictions)
    manual_metrics = evaluate_logistic_regression(y_test, manual_predictions)
    assert metrics[:4] == manual_metrics[:4]
    np.testing.assert_array_equal(metrics[4], manual_metrics[4])
    pd.testing.assert_frame_equal(X_train, original_train)
    pd.testing.assert_frame_equal(X_test, original_test)
    pd.testing.assert_series_equal(y_train, original_train_target)
    pd.testing.assert_series_equal(y_test, original_test_target)


def test_scaling_learns_only_from_train_and_preserves_original_data() -> None:
    X_train = pd.DataFrame({"hours": [1.0, 3.0], "deposits": [10.0, 30.0]})
    X_test = pd.DataFrame({"hours": [100.0], "deposits": [1000.0]})
    original_train = X_train.copy()
    original_test = X_test.copy()

    scaler, train_scaled, test_scaled = scale_train_test_features(X_train, X_test)

    np.testing.assert_allclose(scaler.mean_, X_train.mean().to_numpy())
    np.testing.assert_allclose(scaler.scale_, X_train.std(ddof=0).to_numpy())
    assert scaler.n_samples_seen_ == len(X_train)
    assert train_scaled.shape == X_train.shape
    assert test_scaled.shape == X_test.shape
    np.testing.assert_allclose(train_scaled.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(train_scaled.std(axis=0), 1)
    np.testing.assert_allclose(
        test_scaled, (X_test.to_numpy() - scaler.mean_) / scaler.scale_
    )
    pd.testing.assert_frame_equal(X_train, original_train)
    pd.testing.assert_frame_equal(X_test, original_test)


def test_scaled_model_predictions_and_evaluation_are_consistent() -> None:
    X_train, X_test, y_train, y_test = prepare_train_test_data()
    original_target = y_train.copy()
    _, train_scaled, test_scaled = scale_train_test_features(X_train, X_test)

    model = train_logistic_regression(train_scaled, y_train)
    predictions = model.predict(test_scaled)
    accuracy, precision, recall, f1, matrix = evaluate_logistic_regression(
        y_test, predictions
    )

    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({"NO_RISK", "RISK"})
    assert model.get_params() == train_logistic_regression(X_train, y_train).get_params()
    assert matrix.shape == (2, 2)
    assert matrix.sum() == len(y_test)
    tn, fp, fn, tp = matrix.ravel()
    assert accuracy == pytest.approx((tn + tp) / len(y_test))
    assert precision == pytest.approx(tp / (tp + fp))
    assert recall == pytest.approx(tp / (tp + fn))
    assert f1 == pytest.approx(2 * tp / (2 * tp + fp + fn))
    pd.testing.assert_series_equal(y_train, original_target)


def test_evaluation_uses_risk_as_positive_class_and_explicit_matrix_order() -> None:
    # TN = 3, FP = 1, FN = 2, TP = 2.
    target = pd.Series(["NO_RISK"] * 4 + ["RISK"] * 4)
    predictions = np.array(
        ["NO_RISK", "NO_RISK", "NO_RISK", "RISK", "NO_RISK", "NO_RISK", "RISK", "RISK"]
    )

    accuracy, precision, recall, f1, matrix = evaluate_logistic_regression(
        target, predictions
    )

    assert accuracy == pytest.approx(5 / 8)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 4)
    assert f1 == pytest.approx(4 / 7)
    np.testing.assert_array_equal(matrix, [[3, 1], [2, 2]])


@pytest.mark.parametrize("target", [["NO_RISK", "RISK"], ["NO_RISK", "NO_RISK"]])
def test_evaluation_handles_no_risk_predictions_explicitly(target: list[str]) -> None:
    predictions = np.array(["NO_RISK", "NO_RISK"])

    _, precision, recall, f1, matrix = evaluate_logistic_regression(
        pd.Series(target), predictions
    )

    assert precision == recall == f1 == 0.0
    assert matrix[:, 1].sum() == 0


def test_trained_model_predicts_valid_labels_for_test_observations() -> None:
    X_train, X_test, y_train, _ = prepare_train_test_data()
    original_features = X_train.copy()
    original_target = y_train.copy()

    model = train_logistic_regression(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({"RISK", "NO_RISK"})
    assert set(model.classes_) == {"RISK", "NO_RISK"}
    assert model.n_features_in_ == 5
    pd.testing.assert_frame_equal(X_train, original_features)
    pd.testing.assert_series_equal(y_train, original_target)


def test_model_learns_from_training_labels() -> None:
    features = pd.DataFrame({"session_hours": [-3.0, -2.0, 2.0, 3.0]})
    target = pd.Series(["NO_RISK", "NO_RISK", "RISK", "RISK"])
    reversed_target = pd.Series(["RISK", "RISK", "NO_RISK", "NO_RISK"])

    model = train_logistic_regression(features, target)
    reversed_model = train_logistic_regression(features, reversed_target)

    np.testing.assert_array_equal(model.predict(features), target.to_numpy())
    np.testing.assert_array_equal(
        reversed_model.predict(features), reversed_target.to_numpy()
    )
