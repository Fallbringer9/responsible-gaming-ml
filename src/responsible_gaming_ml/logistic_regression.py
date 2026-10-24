"""Train a first logistic regression on synthetic educational data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from responsible_gaming_ml.data_preparation import prepare_train_test_data


def train_logistic_regression(
    X_train: pd.DataFrame | np.ndarray, y_train: pd.Series
) -> LogisticRegression:
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def train_logistic_regression_pipeline(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """Fit scaling and logistic regression using only training data."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_balanced_logistic_regression_pipeline(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """Fit a balanced logistic regression pipeline on training data only."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def calculate_balanced_class_weights(y_train: pd.Series) -> dict[str, float]:
    """Calculate inverse-frequency class weights using training labels only."""
    class_counts = y_train.value_counts()
    return {
        label: len(y_train) / (len(class_counts) * count)
        for label, count in class_counts.items()
    }


def scale_train_test_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    """Learn scaling statistics on training features and transform both sets."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def evaluate_logistic_regression(
    y_test: pd.Series, predictions: np.ndarray
) -> tuple[float, float, float, float, np.ndarray]:
    """Return RISK metrics and a matrix with NO_RISK before RISK."""
    accuracy = accuracy_score(y_test, predictions)
    risk_precision = precision_score(y_test, predictions, pos_label="RISK", zero_division=0)
    risk_recall = recall_score(y_test, predictions, pos_label="RISK", zero_division=0)
    risk_f1 = f1_score(y_test, predictions, pos_label="RISK", zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=["NO_RISK", "RISK"])
    return accuracy, risk_precision, risk_recall, risk_f1, matrix


def select_risk_probabilities(
    probabilities: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """Select the RISK column using the classifier's class order."""
    risk_column = list(classes).index("RISK")
    return probabilities[:, risk_column]


def predict_with_threshold(
    risk_probabilities: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Predict RISK when its probability is at least the threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1 inclusive")
    return np.where(risk_probabilities >= threshold, "RISK", "NO_RISK")


def evaluate_risk_scores(
    y_test: pd.Series, risk_probabilities: np.ndarray
) -> tuple[float, float]:
    """Return ROC-AUC and AP with RISK explicitly encoded as positive."""
    is_risk = y_test == "RISK"
    return (
        roc_auc_score(is_risk, risk_probabilities),
        average_precision_score(is_risk, risk_probabilities),
    )


def plot_risk_score_curves(
    y_test: pd.Series, scores_by_model: dict[str, np.ndarray]
) -> plt.Figure:
    """Plot ROC and Precision-Recall curves for educational comparison."""
    is_risk = y_test == "RISK"
    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(12, 5))
    for name, scores in scores_by_model.items():
        roc_auc, average_precision = evaluate_risk_scores(y_test, scores)
        false_positive_rate, true_positive_rate, _ = roc_curve(is_risk, scores)
        precision, recall, _ = precision_recall_curve(is_risk, scores)
        roc_axis.plot(
            false_positive_rate, true_positive_rate,
            label=f"{name} (ROC-AUC={roc_auc:.4f})",
        )
        pr_axis.step(
            recall, precision, where="post",
            label=f"{name} (AP={average_precision:.4f})",
        )
    roc_axis.plot([0, 1], [0, 1], "k--", label="Random ranking reference")
    pr_axis.axhline(
        is_risk.mean(), color="black", linestyle="--",
        label=f"RISK prevalence ({is_risk.mean():.2%})",
    )
    roc_axis.set(xlabel="False positive rate", ylabel="True positive rate (RISK recall)",
                 title="ROC — synthetic test data")
    pr_axis.set(xlabel="Recall RISK", ylabel="Precision RISK",
                title="Precision-Recall — synthetic test data")
    for axis in (roc_axis, pr_axis):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.05)
        axis.legend(loc="lower left")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def main() -> None:
    X_train, X_test, y_train, y_test = prepare_train_test_data()
    model = train_logistic_regression(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy, risk_precision, risk_recall, risk_f1, matrix = (
        evaluate_logistic_regression(y_test, predictions)
    )
    tn, fp, fn, tp = matrix.ravel()

    print(f"Training observations: {len(X_train)}")
    print(f"Test observations: {len(X_test)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision RISK: {risk_precision:.2%}")
    print(f"Recall RISK: {risk_recall:.2%}")
    print(f"F1 RISK: {risk_f1:.2%}")
    print("\nConfusion matrix (rows: actual, columns: predicted):")
    print(
        pd.DataFrame(
            matrix,
            index=["Actual NO_RISK", "Actual RISK"],
            columns=["Predicted NO_RISK", "Predicted RISK"],
        ).to_string()
    )
    print(f"TN (actual NO_RISK, predicted NO_RISK): {tn}")
    print(f"FP (actual NO_RISK, predicted RISK): {fp}")
    print(f"FN (actual RISK, predicted NO_RISK): {fn}")
    print(f"TP (actual RISK, predicted RISK): {tp}")
    print("\nBaseline accuracy: 96.20%")
    print("Baseline recall RISK: 0.00%")
    print("\nPrediction counts:")
    print(pd.Series(predictions).value_counts().to_string())
    print("\nModel configuration:")
    print(model.get_params())
    print("\nLearned coefficients (feature order):")
    print(list(X_train.columns))
    print(model.coef_)
    print(f"Intercept: {model.intercept_}")

    scaler, X_train_scaled, X_test_scaled = scale_train_test_features(X_train, X_test)
    scaled_model = train_logistic_regression(X_train_scaled, y_train)
    scaled_predictions = scaled_model.predict(X_test_scaled)
    scaled_accuracy, scaled_precision, scaled_recall, scaled_f1, scaled_matrix = (
        evaluate_logistic_regression(y_test, scaled_predictions)
    )

    print("\nLogistic Regression without scaling VS with StandardScaler:")
    print(
        pd.DataFrame(
            {
                "Without scaling": [accuracy, risk_precision, risk_recall, risk_f1],
                "With StandardScaler": [
                    scaled_accuracy, scaled_precision, scaled_recall, scaled_f1
                ],
            },
            index=["Accuracy", "Precision RISK", "Recall RISK", "F1 RISK"],
        ).to_string(float_format=lambda value: f"{value:.2%}")
    )
    print("\nScaled confusion matrix (rows: actual, columns: predicted):")
    print(
        pd.DataFrame(
            scaled_matrix,
            index=["Actual NO_RISK", "Actual RISK"],
            columns=["Predicted NO_RISK", "Predicted RISK"],
        ).to_string()
    )
    tn, fp, fn, tp = scaled_matrix.ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print("\nTraining scaling statistics and scaled model coefficients by feature:")
    print(
        pd.DataFrame(
            {
                "mean_": scaler.mean_,
                "scale_": scaler.scale_,
                "Scaled coefficient (RISK)": scaled_model.coef_[0],
            },
            index=X_train.columns,
        ).to_string()
    )
    print(f"Scaled model intercept: {scaled_model.intercept_}")

    pipeline = train_logistic_regression_pipeline(X_train, y_train)
    pipeline_predictions = pipeline.predict(X_test)
    pipeline_accuracy, pipeline_precision, pipeline_recall, pipeline_f1, pipeline_matrix = (
        evaluate_logistic_regression(y_test, pipeline_predictions)
    )
    print("\nManual StandardScaler VS Pipeline:")
    print(
        pd.DataFrame(
            {
                "Manual scaling": [
                    scaled_accuracy, scaled_precision, scaled_recall, scaled_f1
                ],
                "Pipeline": [
                    pipeline_accuracy, pipeline_precision, pipeline_recall, pipeline_f1
                ],
            },
            index=["Accuracy", "Precision RISK", "Recall RISK", "F1 RISK"],
        ).to_string(float_format=lambda value: f"{value:.2%}")
    )
    print("\nPipeline confusion matrix (rows: actual, columns: predicted):")
    print(
        pd.DataFrame(
            pipeline_matrix,
            index=["Actual NO_RISK", "Actual RISK"],
            columns=["Predicted NO_RISK", "Predicted RISK"],
        ).to_string()
    )
    print(f"Identical predictions: {np.array_equal(scaled_predictions, pipeline_predictions)}")
    print(f"Identical confusion matrices: {np.array_equal(scaled_matrix, pipeline_matrix)}")

    probabilities = pipeline.predict_proba(X_test)
    risk_probabilities = select_risk_probabilities(probabilities, pipeline.classes_)
    threshold_predictions = predict_with_threshold(risk_probabilities, threshold=0.5)
    np.testing.assert_array_equal(threshold_predictions, pipeline_predictions)
    print("\nThreshold 0.5 predictions match pipeline.predict(): True")
    print("Threshold comparison on synthetic test data (no threshold selection):")
    for threshold in [0.3, 0.5, 0.7]:
        threshold_predictions = predict_with_threshold(risk_probabilities, threshold)
        _, precision, recall, f1, threshold_matrix = evaluate_logistic_regression(
            y_test, threshold_predictions
        )
        print(f"\nThreshold: {threshold}")
        print(f"RISK predictions: {np.count_nonzero(threshold_predictions == 'RISK')}")
        print(f"Precision RISK: {precision:.2%}")
        print(f"Recall RISK: {recall:.2%}")
        print(f"F1 RISK: {f1:.2%}")
        print("Confusion matrix (rows: actual, columns: predicted):")
        print(
            pd.DataFrame(
                threshold_matrix,
                index=["Actual NO_RISK", "Actual RISK"],
                columns=["Predicted NO_RISK", "Predicted RISK"],
            ).to_string()
        )

    balanced_pipeline = train_balanced_logistic_regression_pipeline(X_train, y_train)
    print("\nBalanced class weights calculated from y_train only:")
    print(calculate_balanced_class_weights(y_train))
    print("\nClass weight comparison at threshold 0.5 (no model selection):")
    for name, comparison_pipeline in [
        ("Reference (class_weight=None)", pipeline),
        ("Balanced (class_weight='balanced')", balanced_pipeline),
    ]:
        comparison_probabilities = select_risk_probabilities(
            comparison_pipeline.predict_proba(X_test), comparison_pipeline.classes_
        )
        comparison_predictions = predict_with_threshold(comparison_probabilities, 0.5)
        accuracy, precision, recall, f1, comparison_matrix = evaluate_logistic_regression(
            y_test, comparison_predictions
        )
        print(f"\n{name}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision RISK: {precision:.2%}")
        print(f"Recall RISK: {recall:.2%}")
        print(f"F1 RISK: {f1:.2%}")
        print(f"RISK predictions: {np.count_nonzero(comparison_predictions == 'RISK')}")
        print("Confusion matrix (rows: actual, columns: predicted):")
        print(
            pd.DataFrame(
                comparison_matrix,
                index=["Actual NO_RISK", "Actual RISK"],
                columns=["Predicted NO_RISK", "Predicted RISK"],
            ).to_string()
        )


    scores_by_model = {}
    score_rows = []
    for name, score_pipeline in [("Reference", pipeline), ("Balanced", balanced_pipeline)]:
        scores = select_risk_probabilities(
            score_pipeline.predict_proba(X_test), score_pipeline.classes_
        )
        scores_by_model[name] = scores
        roc_auc, average_precision = evaluate_risk_scores(y_test, scores)
        score_rows.append({
            "Model": name,
            "ROC-AUC": roc_auc,
            "Average Precision": average_precision,
            "RISK prevalence": (y_test == "RISK").mean(),
        })
    print("\nROC-AUC / Average Precision on synthetic test data (no model selection):")
    print(pd.DataFrame(score_rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("AP is not the trapezoidal area under the Precision-Recall curve.")
    figure = plot_risk_score_curves(y_test, scores_by_model)
    figure.savefig("risk_score_curves.png", dpi=150)
    print("Curves saved to risk_score_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
