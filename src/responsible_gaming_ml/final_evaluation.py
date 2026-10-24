"""Refit the selected configuration and report one final educational evaluation."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from responsible_gaming_ml.data_preparation import prepare_train_validation_test_data
from responsible_gaming_ml.logistic_regression import (
    evaluate_logistic_regression,
    evaluate_risk_scores,
    select_risk_probabilities,
)


def combine_train_validation(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Concatenate features and targets in the same order, preserving indices."""
    return pd.concat([X_train, X_validation]), pd.concat([y_train, y_validation])


def train_final_pipeline(X_fit: pd.DataFrame, y_fit: pd.Series) -> Pipeline:
    """Fit the fixed C=100 L2 configuration on train plus validation only."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=100)),
    ])
    pipeline.fit(X_fit, y_fit)
    return pipeline


def evaluate_final_pipeline(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[pd.Series, pd.DataFrame]:
    """Evaluate a fixed pipeline once, using its default prediction threshold."""
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    risk_scores = select_risk_probabilities(probabilities, pipeline.classes_)
    accuracy, precision, recall, f1, matrix = evaluate_logistic_regression(
        y_test, predictions
    )
    roc_auc, average_precision = evaluate_risk_scores(y_test, risk_scores)
    metrics = pd.Series({
        "Accuracy": accuracy,
        "Precision RISK": precision,
        "Recall RISK": recall,
        "F1 RISK": f1,
        "ROC-AUC": roc_auc,
        "Average Precision": average_precision,
        "RISK predictions": int((predictions == "RISK").sum()),
    })
    confusion = pd.DataFrame(
        matrix,
        index=["Actual NO_RISK", "Actual RISK"],
        columns=["Predicted NO_RISK", "Predicted RISK"],
    )
    return metrics, confusion


def main() -> None:
    X_train, X_validation, X_test, y_train, y_validation, y_test = (
        prepare_train_validation_test_data()
    )
    X_fit, y_fit = combine_train_validation(
        X_train, X_validation, y_train, y_validation
    )
    for name, y in [
        ("Train", y_train), ("Validation", y_validation),
        ("Train + Validation", y_fit), ("Test", y_test),
    ]:
        print(f"\n{name}: {len(y)} observations")
        print(pd.DataFrame({
            "count": y.value_counts(),
            "proportion": y.value_counts(normalize=True),
        }).to_string())

    print("\nFixed configuration: StandardScaler + LogisticRegression(C=100).")
    print("The test set was viewed in earlier educational experiments.")
    print("These metrics are not a perfectly independent estimate of performance.")
    pipeline = train_final_pipeline(X_fit, y_fit)
    metrics, confusion = evaluate_final_pipeline(pipeline, X_test, y_test)
    print("\nFinal test metrics:")
    print(metrics.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nConfusion matrix (rows: actual, columns: predicted):")
    print(confusion.to_string())
    print("Synthetic educational data only; not for real Responsible Gaming decisions.")


if __name__ == "__main__":
    main()
