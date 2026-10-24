"""Evaluate a constant NO_RISK baseline on synthetic educational data."""

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

from responsible_gaming_ml.data_preparation import prepare_train_test_data


def predict_no_risk(X_test: pd.DataFrame) -> pd.Series:
    """Predict NO_RISK without inspecting feature values or true labels."""
    return pd.Series(["NO_RISK"] * len(X_test), index=X_test.index, dtype="str")


def evaluate_baseline(y_test: pd.Series, predictions: pd.Series) -> tuple[float, float]:
    """Return accuracy and recall for the RISK class."""
    accuracy = accuracy_score(y_test, predictions)
    risk_recall = recall_score(y_test, predictions, pos_label="RISK")
    return accuracy, risk_recall


def main() -> None:
    _, X_test, _, y_test = prepare_train_test_data()
    predictions = predict_no_risk(X_test)
    accuracy, risk_recall = evaluate_baseline(y_test, predictions)

    print(f"Number of predictions: {len(predictions)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Recall RISK: {risk_recall:.2%}")


if __name__ == "__main__":
    main()
