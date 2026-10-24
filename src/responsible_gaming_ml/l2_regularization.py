"""Compare L2 regularization on synthetic training and validation data only."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from responsible_gaming_ml.data_preparation import prepare_train_validation_test_data
from responsible_gaming_ml.logistic_regression import evaluate_logistic_regression


def compare_l2_regularization(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
) -> tuple[pd.DataFrame, dict[float, Pipeline]]:
    """Fit five pipelines on the same train set and report RISK metrics."""
    rows = []
    pipelines = {}
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=C)),
        ])
        pipeline.fit(X_train, y_train)
        train_predictions = pipeline.predict(X_train)
        validation_predictions = pipeline.predict(X_validation)
        _, _, _, train_f1, _ = evaluate_logistic_regression(y_train, train_predictions)
        _, precision, recall, validation_f1, _ = evaluate_logistic_regression(
            y_validation, validation_predictions
        )
        rows.append({
            "C": C,
            "F1 RISK Train": train_f1,
            "F1 RISK Validation": validation_f1,
            "Train - Validation": train_f1 - validation_f1,
            "Precision RISK Validation": precision,
            "Recall RISK Validation": recall,
            "RISK predictions Train": int((train_predictions == "RISK").sum()),
            "RISK predictions Validation": int((validation_predictions == "RISK").sum()),
        })
        pipelines[C] = pipeline
    return pd.DataFrame(rows), pipelines


def select_best_l2_C(results: pd.DataFrame) -> float:
    """Maximize unrounded validation F1; break exact ties with the smallest C."""
    ranked = results.sort_values(
        ["F1 RISK Validation", "C"], ascending=[False, True]
    )
    return float(ranked.iloc[0]["C"])


def main() -> None:
    X_train, X_validation, _, y_train, y_validation, _ = (
        prepare_train_validation_test_data()
    )
    results, _ = compare_l2_regularization(
        X_train, X_validation, y_train, y_validation
    )
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSelected C: {select_best_l2_C(results):g}")
    print("Selection: highest validation F1; smallest C in an exact tie.")
    print("A train/validation gap alone is not proof of overfitting.")
    print("Synthetic educational experiment; no test metrics computed.")


if __name__ == "__main__":
    main()
