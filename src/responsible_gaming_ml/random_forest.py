"""Compare random forests and logistic regression on synthetic validation data."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from responsible_gaming_ml.data_preparation import prepare_train_validation_test_data
from responsible_gaming_ml.logistic_regression import evaluate_logistic_regression


def compare_random_forest_and_logistic_regression(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
) -> tuple[pd.DataFrame, dict[str, RandomForestClassifier | Pipeline]]:
    """Train four models on identical training data, without receiving test data."""
    models: dict[str, RandomForestClassifier | Pipeline] = {
        "Logistic Regression C=100": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=100)),
        ]),
    }
    for depth in [3, 5, None]:
        models[f"Random Forest depth={depth}"] = RandomForestClassifier(
            n_estimators=100, max_depth=depth, random_state=42
        )

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_predictions = model.predict(X_train)
        validation_predictions = model.predict(X_validation)
        _, _, _, train_f1, _ = evaluate_logistic_regression(y_train, train_predictions)
        _, precision, recall, validation_f1, _ = evaluate_logistic_regression(
            y_validation, validation_predictions
        )
        depth = (
            str(model.max_depth) if isinstance(model, RandomForestClassifier) else "N/A"
        )
        rows.append({
            "Model": name,
            "max_depth": depth,
            "F1 RISK Train": train_f1,
            "F1 RISK Validation": validation_f1,
            "Precision RISK Validation": precision,
            "Recall RISK Validation": recall,
            "RISK predictions Validation": int((validation_predictions == "RISK").sum()),
        })
    return pd.DataFrame(rows), models


def select_best_validation_model(results: pd.DataFrame) -> str:
    """Maximize validation F1; exact ties favor LR, then RF depths 3, 5, None."""
    priority = [
        "Logistic Regression C=100",
        "Random Forest depth=3",
        "Random Forest depth=5",
        "Random Forest depth=None",
    ]
    best_f1 = results["F1 RISK Validation"].max()
    tied_names = results.loc[results["F1 RISK Validation"] == best_f1, "Model"]
    return min(tied_names, key=priority.index)


def main() -> None:
    X_train, X_validation, _, y_train, y_validation, _ = (
        prepare_train_validation_test_data()
    )
    results, _ = compare_random_forest_and_logistic_regression(
        X_train, X_validation, y_train, y_validation
    )
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nHighest validation F1: {select_best_validation_model(results)}")
    print("Exact tie priority: logistic regression, then forest depths 3, 5, None.")
    print("Training F1 is measured on data used for fitting; validation F1 is not.")
    print("A train/validation gap alone does not prove overfitting.")
    print("Results describe this synthetic split only, not real-world performance.")
    print("No test metrics computed; not for real Responsible Gaming decisions.")


if __name__ == "__main__":
    main()
