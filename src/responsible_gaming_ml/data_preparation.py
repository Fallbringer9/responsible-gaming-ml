"""Prepare a train/test split of the educational synthetic dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split

from responsible_gaming_ml.synthetic_data import generate_dataset


def prepare_train_test_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    dataset = generate_dataset()
    X = dataset[
        [
            "deposits_7d",
            "losses_7d",
            "night_sessions",
            "session_hours",
            "deposit_frequency",
        ]
    ]
    y = dataset["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def prepare_train_validation_test_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
]:
    """Return a stratified 60/20/20 split, keeping the previously viewed test set."""
    X_train_full, X_test, y_train_full, y_test = prepare_train_test_data()
    # Validation takes 25% of the remaining 80%, i.e. 20% of all observations.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=42,
        stratify=y_train_full,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def demonstrate_train_validation_test_split() -> None:
    """Display subset sizes and class distributions without training a model."""
    X_train, X_validation, X_test, y_train, y_validation, y_test = (
        prepare_train_validation_test_data()
    )
    total = len(X_train) + len(X_validation) + len(X_test)
    for name, X, y in [
        ("Train", X_train, y_train),
        ("Validation", X_validation, y_validation),
        ("Test", X_test, y_test),
    ]:
        print(f"\n{name}: {len(X)} observations ({len(X) / total:.0%}), {X.shape[1]} features")
        print(pd.DataFrame({
            "count": y.value_counts(),
            "proportion": y.value_counts(normalize=True),
        }).to_string())
    print("\nThe test set was already viewed; this split does not restore independence.")
    print("Synthetic educational data only; not for real Responsible Gaming decisions.")


def main() -> None:
    X_train, X_test, y_train, y_test = prepare_train_test_data()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\nTraining target counts:")
    print(y_train.value_counts().to_string())
    print("\nTraining target proportions:")
    print(y_train.value_counts(normalize=True).to_string())

    print("\nTest target counts:")
    print(y_test.value_counts().to_string())
    print("\nTest target proportions:")
    print(y_test.value_counts(normalize=True).to_string())


if __name__ == "__main__":
    main()
