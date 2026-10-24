from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from responsible_gaming_ml.synthetic_data import generate_dataset

CSV_PATH = Path(__file__).resolve().parent / "data" / "players_features.csv"
FEATURE_LABELS = {
    "deposits_7d": "Total deposit amount over 7 days",
    "losses_7d": "Total loss amount over 7 days",
    "night_sessions": "Number of night sessions",
    "session_hours": "Session duration (hours)",
    "deposit_frequency": "Number of deposits",
}
INTEGER_FEATURES = {"night_sessions", "deposit_frequency"}


def analyze_target_distribution() -> None:
    dataset = generate_dataset()
    target = dataset["target"]
    target_counts = target.value_counts()
    target_proportions = target.value_counts(normalize=True)

    print("Target counts:")
    print(target_counts.to_string())
    print("\nTarget proportions:")
    print(target_proportions.to_string())


def main() -> None:
    analyze_target_distribution()
    df = pd.read_csv(CSV_PATH)

    for feature, label in FEATURE_LABELS.items():
        _, ax = plt.subplots()
        if feature in INTEGER_FEATURES:
            minimum = int(df[feature].min())
            maximum = int(df[feature].max())
            bins = [value - 0.5 for value in range(minimum, maximum + 2)]
            ax.hist(df[feature], bins=bins)
        else:
            ax.hist(df[feature])
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(label)
        ax.set_ylabel("Number of players")

    scatter_pairs = [
        ("deposits_7d", "losses_7d"),
        ("session_hours", "night_sessions"),
        ("session_hours", "losses_7d"),
    ]
    for x_feature, y_feature in scatter_pairs:
        _, ax = plt.subplots()
        ax.scatter(df[x_feature], df[y_feature], s=10, alpha=0.3)
        ax.set_title(
            f"{FEATURE_LABELS[y_feature]} vs {FEATURE_LABELS[x_feature]}"
        )
        ax.set_xlabel(FEATURE_LABELS[x_feature])
        ax.set_ylabel(FEATURE_LABELS[y_feature])

    plt.show()


if __name__ == "__main__":
    main()
