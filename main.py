"""
DecodeLabs Project 2: Data Classification Using AI
Dataset: Iris (built into scikit-learn)
Algorithm: Decision Tree Classifier

Run:
    python main.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Allows plots to be saved without opening a GUI.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 62)
    print("DECODELABS PROJECT 2: AI FLOWER CLASSIFICATION")
    print("=" * 62)

    # 1. Load and understand the dataset.
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    df = iris.frame.copy()
    df["species"] = df["target"].map(
        {index: name for index, name in enumerate(iris.target_names)}
    )
    df = df.drop(columns=["target"])

    print("\n1. DATASET OVERVIEW")
    print("-" * 62)
    print(f"Number of rows: {len(df)}")
    print(f"Number of input features: {X.shape[1]}")
    print(f"Target classes: {', '.join(iris.target_names)}")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print("\nClass counts:")
    print(df["species"].value_counts().to_string())

    # Save a readable copy of the dataset.
    df.to_csv(OUTPUT_DIR / "iris_dataset.csv", index=False)

    # 2. Split into training and testing data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print("\n2. TRAIN / TEST SPLIT")
    print("-" * 62)
    print(f"Training examples: {len(X_train)} (80%)")
    print(f"Testing examples:  {len(X_test)} (20%)")

    # 3. Train the classification model.
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    print("\n3. MODEL TRAINING")
    print("-" * 62)
    print("Decision Tree classifier trained successfully.")

    # 4. Evaluate the model on unseen test data.
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred)

    print("\n4. MODEL EVALUATION")
    print("-" * 62)
    print(f"Test accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(matrix)

    # Save the evaluation report.
    report_text = (
        "DecodeLabs Project 2 - Iris Classification Results\n"
        f"Algorithm: Decision Tree (max_depth=4)\n"
        f"Training rows: {len(X_train)}\n"
        f"Testing rows: {len(X_test)}\n"
        f"Test accuracy: {accuracy:.2%}\n\n"
        f"Classification report:\n{report}\n"
        f"Confusion matrix:\n{matrix}\n"
    )
    (OUTPUT_DIR / "evaluation_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    # Save confusion matrix visualization.
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=iris.target_names,
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Iris Classification - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # Save feature importance visualization.
    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values()
    ax = importances.plot(kind="barh", figsize=(8, 4), title="Feature Importance")
    ax.set_xlabel("Importance score")
    ax.set_ylabel("Input feature")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
    plt.close()

    # 5. Demonstrate prediction for a new flower.
    # Values are in centimeters: sepal length, sepal width,
    # petal length, petal width.
    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=X.columns,
    )
    predicted_class = int(model.predict(sample)[0])
    predicted_species = iris.target_names[predicted_class]

    print("\n5. NEW FLOWER PREDICTION")
    print("-" * 62)
    print("Sample measurements (cm):")
    print(sample.to_string(index=False))
    print(f"Predicted species: {predicted_species}")

    print("\nOUTPUT FILES")
    print("-" * 62)
    print(f"Dataset CSV:       {OUTPUT_DIR / 'iris_dataset.csv'}")
    print(f"Evaluation report: {OUTPUT_DIR / 'evaluation_report.txt'}")
    print(f"Confusion matrix:  {OUTPUT_DIR / 'confusion_matrix.png'}")
    print(f"Feature importance:{OUTPUT_DIR / 'feature_importance.png'}")
    print("\nProject completed successfully.")


if __name__ == "__main__":
    main()
