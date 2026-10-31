"""
IR / Text Classification Project
Compares KNN, Multinomial Naive Bayes, and Logistic Regression
on four categories from the 20 Newsgroups dataset.
"""

import os
import time
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Four subject areas: computer graphics, baseball, medicine, and politics.
CATEGORIES = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "talk.politics.misc",
]

def load_dataset():
    print("Loading 20 Newsgroups dataset (first run may download it)...")
    data = fetch_20newsgroups(
        subset="all",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    return data

def main():
    os.makedirs("results", exist_ok=True)
    dataset = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.25,
        random_state=42,
        stratify=dataset.target,
    )

    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    rows = []
    labels = [dataset.target_names[i] for i in range(len(dataset.target_names))]

    print(f"\nDataset: {len(dataset.data)} documents")
    print(f"Categories: {', '.join(dataset.target_names)}")
    print(f"Training documents: {len(X_train)} | Testing documents: {len(X_test)}\n")

    for name, classifier in models.items():
        print(f"Training {name}...")
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                max_features=20000,
                ngram_range=(1, 2),
                min_df=2,
            )),
            ("classifier", classifier),
        ])

        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start

        start = time.perf_counter()
        predictions = model.predict(X_test)
        prediction_time = time.perf_counter() - start

        metrics = {
            "Algorithm": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "Training Time (seconds)": training_time,
            "Prediction Time (seconds)": prediction_time,
        }
        rows.append(metrics)

        safe_name = name.lower().replace(" ", "_").replace("-", "")
        report = classification_report(
            y_test, predictions, target_names=dataset.target_names, zero_division=0
        )
        with open(f"results/{safe_name}_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        cm = confusion_matrix(y_test, predictions)
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(9, 7))
        display.plot(ax=ax, xticks_rotation=25, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix - {name}")
        fig.tight_layout()
        fig.savefig(f"results/{safe_name}_confusion_matrix.png", dpi=160)
        plt.close(fig)

        print(f"  Accuracy: {metrics['Accuracy']:.4f} | F1: {metrics['F1 Score']:.4f}")

    results = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
    results.insert(0, "Rank", range(1, len(results) + 1))
    results.to_csv("results/model_comparison.csv", index=False)

    metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score"]
    ax = results.set_index("Algorithm")[metric_columns].plot(
        kind="bar", figsize=(11, 6), ylim=(0, 1), rot=15
    )
    ax.set_title("Comparison of Text Classification Algorithms")
    ax.set_ylabel("Score")
    ax.set_xlabel("Algorithm")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results/algorithm_comparison.png", dpi=160)
    plt.close()

    with open("results/dataset_summary.txt", "w", encoding="utf-8") as f:
        f.write("Dataset: 20 Newsgroups\n")
        f.write("Selected categories:\n")
        for category in dataset.target_names:
            f.write(f"- {category}\n")
        f.write(f"\nTotal documents: {len(dataset.data)}\n")
        f.write(f"Training documents: {len(X_train)}\n")
        f.write(f"Testing documents: {len(X_test)}\n")
        f.write("\nThe dataset is a collection of discussion-group messages grouped by subject.\n")

    print("\n=== FINAL RANKING (based on weighted F1-score) ===")
    print(results[["Rank", "Algorithm", "Accuracy", "F1 Score"]].to_string(index=False))
    print("\nAll result files have been saved in the 'results' folder.")

if __name__ == "__main__":
    main()
