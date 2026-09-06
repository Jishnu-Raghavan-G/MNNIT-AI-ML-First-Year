import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def load_dataset():
    """Load the spam-message dataset."""
    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "spam_messages.csv"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    return pd.read_csv(dataset_path)


def inspect_dataset(df):
    """Display basic information about the dataset."""
    print("\n=== DATASET ===")
    print(df)

    print("\n=== SHAPE ===")
    print(df.shape)

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    print("\n=== LABEL COUNTS ===")
    print(df["label"].value_counts())


def prepare_data(df):
    """Prepare text and binary labels."""
    text = df["message"].astype(str)

    labels = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # spam = 1, ham = 0
    y = labels.map({
        "ham": 0,
        "spam": 1
    })

    if y.isnull().any():
        unknown_labels = labels[y.isnull()].unique()

        raise ValueError(
            "Unknown labels found: "
            f"{unknown_labels}"
        )

    return text, y.to_numpy(dtype=int)


def evaluate_model(y_test, predictions):
    """Calculate and display classification metrics."""
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\n=== EVALUATION ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\n=== CONFUSION MATRIX ===")
    print(matrix)

    print("\n=== CLASSIFICATION REPORT ===")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["ham", "spam"],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix
    }


def predict_new_messages(model, vectorizer):
    """Predict classes for new messages."""
    messages = [
        "Hey, are we meeting at 5 today?",
        "Congratulations! You won a free prize. Claim now!",
        "Please submit the assignment before tomorrow.",
        "URGENT! You have been selected for a cash reward."
    ]

    X_new = vectorizer.transform(messages)

    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)

    print("\n=== NEW MESSAGE PREDICTIONS ===")

    for message, prediction, probability in zip(
        messages,
        predictions,
        probabilities
    ):
        label = "spam" if prediction == 1 else "ham"

        print("\nMessage:")
        print(message)

        print(f"Prediction: {label}")

        print(
            f"Probability of ham:  "
            f"{probability[0]:.4f}"
        )

        print(
            f"Probability of spam: "
            f"{probability[1]:.4f}"
        )


def main():
    print("SPAM MESSAGE CLASSIFIER")

    # 1. Load data
    df = load_dataset()

    # 2. Inspect data
    inspect_dataset(df)

    # 3. Prepare text and labels
    text, y = prepare_data(df)

    # 4. Split the dataset
    X_train_text, X_test_text, y_train, y_test = (
        train_test_split(
            text,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print("\n=== DATA SPLIT ===")
    print(f"Training messages: {len(X_train_text)}")
    print(f"Testing messages:  {len(X_test_text)}")

    # 5. Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )

    # 6. Fit only on training text
    X_train = vectorizer.fit_transform(
        X_train_text
    )

    # 7. Transform test text using the same vectorizer
    X_test = vectorizer.transform(
        X_test_text
    )

    print("\n=== TEXT FEATURES ===")
    print(
        f"Training matrix shape: {X_train.shape}"
    )
    print(
        f"Testing matrix shape:  {X_test.shape}"
    )
    print(
        f"Vocabulary size: "
        f"{len(vectorizer.vocabulary_)}"
    )

    # 8. Create Naive Bayes model
    model = MultinomialNB()

    # 9. Train model
    model.fit(
        X_train,
        y_train
    )

    # 10. Predict test data
    predictions = model.predict(
        X_test
    )

    # 11. Evaluate model
    evaluate_model(
        y_test,
        predictions
    )

    # 12. Predict new messages
    predict_new_messages(
        model,
        vectorizer
    )


if __name__ == "__main__":
    main()
