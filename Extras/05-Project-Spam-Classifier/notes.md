# Spam Message Classifier

## 1. Project Overview

This project builds a machine learning classifier that predicts whether a text message is:

- Spam
- Ham

The project demonstrates a complete text-classification workflow using:

- Text preprocessing
- Feature extraction
- Naive Bayes
- Model evaluation

---

## 2. Dataset

The project uses:

    datasets/spam_messages.csv

Columns:

| Column | Description |
|---|---|
| message | Text message |
| label | spam or ham |

Target:

    label

Input:

    message

---

## 3. Why Naive Bayes?

Naive Bayes is well suited to many text-classification problems.

For a message X and class C:

    P(C | X) ∝ P(C) P(X | C)

The classifier estimates the probability of each class and chooses the most likely class.

For text classification, the message is converted into numerical features before training.

---

## 4. Text Feature Extraction

Machine learning algorithms cannot directly work with raw text.

A vectorizer converts text into numerical features.

This project uses TF-IDF.

TF-IDF represents the importance of words in documents.

Words that occur frequently in a particular document but are less common across the complete collection can receive higher importance.

---

## 5. Project Workflow

    Load Messages
        ↓
    Inspect Data
        ↓
    Separate Text and Labels
        ↓
    Train/Test Split
        ↓
    Convert Text to TF-IDF Features
        ↓
    Train Naive Bayes
        ↓
    Predict
        ↓
    Evaluate
        ↓
    Predict New Messages

---

## 6. Train/Test Split

The dataset is divided into:

### Training Set

Used to learn the relationship between message features and labels.

### Test Set

Used to evaluate the classifier on unseen messages.

The test set should not be used to train the model.

---

## 7. TF-IDF

TF-IDF stands for:

    Term Frequency - Inverse Document Frequency

It combines two ideas.

### Term Frequency

How frequently a term occurs in a document.

### Inverse Document Frequency

How informative a term is across the collection of documents.

A common form is:

    TF-IDF(t,d) = TF(t,d) × IDF(t)

The result is a numerical representation of text.

---

## 8. Multinomial Naive Bayes

Multinomial Naive Bayes is commonly used for discrete/count-like text features.

The model estimates class probabilities using the features extracted from messages.

The classifier then predicts the class with the highest estimated probability.

---

## 9. Evaluation

Important classification metrics include:

### Accuracy

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

### Precision

    Precision = TP / (TP + FP)

Precision answers:

"Of the messages predicted as spam, how many were actually spam?"

### Recall

    Recall = TP / (TP + FN)

Recall answers:

"Of all actual spam messages, how many did the model detect?"

### F1 Score

    F1 = 2 × Precision × Recall
             --------------------
             Precision + Recall

F1 balances precision and recall.

---

## 10. Confusion Matrix

For spam classification:

- True Positive → spam correctly classified as spam.
- True Negative → ham correctly classified as ham.
- False Positive → ham incorrectly classified as spam.
- False Negative → spam incorrectly classified as ham.

False positives can be inconvenient because legitimate messages may be classified as spam.

False negatives are also important because actual spam can reach the user.

---

## 11. Predicting New Messages

After training, a new message must go through the same vectorizer used during training.

Example:

    "Congratulations! You won a prize."

The trained model converts the message into TF-IDF features and predicts its class.

---

## 12. Important Principle

The test data must remain unseen during training.

The vectorizer should also be fitted only on the training text.

Correct workflow:

    Training text
        ↓
    Fit vectorizer
        ↓
    Transform training text
        ↓
    Transform test text using same vectorizer
        ↓
    Train classifier
        ↓
    Evaluate on test text

This prevents information from the test set from leaking into training.

---

## 13. Advantages

- Simple workflow.
- Fast to train.
- Effective for many text-classification tasks.
- Naive Bayes is computationally efficient.
- Easy to use as a baseline classifier.

---

## 14. Limitations

- The Naive Bayes independence assumption is simplified.
- Text preprocessing can strongly affect results.
- Vocabulary changes can affect performance.
- Real-world spam can evolve over time.
- Accuracy alone may not be sufficient for evaluating spam detection.

---

## 15. Learning Objectives

After completing this project, you should understand:

- How text classification works.
- Why text must be converted into numerical features.
- What TF-IDF represents.
- How Naive Bayes can classify text.
- How train/test splitting works.
- How to evaluate a spam classifier.
- How to classify new messages.

---

## 16. How to Run

From the repository root:

    python Extras/05-Project-Spam-Classifier/spam_classifier_project.py

Required libraries:

    numpy
    pandas
    scikit-learn
