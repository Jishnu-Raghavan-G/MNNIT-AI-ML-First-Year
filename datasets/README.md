# Datasets

This folder contains small datasets used throughout the MNNIT AI/ML First-Year repository.

The datasets are designed for learning and implementing different machine learning algorithms.

## Datasets

### 1. student_performance.csv

Contains student-related features and their final scores.

Columns:

- `hours` — Study hours
- `attendance` — Attendance percentage
- `assignments` — Number of assignments completed
- `score` — Final score

This dataset can be used for:

- Regression
- Linear Regression
- Model evaluation
- Train/test splitting
- Studying relationships between features and target values

The target variable is:

`score`

---

### 2. customers.csv

Contains basic customer information and spending behavior.

Columns:

- `age` — Customer age
- `income` — Income
- `spending_score` — Spending score

This dataset can be used for:

- Unsupervised learning
- K-Means clustering
- Distance calculations
- Elbow method
- Silhouette score

There is no predefined target variable because the dataset can be used to discover groups of similar customers.

---

### 3. spam_messages.csv

Contains example text messages labelled as either spam or legitimate messages.

Columns:

- `message` — Text message
- `label` — Message category

Labels:

- `spam` — Unwanted/promotional message
- `ham` — Normal/legitimate message

This dataset can be used for:

- Classification
- Naive Bayes
- Text classification
- Feature extraction
- Model evaluation

The target variable is:

`label`

## Purpose

These datasets are intentionally small and easy to understand so that beginners can focus on learning the machine learning algorithms and implementation rather than dealing with complicated real-world datasets.
