# Module 01 — Introduction to Machine Learning

## What is Machine Learning?

Machine Learning (ML) is a subfield of Artificial Intelligence (AI) that focuses on developing algorithms and statistical models that enable computers to perform tasks without explicit programming instructions.

Machine learning algorithms learn from data to identify patterns, extract insights, make predictions, and make decisions.

---

## Types of Machine Learning

There are three main types of machine learning:

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

---

## Supervised Learning

Supervised learning uses labeled data.

The dataset contains:

Features + Label

The model learns the relationship between the input features and the known output.

### Classification

Classification assigns input data to a discrete class.

Examples:

- Spam classification
- Offensive content classification
- Image classification

Types of classification:

- Binary classification
- Multi-class classification
- Multi-label classification

### Regression

Regression predicts a continuous numerical value.

Examples:

- House price prediction
- Employee salary prediction
- Student score prediction

---

## Unsupervised Learning

Unsupervised learning works with unlabeled data.

The goal is to identify hidden patterns or structures within the data.

Common tasks include:

- Clustering
- Dimensionality reduction

### Clustering

Clustering groups data points based on similarity.

Example:

Customer data → Clustering → Customer groups

K-Means is an important clustering algorithm covered later in this repository.

---

## Reinforcement Learning

Reinforcement learning involves an agent interacting with an environment.

The agent:

1. Observes the environment.
2. Takes an action.
3. Receives a reward or penalty.
4. Learns from the feedback.
5. Attempts to maximize cumulative reward.

---

## Features and Labels

### Feature

A feature is an input variable used by a machine learning model.

For example, in student performance prediction:

- Study hours
- Attendance
- Assignments completed

can be features.

### Label

A label is the known output associated with an example in supervised learning.

For example:

Features → Score

Here, `score` is the target or label.

---

## Training, Validation and Testing

A dataset can be divided into:

- Training set
- Validation set
- Test set

### Training Set

The training set is used to train the model.

The model learns patterns and relationships between the input features and target variable.

### Validation Set

The validation set is used during model development.

It helps evaluate generalization and identify possible overfitting or underfitting.

### Test Set

The test set is used for final evaluation.

It contains data that was not used to train the model.

---

## General Machine Learning Workflow

Data
↓
Data Preparation
↓
Feature Selection
↓
Train/Test Split
↓
Model Selection
↓
Training
↓
Validation
↓
Evaluation
↓
Prediction

---

## Algorithms Introduced

### Classification

- Naive Bayes
- Decision Tree
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Logistic Regression

### Unsupervised Learning

- K-Means Clustering

Other topics covered in this repository include:

- Regression
- Model evaluation
- Overfitting
- Underfitting
- Bias and variance
- Regularization
- Neural network basics

---

## Key Terms

| Term | Meaning |
|---|---|
| Artificial Intelligence | Field concerned with creating systems capable of performing tasks associated with intelligence |
| Machine Learning | Learning patterns from data to make predictions or decisions |
| Feature | Input variable used by a model |
| Label | Known target/output |
| Model | Learned representation of patterns in data |
| Classification | Predicting a discrete class |
| Regression | Predicting a continuous value |
| Clustering | Grouping similar data points |
| Training | Learning from training data |
| Validation | Evaluating a model during development |
| Testing | Final evaluation on unseen data |
| Reinforcement Learning | Learning through interaction using rewards and penalties |
