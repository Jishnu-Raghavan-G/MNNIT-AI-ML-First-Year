# Student Performance Prediction

## 1. Project Overview

This project builds a complete machine learning workflow to predict a student's score from:

- Study hours
- Attendance
- Assignment performance

The project uses Linear Regression because the target variable, score, is continuous.

---

## 2. Dataset

The project uses:

    datasets/student_performance.csv

Columns:

| Column | Description |
|---|---|
| hours | Study hours |
| attendance | Attendance percentage |
| assignments | Assignment-related score/value |
| score | Student's final score |

Target:

    score

Features:

    hours
    attendance
    assignments

---

## 3. Project Workflow

The project follows:

    Load Data
        ↓
    Inspect Data
        ↓
    Separate Features and Target
        ↓
    Train/Test Split
        ↓
    Train Linear Regression
        ↓
    Predict Scores
        ↓
    Evaluate Model
        ↓
    Predict New Student

---

## 4. Why Linear Regression?

Linear Regression predicts a continuous numerical value.

Student score is continuous, so the model can learn a relationship such as:

    score = β₀
          + β₁(hours)
          + β₂(attendance)
          + β₃(assignments)

The coefficients are learned from the training data.

---

## 5. Evaluation Metrics

The project uses:

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted scores.

    MAE = mean(|actual - predicted|)

### MSE

Mean Squared Error squares the errors before averaging.

    MSE = mean((actual - predicted)²)

### RMSE

    RMSE = √MSE

RMSE has the same units as the score.

---

## 6. Example Prediction

A new student can be represented as:

    hours = 6
    attendance = 90
    assignments = 9

The trained model uses these features to estimate the student's score.

---

## 7. Learning Objectives

After completing this project, you should understand:

- How to load a dataset.
- How to select features and target.
- How to split data.
- How Linear Regression works.
- How predictions are generated.
- How regression metrics are calculated.
- How a trained model can be used for new predictions.

---

## 8. How to Run

From the repository root:

    python Extras/03-Project-Student-Performance/student_performance_project.py

Required libraries:

    numpy
    pandas
    scikit-learn
