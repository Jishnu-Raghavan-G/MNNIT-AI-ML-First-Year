# Module 02 — Data Preparation

## 1. Introduction

Data preparation is the process of preparing raw data so that it can be effectively used by machine learning algorithms.

A basic machine learning workflow is:

Raw Data
↓
Data Preparation
↓
Feature Selection
↓
Training
↓
Validation
↓
Testing
↓
Prediction

The quality of the data has a major effect on machine learning performance.

## 2. Features

Features are the input variables used by a machine learning model.

For the student performance dataset:

- `hours`
- `attendance`
- `assignments`

are features.

## 3. Target Variable

The target variable is the output that a supervised learning model tries to predict.

For the student performance dataset:

`score`

is the target.

Therefore:

Features → hours, attendance, assignments  
Target → score

## 4. Loading Data

Datasets can be stored in formats such as:

- CSV
- Excel
- JSON
- Databases

CSV files are commonly used because they are simple and easy to work with.

Pandas can be used to load CSV data:

```python
import pandas as pd

data = pd.read_csv("dataset.csv")
```

## 5. Inspecting Data

Before using a dataset, it should be inspected.

### First Rows

```python
data.head()
```

### Last Rows

```python
data.tail()
```

### Dataset Shape

```python
data.shape
```

This returns the number of rows and columns in the dataset.

### Dataset Information

```python
data.info()
```

This displays information such as:

- Column names
- Data types
- Number of non-null values
- Memory usage

### Statistical Summary

```python
data.describe()
```

This provides statistical information about numerical columns, including:

- Count
- Mean
- Standard deviation
- Minimum
- Maximum
- Quartiles

## 6. Missing Values

Real-world datasets can contain missing values.

A missing value is commonly represented as `NaN`.

Missing values can be checked using:

```python
data.isnull().sum()
```

This shows the number of missing values in each column.

Missing values can be handled in different ways depending on the dataset and problem.

### Removing Rows Containing Missing Values

```python
data.dropna()
```

### Filling Missing Values

A missing numerical value can sometimes be replaced using statistical values such as the mean or median.

For example:

```python
data["income"] = data["income"].fillna(
    data["income"].median()
)
```

The correct method depends on the meaning and distribution of the data.

Removing too many observations can result in information loss, while incorrectly filling missing values can introduce bias.

## 7. Duplicate Data

Datasets may contain duplicate observations.

Duplicate rows can cause repeated information to influence the analysis or model.

Duplicates can be detected using:

```python
data.duplicated()
```

The total number of duplicate rows can be found using:

```python
data.duplicated().sum()
```

Duplicate rows can be removed using:

```python
data.drop_duplicates()
```

Whether duplicates should be removed depends on the dataset because repeated observations are not always errors.

## 8. Categorical Data

Categorical data contains values representing categories rather than continuous numerical measurements.

Examples include:

- Red
- Blue
- Green
- Male
- Female
- Spam
- Ham

Many machine learning algorithms require numerical input.

Categorical variables may therefore need to be converted into numerical representations.

Common techniques include:

1. Label Encoding
2. One-Hot Encoding

### Label Encoding

Label encoding assigns a numerical value to each category.

For example:

Spam → 1  
Ham → 0

### One-Hot Encoding

One-hot encoding creates separate binary columns for categories.

For example, a `color` feature containing:

Red, Blue, Green

could become:

`color_red`  
`color_blue`  
`color_green`

Each column indicates whether an observation belongs to that category.

## 9. Feature Scaling

Features can have very different numerical ranges.

For example:

Age: 18–70  
Income: 20,000–200,000

Without scaling, a feature containing large numerical values can have a much greater influence on distance-based calculations.

Some machine learning algorithms are especially sensitive to differences in feature scale.

Examples include:

- K-Nearest Neighbors
- K-Means
- Support Vector Machines
- Algorithms using gradient-based optimization

Feature scaling transforms features into comparable numerical ranges.

Two common methods are:

1. Standardization
2. Min-Max Normalization

## 10. Standardization

Standardization transforms a feature so that it has a mean close to 0 and a standard deviation close to 1.

The formula is:

`z = (x - μ) / σ`

Where:

- `x` = original value
- `μ` = mean of the feature
- `σ` = standard deviation of the feature
- `z` = standardized value

Using scikit-learn:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Standardization does not necessarily restrict values to a particular interval.

Values can be positive or negative depending on whether they are above or below the mean.

## 11. Min-Max Normalization

Min-Max normalization commonly transforms values into the range `[0, 1]`.

The formula is:

`x' = (x - xmin) / (xmax - xmin)`

Where:

- `x` = original value
- `xmin` = minimum value of the feature
- `xmax` = maximum value of the feature
- `x'` = normalized value

Using scikit-learn:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
```

After Min-Max normalization, the smallest value normally becomes 0 and the largest becomes 1.

## 12. Separating Features and Target

In supervised learning, the dataset is generally separated into:

- Input features `X`
- Target variable `y`

For the student performance dataset:

```python
X = data[
    ["hours", "attendance", "assignments"]
]

y = data["score"]
```

Here:

`X` contains the information given to the model.

`y` contains the values that the model is expected to predict.

## 13. Train-Test Split

A dataset should generally be separated into training and testing data.

The process can be represented as:

Complete Dataset
↓
Train-Test Split
↓
Training Data + Testing Data

### Training Data

Training data is used by the machine learning algorithm to learn patterns and relationships.

### Testing Data

Testing data is kept separate during training.

After the model has been trained, testing data is used to evaluate how well the model performs on unseen observations.

A common split is:

80% → Training Data  
20% → Testing Data

Example:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Here:

- `X_train` contains training features.
- `X_test` contains testing features.
- `y_train` contains training target values.
- `y_test` contains testing target values.
- `test_size=0.2` means approximately 20% of the dataset is used for testing.
- The remaining approximately 80% is used for training.
- `random_state=42` makes the random split reproducible.

## 14. Training, Validation and Testing Data

In larger machine learning projects, data may be divided into three parts:

Training Data  
↓  
Used to train the model

Validation Data  
↓  
Used during model development and parameter selection

Testing Data  
↓  
Used for final evaluation

The test set should represent unseen data and should not be repeatedly used while designing the model.

## 15. Data Leakage

Data leakage occurs when information that should not be available during model training accidentally influences the training process.

This can produce misleadingly high model performance.

One important example involves feature scaling.

An incorrect approach is:

Complete Dataset  
↓  
Scale Complete Dataset  
↓  
Train-Test Split

In this situation, information from the test data can influence the scaling parameters.

A better process is:

Complete Dataset
↓
Train-Test Split
↓
Fit Scaler on Training Data
↓
Transform Training Data
↓
Transform Testing Data

For example:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

`fit_transform()` is used on the training data because the scaler learns the mean and standard deviation from the training set.

`transform()` is used on the test data so that the same learned transformation is applied without learning anything from the test set.

The test data should not influence the training process.

## 16. Student Performance Dataset

The repository contains:

`datasets/student_performance.csv`

The dataset contains the columns:

- `hours`
- `attendance`
- `assignments`
- `score`

For a regression problem:

Features:

- `hours`
- `attendance`
- `assignments`

Target:

- `score`

The objective is to use the input features to predict a student's score.

## 17. Customer Dataset

The repository contains:

`datasets/customers.csv`

The dataset contains:

- `age`
- `income`
- `spending_score`

This dataset does not contain a target variable.

It can therefore be used later for unsupervised learning and clustering.

For example, K-Means can group customers based on similarities in their features.

Because K-Means relies on distances between observations, feature scaling can be important.

Without scaling, a feature such as income could dominate another feature such as age simply because its numerical values are much larger.

## 18. Spam Messages Dataset

The repository also contains:

`datasets/spam_messages.csv`

It contains:

- `message`
- `label`

The message is the input data.

The label identifies whether the message belongs to a class such as:

- Spam
- Ham

Unlike numerical datasets, text cannot normally be passed directly to most machine learning algorithms.

The text must first be converted into numerical features.

This idea will be used in the Naive Bayes module.

## 19. Reproducibility

Machine learning experiments often involve random operations such as randomly splitting a dataset.

Using a fixed random state allows the same operation to produce the same result when the program is executed again.

For example:

```python
random_state=42
```

The value `42` itself is not special.

Any fixed integer can be used.

The important point is that using the same random state makes experiments easier to reproduce and compare.

## 20. Basic Data Preparation Pipeline

A typical preparation pipeline can be represented as:

Raw Dataset
↓
Load Dataset
↓
Inspect Data
↓
Understand Columns
↓
Check Missing Values
↓
Handle Missing Values
↓
Check Duplicate Observations
↓
Remove Invalid Duplicates
↓
Handle Categorical Features
↓
Separate Features and Target
↓
Split Training and Testing Data
↓
Fit Preprocessing on Training Data
↓
Transform Training Data
↓
Transform Testing Data
↓
Train Machine Learning Model

Not every dataset requires every step.

For example:

- A dataset with no missing values does not require missing-value treatment.
- A dataset containing only numerical variables does not require categorical encoding.
- Some algorithms are much more sensitive to feature scaling than others.

The preparation process should therefore depend on both the dataset and the machine learning algorithm.

## 21. Important Principles

### Understand the Data Before Training

A machine learning algorithm should not simply be applied immediately after loading a dataset.

The columns, data types, ranges and possible errors should first be understood.

### Keep Test Data Separate

Testing data represents unseen observations.

Allowing information from the test set to influence training can make evaluation unreliable.

### Apply the Same Transformation

If training data is standardized using a particular scaler, the same scaler must be used to transform the test data and future input data.

### Avoid Unnecessary Transformations

Data preparation should solve actual problems in the dataset.

Transformations should not be applied simply because they are available.

## 22. Key Points

- Data preparation converts raw data into a form suitable for machine learning.
- Features are the input variables used by a model.
- The target variable is the output that a supervised model attempts to predict.
- A dataset should be inspected before model training.
- Missing values should be identified and handled appropriately.
- Duplicate observations should be investigated.
- Categorical variables may require numerical encoding.
- Feature scaling places numerical features on more comparable scales.
- Standardization and Min-Max normalization are common scaling techniques.
- Supervised datasets are separated into features `X` and target `y`.
- Training data is used to learn the model.
- Testing data is used to evaluate performance on unseen data.
- Validation data can be used during model development.
- Preprocessing should be fitted using training data rather than test data.
- Data leakage can produce misleading evaluation results.
- A fixed random state helps make experiments reproducible.
- Data preparation should always be adapted to the dataset and the algorithm being used.
