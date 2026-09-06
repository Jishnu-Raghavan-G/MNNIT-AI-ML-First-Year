from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Dataset Path
# ---------------------------------------------------------

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "student_performance.csv"
)


# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

data = pd.read_csv(DATA_PATH)

print("=" * 60)
print("STUDENT PERFORMANCE DATASET")
print("=" * 60)

print("\nFirst five rows:")
print(data.head())

print("\nLast five rows:")
print(data.tail())


# ---------------------------------------------------------
# 2. Inspect Dataset
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nDetailed information:")
data.info()

print("\nStatistical summary:")
print(data.describe())


# ---------------------------------------------------------
# 3. Check Missing Values
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(data.isnull().sum())

total_missing = data.isnull().sum().sum()

print("\nTotal missing values:", total_missing)


# ---------------------------------------------------------
# 4. Handle Missing Values
# ---------------------------------------------------------

# For this dataset, missing numerical values can be filled
# using the median if any are present.

numeric_columns = [
    "hours",
    "attendance",
    "assignments",
    "score",
]

for column in numeric_columns:
    if data[column].isnull().any():
        median_value = data[column].median()
        data[column] = data[column].fillna(median_value)

print("\nMissing values after handling:")
print(data.isnull().sum())


# ---------------------------------------------------------
# 5. Check Duplicate Rows
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE DATA")
print("=" * 60)

duplicate_count = data.duplicated().sum()

print("\nNumber of duplicate rows:")
print(duplicate_count)


# ---------------------------------------------------------
# 6. Remove Duplicate Rows
# ---------------------------------------------------------

data = data.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(data.shape)


# ---------------------------------------------------------
# 7. Separate Features and Target
# ---------------------------------------------------------

features = [
    "hours",
    "attendance",
    "assignments",
]

target = "score"

X = data[features]
y = data[target]

print("\n" + "=" * 60)
print("FEATURES")
print("=" * 60)

print(X.head())

print("\n" + "=" * 60)
print("TARGET")
print("=" * 60)

print(y.head())


# ---------------------------------------------------------
# 8. Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print("\nTotal samples:", len(data))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining features:")
print(X_train)

print("\nTesting features:")
print(X_test)

print("\nTraining target:")
print(y_train)

print("\nTesting target:")
print(y_test)


# ---------------------------------------------------------
# 9. Standardize Features
# ---------------------------------------------------------

scaler = StandardScaler()

# Learn scaling parameters ONLY from training data.
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to testing data.
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 10. Convert Scaled Arrays Back to DataFrames
# ---------------------------------------------------------

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=features,
    index=X_train.index,
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=features,
    index=X_test.index,
)


# ---------------------------------------------------------
# 11. Display Scaling Parameters
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SCALING PARAMETERS")
print("=" * 60)

print("\nMean learned from training data:")

for feature, mean in zip(features, scaler.mean_):
    print(f"{feature}: {mean:.4f}")

print("\nStandard deviation learned from training data:")

for feature, scale in zip(features, scaler.scale_):
    print(f"{feature}: {scale:.4f}")


# ---------------------------------------------------------
# 12. Display Scaled Data
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SCALED TRAINING DATA")
print("=" * 60)

print(X_train_scaled)

print("\n" + "=" * 60)
print("SCALED TESTING DATA")
print("=" * 60)

print(X_test_scaled)


# ---------------------------------------------------------
# 13. Compare Original and Scaled Data
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ORIGINAL VS SCALED TRAINING DATA")
print("=" * 60)

print("\nOriginal:")
print(X_train.head())

print("\nScaled:")
print(X_train_scaled.head())


# ---------------------------------------------------------
# 14. Verify Standardization
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SCALED TRAINING DATA STATISTICS")
print("=" * 60)

print("\nMean after scaling:")
print(X_train_scaled.mean())

print("\nStandard deviation after scaling:")
print(X_train_scaled.std(ddof=0))


# ---------------------------------------------------------
# 15. Final Summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)

print(
    """
Completed steps:

1. Loaded the dataset
2. Inspected the dataset
3. Checked missing values
4. Handled missing values if present
5. Checked duplicate rows
6. Removed duplicate rows
7. Separated features and target
8. Split data into training and testing sets
9. Fitted StandardScaler only on training data
10. Transformed training and testing features
11. Verified the scaled training data

The data is now ready for use by a machine learning model.
"""
)
