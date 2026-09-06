import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "student_performance.csv"

data = pd.read_csv(DATA_PATH)

print("Dataset:")
print(data.head())


# Select features and target

features = ["hours", "attendance", "assignments"]
target = "score"

X = data[features]
y = data[target]


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model

model = LinearRegression()


# Train model

model.fit(X_train, y_train)


# Make predictions

predictions = model.predict(X_test)


# Evaluate model

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.4f}")


# Display actual and predicted values

results = pd.DataFrame({
    "Actual Score": y_test.values,
    "Predicted Score": predictions
})

print("\nActual vs Predicted:")
print(results.to_string(index=False))


# Predict score for a new student

new_student = pd.DataFrame({
    "hours": [7.0],
    "attendance": [88.0],
    "assignments": [8]
})

predicted_score = model.predict(new_student)[0]

print("\nPrediction for New Student:")
print(f"Predicted Score: {predicted_score:.2f}")
