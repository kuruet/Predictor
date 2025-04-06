import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("dataset/heart.csv")
print("Dataset loaded successfully.")

# Strip column names to remove any unwanted spaces
df.columns = df.columns.str.strip()

# Verify if 'target' column exists
if "target" not in df.columns:
    raise ValueError("Error: 'target' column missing from dataset. Available columns: " + str(df.columns))

# One-hot encoding categorical columns (cp, thal, slope)
df = pd.get_dummies(df, columns=["cp", "thal", "slope"], drop_first=True)

# Splitting features and target
X = df.drop("target", axis=1)
y = df["target"]

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling numerical columns
scaler = StandardScaler()
numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
train_accuracy = model.score(X_train, y_train) * 100
test_accuracy = model.score(X_test, y_test) * 100
print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")

# Save Model and Scaler
model_data = {
    "model": model,
    "scaler": scaler,
    "features": X_train.columns.tolist()
}

with open("heart_disease_model.pkl", "wb") as model_file:
    pickle.dump(model_data, model_file)

print("Model saved successfully as 'heart_disease_model.pkl'.")
