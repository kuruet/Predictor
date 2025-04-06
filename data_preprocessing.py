import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset/heart.csv")
print(df["target"].value_counts())

# One-hot encoding for categorical features
df = pd.get_dummies(df, columns=["cp", "thal", "slope"], drop_first=True)

# Standardizing numerical features
scaler = StandardScaler()
columns_to_scale = ["age", "trestbps", "chol", "thalach", "oldpeak"]
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Splitting data
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model with better parameters
log_reg_model = LogisticRegression(C=0.1, solver='liblinear', max_iter=1000, random_state=42)
log_reg_model.fit(X_train, y_train)

# Save model & preprocessing steps
model_data = {
    "model": log_reg_model,
    "features": X_train.columns.tolist(),  # Store correct feature names
    "scaler": scaler
}

with open("heart_disease_model.pkl", "wb") as model_file:
    pickle.dump(model_data, model_file)

print("\u2705 Model retrained and saved successfully!")
