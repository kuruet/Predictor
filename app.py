from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and preprocessor
with open("heart_disease_model.pkl", "rb") as model_file:
    model_data = pickle.load(model_file)

model = model_data["model"]
scaler = model_data["scaler"]
feature_names = model_data["features"]  # Ensuring correct feature order

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form values
        form_data = request.form.to_dict()

        # Numeric fields
        numeric_fields = ["age", "sex", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "ca"]
        for field in numeric_fields:
            form_data[field] = float(form_data[field])

        # Categorical values
        cp_value = int(form_data.pop("cp"))
        thal_value = int(form_data.pop("thal"))
        slope_value = int(form_data.pop("slope"))

        # One-hot encoding (Ensuring feature names match trained model)
        for i in range(1, 4):  # Adjusted range based on training data
            form_data[f"cp_{i}"] = 1 if cp_value == i else 0
            form_data[f"thal_{i}"] = 1 if thal_value == i else 0
            form_data[f"slope_{i}"] = 1 if slope_value == i else 0

        # Convert to DataFrame
        input_df = pd.DataFrame([form_data])

        # Ensure input matches trained feature names
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0  # Set missing columns to 0

        # Apply scaling
        columns_to_scale = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])

        # Ensure correct column order
        input_df = input_df[feature_names]

        # Make prediction
        prediction = model.predict(input_df)[0]
        result = "No Heart Disease 😊" if prediction == 1 else "Heart Disease Detected 😢"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
