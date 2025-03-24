from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import shap

app = Flask(__name__)

# Load model and encoders
model = joblib.load("random_forest_best_model.pkl")
binary_encodings = joblib.load("binary_encodings.pkl")
label_encodings = joblib.load("label_encodings.pkl")
best_features = joblib.load("selected_features.pkl")
mae = joblib.load("final_mae.pkl")  # Load MAE
adjusted_r2 = joblib.load("final_adjusted_r2.pkl")  # Load Adjusted R²
X_train = pd.read_csv("X_train.csv").drop(columns=["Unnamed: 0"], errors="ignore")

# Load actual vs. predicted values
actual_vs_predicted = joblib.load("actual_vs_predicted.pkl")

# Convert both key and value to float
actual_vs_predicted = {float(key): float(value) for key, value in actual_vs_predicted.items()}

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical features
        for col, mapping in binary_encodings.items():
            if col in input_df:
                input_df[col] = input_df[col].map(mapping).fillna(0)
        
        for col, mapping in label_encodings.items():
            if col in input_df:
                input_df[col] = input_df[col].astype("category").cat.set_categories(mapping.values()).cat.codes.fillna(-1)
        
        # Ensure feature alignment
        input_df = input_df.reindex(columns=best_features, fill_value=0)
        
        # Predict claim cost
        prediction = model.predict(input_df)[0]
        
        # Compute SHAP values
        shap_values = explainer.shap_values(input_df)
        
        # Extract top 5 influential features
        shap_list = sorted(
            [(feature, input_data.get(feature, None), float(value)) for feature, value in zip(best_features, shap_values[0])],
            key=lambda x: abs(x[2]), reverse=True
        )[:5]
        
        # Response with all actual vs predicted values
        response = {
            "prediction": float(prediction),
            "mae": float(mae),  # Include MAE
            "adjusted_r2": float(adjusted_r2),  # Include Adjusted R²
            "top_5_shap_values": shap_list,
            "all_actual_vs_predicted": actual_vs_predicted  # Include all actual vs predicted values
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
