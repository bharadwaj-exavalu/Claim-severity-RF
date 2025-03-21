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
X_train = pd.read_csv("X_train.csv").drop(columns=["Unnamed: 0"], errors="ignore")

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
        
        response = {
            "prediction": float(prediction),
            "mae": float(mae),  # Include MAE in response
            "top_5_shap_values": shap_list
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
