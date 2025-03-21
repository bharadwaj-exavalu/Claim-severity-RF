from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import shap

app = Flask(__name__)

# Load saved model, scaler, and evaluation metrics
model = joblib.load("random_forest_best_model.pkl")
scaler = joblib.load("scaler.pkl")
r2 = joblib.load("r2_score.pkl")
mae = joblib.load("mae.pkl")

# Load feature encoders if they exist
try:
    label_encoders = joblib.load("label_encoders.pkl")
    onehot_encoder = joblib.load("onehot_encoder.pkl")
except FileNotFoundError:
    label_encoders, onehot_encoder = None, None
    print("⚠️ Warning: No categorical encoders found! Ensure categorical features are handled correctly.")

# Load selected features (ensures feature order matches training)
selected_features = joblib.load("selected_features.pkl")

# Load training data for SHAP (to use as background data)
train_df = pd.read_csv("X_train.csv").drop(columns=["Unnamed: 0"], errors="ignore")

# Initialize SHAP Explainer
explainer = shap.Explainer(model, train_df[selected_features])

print("✅ Model, scaler, encoders, and SHAP explainer loaded successfully!")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Claim Severity Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data
        input_data = request.get_json()
        print("📥 Received Input Data:", input_data)

        input_df = pd.DataFrame([input_data])

        # Apply categorical encoding if encoders exist
        if label_encoders or onehot_encoder:
            for col in input_df.columns:
                if label_encoders and col in label_encoders:
                    input_df[col] = input_df[col].map(label_encoders[col]).fillna(0)
                elif onehot_encoder and col in onehot_encoder.categories_:
                    encoded_cols = pd.DataFrame(onehot_encoder.transform(input_df[[col]]).toarray(),
                                                columns=onehot_encoder.get_feature_names_out([col]))
                    input_df.drop(columns=[col], inplace=True)
                    input_df = pd.concat([input_df, encoded_cols], axis=1)

        # Align features with training data
        input_df = input_df.reindex(columns=selected_features, fill_value=0)

        # Predict claim cost
        prediction = model.predict(input_df)[0]
        print("✅ Prediction:", prediction)

        # Compute SHAP values
        shap_values = explainer(input_df)

        # Extract top 5 influential features
        shap_list = sorted(
            [(feature, float(value)) for feature, value in zip(selected_features, shap_values.values[0])],
            key=lambda x: abs(x[1]), reverse=True
        )[:5]
        print("📊 Top 5 SHAP Features:", shap_list)

        response = {
            "prediction": float(prediction),
            "r2_score": float(r2),
            "mae": float(mae),
            "top_5_shap_values": shap_list
        }

        return jsonify(response)

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
