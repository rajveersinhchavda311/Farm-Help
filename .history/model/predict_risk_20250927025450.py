# model/predict_risk.py

import sys
import json
import argparse
import os
import pickle
import pandas as pd

# --- Verify these filenames match your files in the 'model' folder ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
RISK_MODEL_PATH = os.path.join(MODEL_DIR, 'disease_risk_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'risk_scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'risk_label_encoder.pkl')
# --------------------------------------------------------------------

try:
    with open(RISK_MODEL_PATH, 'rb') as f:
        risk_model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print(json.dumps({"error": f"Error loading risk model/scaler/encoder: {str(e)}"}))
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json_string', type=str)
    args = parser.parse_args()

    try:
        # Parse the JSON string from Node.js into a Python dictionary
        input_data = json.loads(args.input_json_string)
        
        # IMPORTANT: Create a pandas DataFrame with the columns in the exact
        # order your model was trained on.
        feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        df = pd.DataFrame([input_data], columns=feature_order)

        # Scale the input data using the loaded scaler
        scaled_features = scaler.transform(df)
        
        # Make predictions
        prediction_idx = risk_model.predict(scaled_features)[0]
        prediction_label = label_encoder.inverse_transform([prediction_idx])[0]
        
        # Get probabilities for all classes
        probabilities = risk_model.predict_proba(scaled_features)[0]
        prob_dict = {label_encoder.classes_[i]: float(probabilities[i]) for i in range(len(probabilities))}

        result = {
            "disease_risk": prediction_label,
            "probabilities": prob_dict
        }
        
        # This print statement sends the result back to Node.js
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()