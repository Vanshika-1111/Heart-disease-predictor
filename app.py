import json
import joblib # V. Important: joblib is imported globally
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # Import CORS
import os

# --- Configuration (Model Artifacts and Column Names) ---
ARTIFACT_DIR = 'model_artifacts'
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'best_model.pkl')
OHE_PATH = os.path.join(ARTIFACT_DIR, 'onehot_encoder.pkl')
PT_PATH = os.path.join(ARTIFACT_DIR, 'power_transformer.pkl')

# Column definitions (Total 13 features) - MUST MATCH train_new.py
ONE_COLS = ['cp', 'restecg', 'slope', 'ca', 'thal']
CONT_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
ORDINAL_COLS = ['sex', 'fbs', 'exang']

# Initialize Flask application
app = Flask(__name__)
# FIX: Apply CORS globally for maximum compatibility 
CORS(app) 

# Global variables for loaded artifacts
model = None
ohe = None
pt = None
ARTIFACTS_LOADED = False

# FIX: We rely on the global joblib import, so no change needed inside load_artifacts function.

def load_artifacts():
    """Loads the saved model and preprocessors from the training script."""
    global model, ohe, pt, ARTIFACTS_LOADED
    
    if ARTIFACTS_LOADED:
        return True

    print("Loading model artifacts...")
    try:
        # Load the saved model, OneHotEncoder, and PowerTransformer (Yeo-Johnson)
        # The joblib.load calls rely on the global import at the top of the file.
        model = joblib.load(MODEL_PATH)
        ohe = joblib.load(OHE_PATH)
        pt = joblib.load(PT_PATH)
        ARTIFACTS_LOADED = True
        print("Model and preprocessors loaded successfully.")
        return True
    except FileNotFoundError as e:
        print(f"Error loading model: {e}. Please ensure you ran 'python train_new.py' first.")
        return False
    except Exception as e:
        print(f"Unexpected error: Failed to load model: {e}")
        return False

# Load model artifacts when the application starts
load_artifacts()
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# 1. Home Route now serves index.html directly from the Flask server
@app.route('/')
def home():
    """Serves the main HTML file to eliminate CORS issues."""
    try:
        # Flask serves the index.html file located in the same directory
        return render_template('index.html')
    except Exception as e:
        return f"Error loading index.html: {e}", 500


# 2. Prediction Route
@app.route('/api/predict', methods=['POST']) 
def predict():
    """Receives data from the frontend and performs prediction."""
    if not load_artifacts():
        return jsonify({"error": "Internal Error: Model artifacts failed to load."}), 500

    if not request.is_json:
        return jsonify({"error": "Request Content-Type must be 'application/json'."}), 400

    try:
        data = request.get_json()
        
        # 1. Check for all 13 required features
        required_features = ONE_COLS + CONT_COLS + ORDINAL_COLS
        
        input_data = {}
        for feature in required_features:
            value = data.get(feature)
            if value is None:
                return jsonify({"error": f"Required feature '{feature}' is missing."}), 400
            input_data[feature] = value

        # 2. Prepare DataFrame for model input
        df_new = pd.DataFrame([input_data])
        
        # 3. Preprocessing steps (MUST mirror train_new.py exactly)
        
        # 3a. One-Hot Encoding (OHE)
        X_ohe = ohe.transform(df_new[ONE_COLS])
        ohe_feature_names = ohe.get_feature_names_out(ONE_COLS)
        X_ohe_df = pd.DataFrame(X_ohe, columns=ohe_feature_names)
        
        # 3b. Power Transformation (Yeo-Johnson)
        X_cont = pt.transform(df_new[CONT_COLS])
        X_cont_df = pd.DataFrame(X_cont, columns=CONT_COLS)
        
        # 3c. Concatenate all features
        X_processed = pd.concat([
            X_ohe_df, 
            X_cont_df, 
            df_new[ORDINAL_COLS].reset_index(drop=True)
        ], axis=1)
        
        # 4. Prediction
        prediction_result = model.predict(X_processed)[0] # 0 (No Disease) or 1 (Disease)
        prediction_proba = model.predict_proba(X_processed)[0]
        
        # 5. Send result
        response_data = {
            "status": "success",
            "model_used": "RandomForestClassifier",
            "prediction": int(prediction_result), 
            "probability_disease": float(prediction_proba[1]),
            "message": "Prediction completed."
        }
        
        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# Run the application
if __name__ == '__main__':
    # Fix: Port 8080 used to match index.html and avoid 5000 conflicts
    app.run(debug=True, port=8080)