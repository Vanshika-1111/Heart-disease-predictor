import pandas as pd
import numpy as np
import joblib
import os

# Import necessary modules from scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score

# --- Configuration (Data column definitions) ---
DATA_FILE = 'Heart.csv'
ARTIFACT_DIR = 'model_artifacts'

# Target Column is set to 'condition' based on your dataset structure.
TARGET_COLUMN = 'condition' 

# Features for One-Hot Encoding (Nominal Categorical)
ONE_COLS = ['cp', 'restecg', 'slope', 'ca', 'thal'] 

# Features for Power Transformation (Continuous/Numerical)
CONT_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'] 

# Binary/Ordinal features (used directly)
ORDINAL_COLS = ['sex', 'fbs', 'exang'] 

# --- Main Training Logic ---
def train_and_save_model():
    print("--- 1. Data Loading and Splitting ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILE}' not found. Please ensure it is in the same directory.")
        return

    # 2. Separate Target Variable (Fixed to use TARGET_COLUMN)
    try:
        # X: Features (drop the target column)
        X = df.drop(TARGET_COLUMN, axis=1) 
        # y: Target variable
        y = df[TARGET_COLUMN]
    except KeyError as e:
        print(f"Critical Error: Column {e} not found in the dataset. Please ensure TARGET_COLUMN ('{TARGET_COLUMN}') is correct.")
        return

    # 3. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Training set size: {len(X_train)} samples")
    
    # --- 2. Preprocessing Pipeline ---
    print("--- 2. Data Preprocessing ---")

    # 2a. Handle Categorical Features using One-Hot Encoding
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_ohe = ohe.fit_transform(X_train[ONE_COLS])
    X_test_ohe = ohe.transform(X_test[ONE_COLS])
    
    ohe_feature_names = ohe.get_feature_names_out(ONE_COLS)
    X_train_ohe_df = pd.DataFrame(X_train_ohe, columns=ohe_feature_names, index=X_train.index)
    X_test_ohe_df = pd.DataFrame(X_test_ohe, columns=ohe_feature_names, index=X_test.index)
    
    # 2b. Handle Continuous Features using Power Transformation
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    X_train_cont = pt.fit_transform(X_train[CONT_COLS])
    X_test_cont = pt.transform(X_test[CONT_COLS])
    
    X_train_cont_df = pd.DataFrame(X_train_cont, columns=CONT_COLS, index=X_train.index)
    X_test_cont_df = pd.DataFrame(X_test_cont, columns=CONT_COLS, index=X_test.index)
    
    # 2c. Concatenate all processed features (reset index for clean concatenation)
    X_train_processed = pd.concat([
        X_train_ohe_df.reset_index(drop=True), 
        X_train_cont_df.reset_index(drop=True), 
        X_train[ORDINAL_COLS].reset_index(drop=True)
    ], axis=1)
    
    X_test_processed = pd.concat([
        X_test_ohe_df.reset_index(drop=True), 
        X_test_cont_df.reset_index(drop=True), 
        X_test[ORDINAL_COLS].reset_index(drop=True)
    ], axis=1)

    y_train_aligned = y_train.reset_index(drop=True)

    # --- 3. Model Training (Random Forest) ---
    print("--- 3. Training Model (Random Forest) ---")
    
    # Initialize the model and parameter grid for hyperparameter tuning
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    
    # Use GridSearchCV to find the best parameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=0)
    grid_search.fit(X_train_processed, y_train_aligned)

    best_model = grid_search.best_estimator_
    print(f"\nBest Model Parameters: {grid_search.best_params_}")

    # --- 4. Evaluation ---
    print("--- 4. Model Evaluation ---")
    y_pred = best_model.predict(X_test_processed)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Recall Score (Crucial for medical prediction): {recall:.4f}")

    # --- 5. Saving Artifacts ---
    print("--- 5. Saving Model Artifacts ---")
    
    # Create the artifacts directory if it doesn't exist
    if not os.path.exists(ARTIFACT_DIR):
        os.makedirs(ARTIFACT_DIR)

    # Save the trained model and preprocessors
    joblib.dump(best_model, os.path.join(ARTIFACT_DIR, 'best_model.pkl'))
    joblib.dump(ohe, os.path.join(ARTIFACT_DIR, 'onehot_encoder.pkl'))
    joblib.dump(pt, os.path.join(ARTIFACT_DIR, 'power_transformer.pkl'))

    print(f"Training completed successfully! Model and preprocessors saved to '{ARTIFACT_DIR}'.")

if __name__ == '__main__':
    train_and_save_model()