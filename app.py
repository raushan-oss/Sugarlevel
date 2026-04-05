from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from utils import preprocess_data

app = Flask(__name__)

MODEL_PATH = 'models/hypoglycemia_model.pkl'
model = None

# Load model if exists
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            return jsonify({'error': 'Model not trained yet. Please provide historical data and train the model first.'}), 500

    try:
        # Get JSON data
        data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # In a real app, time_since_meal_mins and time_since_insulin_mins 
        # would be calculated dynamically from user history up to this point.
        # For the prediction endpoint, if not provided, we give sensible defaults
        if 'time_since_meal_mins' not in df.columns:
            df['time_since_meal_mins'] = 120 # 2 hours
        if 'time_since_insulin_mins' not in df.columns:
            df['time_since_insulin_mins'] = 180 # 3 hours
            
        # Add a dummy previous timestamp to allow rolling and diff if we don't have history
        # (This is simplified. Correctly doing this requires historical session data)
        # We will bypass the rolling avg and diff for single rows by just defaulting them
        
        features_df = preprocess_data(df)
        
        # Ensure correct column order match
        expected_cols = [
            'glucose', 'carbs', 'insulin', 'activity_level', 'sleep',
            'glucose_roc', 'glucose_rolling_avg', 'time_of_day',
            'time_since_meal_mins', 'time_since_insulin_mins'
        ]
        
        # Fill missing features for a single prediction row
        for col in expected_cols:
            if col not in features_df.columns:
                features_df[col] = 0
                
        # If glucose_rolling_avg is NaN, fill it with current glucose
        if 'glucose_rolling_avg' in features_df.columns and pd.isna(features_df['glucose_rolling_avg'].iloc[0]):
            features_df['glucose_rolling_avg'] = features_df['glucose']
            
        X_pred = features_df[expected_cols]
        
        # Make prediction
        pred_class = model.predict(X_pred)[0]
        pred_proba = model.predict_proba(X_pred)[0][1] # Probability of Class 1 (Hypoglycemia)
        
        # Risk thresholds
        if pred_proba > 0.65:
            risk_level = 'High'
        elif pred_proba > 0.35:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
            
        # Determine top contributing factors roughly
        importances = model.feature_importances_
        feature_contributions = X_pred.iloc[0] * importances
        top_factor_idx = np.argsort(feature_contributions.values)[::-1]
        
        explanation = f"{expected_cols[top_factor_idx[0]]} and {expected_cols[top_factor_idx[1]]} are key factors."
        
        return jsonify({
            'risk_level': risk_level,
            'probability': round(pred_proba * 100, 2),
            'explanation': explanation,
            'prediction': int(pred_class)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filepath = os.path.join('data', 'dataset.csv')
        file.save(filepath)
        return jsonify({'success': 'Data uploaded successfully. Ready to train model!'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
