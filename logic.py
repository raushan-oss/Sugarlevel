from datetime import datetime, timedelta
import pandas as pd
import joblib
import os
import numpy as np
from database import get_user_profile, get_recent_readings

# Load legacy light model for immediate risk (optional but kept for safety)
ML_MODEL_PATH = 'models/light_model.pkl'
ml_model = None
if os.path.exists(ML_MODEL_PATH):
    try:
        ml_model = joblib.load(ML_MODEL_PATH)
    except:
        pass

# Load the new 24-Hour Trajectory Regressor
TRAJ_MODEL_PATH = 'models/trajectory_model.pkl'
traj_model = None
if os.path.exists(TRAJ_MODEL_PATH):
    try:
        traj_model = joblib.load(TRAJ_MODEL_PATH)
    except:
        pass

def get_time_since_meal(current_time=None):
    if current_time is None:
        current_time = datetime.now()
        
    profile = get_user_profile()
    if not profile:
        return 0
        
    today_str = current_time.strftime("%Y-%m-%d")
    meals = []
    for m in ['breakfast_time', 'lunch_time', 'dinner_time']:
        if profile[m]:
            dt = datetime.strptime(f"{today_str} {profile[m]}", "%Y-%m-%d %H:%M")
            if dt > current_time:
                dt = dt - pd.Timedelta(days=1)
            meals.append(dt)
            
    if not meals:
        return 0
        
    last_meal = max(meals)
    delta_hours = (current_time - last_meal).total_seconds() / 3600.0
    return delta_hours

def predict_daily_trajectory(current_glucose):
    """
    Simulates the entire day's glucose trajectory based on the Trajectory ML Regressor,
    locking the curve to the ACTUAL current_glucose provided by the user.
    """
    labels = []
    data_points = []
    
    if traj_model is None:
        return labels, data_points, "Stable"
        
    now = datetime.now()
    
    # Generate 12 hours of future simulation (every 30 mins)
    future_minutes = np.arange(0, 12 * 60, 30)
    
    # Prepare the feature matrix: 'minute_of_day'
    current_minute_of_day = now.hour * 60 + now.minute
    
    X_pred = np.zeros((len(future_minutes), 1))
    for i, lead_time in enumerate(future_minutes):
        minute_of_day = (current_minute_of_day + lead_time) % 1440
        X_pred[i] = [minute_of_day]
        
    # Baseline predicted trajectory from the model
    raw_predictions = traj_model.predict(X_pred)
    
    # Calibration Offset: Anchor the model's current predicted point to the User's actual current glucose
    # raw_predictions[0] is theoretically the model's expectation of the current glucose
    offset = current_glucose - raw_predictions[0]
    
    # Apply offset and add some biological smoothing
    calibrated_predictions = raw_predictions + offset
    
    # Extract "Next Dip"
    time_to_dip = "Stable"
    found_dip = False
    
    for i, lead_time in enumerate(future_minutes):
        pred_g = calibrated_predictions[i]
        
        # Format label (HH:MM)
        pred_time = now + timedelta(minutes=float(lead_time))
        labels.append(pred_time.strftime("%H:%M"))
        data_points.append(round(pred_g, 1))
        
        # Check for dip below 70
        if not found_dip and pred_g < 70 and lead_time > 0:
            found_dip = True
            time_to_dip = f"At {pred_time.strftime('%I:%M %p')}"
            
    return labels, data_points, time_to_dip

def predict_risk(current_glucose, activity='Normal'):
    hours_since_meal = get_time_since_meal()
    readings = get_recent_readings(limit=2)
    
    drop_rate_per_min = 0.0
    if len(readings) == 2:
        prev = readings[0]
        curr = readings[1]
        
        try:
            prev_time = datetime.strptime(prev['timestamp'], "%Y-%m-%d %H:%M:%S")
            curr_time = datetime.strptime(curr['timestamp'], "%Y-%m-%d %H:%M:%S")
        except:
            prev_time = pd.to_datetime(prev['timestamp'])
            curr_time = pd.to_datetime(curr['timestamp'])

        time_diff_mins = (curr_time - prev_time).total_seconds() / 60.0
        
        if time_diff_mins > 0:
            drop_rate_per_min = (prev['glucose'] - current_glucose) / time_diff_mins

    # 30min Safety
    predicted_30 = current_glucose
    if drop_rate_per_min > 0:
        predicted_30 = current_glucose - (drop_rate_per_min * 30)

    # Re-use our ML classifier for immediate anomaly detection
    ml_risk_flag = 0
    if ml_model is not None:
        try:
            pred = ml_model.predict([[current_glucose, drop_rate_per_min]])
            ml_risk_flag = int(pred[0])
        except:
            pass

    # Assess overall Risk
    risk_level = "LOW"
    explanation = "Trajectory is safe. AI forecasting stable levels."

    if current_glucose < 70:
        risk_level = "HIGH"
        explanation = "Immediate action required: Glucose is critically low now."
    elif predicted_30 < 70 or ml_risk_flag == 1:
        risk_level = "HIGH"
        explanation = "⚠️ High risk: Rapid mathematical drop OR Machine Learning detected anomalous pattern."
    elif current_glucose < 90 and hours_since_meal > 3.0:
        risk_level = "MEDIUM"
        explanation = f"It's been {hours_since_meal:.1f} hours since your last routine meal. Avoid dropping lower."
    elif drop_rate_per_min > 1.0:
        risk_level = "MEDIUM"
        explanation = "Glucose is dropping rapidly, keep an eye on it."

    return risk_level, explanation, predicted_30
