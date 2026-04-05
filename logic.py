from datetime import datetime
import pandas as pd
import joblib
import os
from database import get_user_profile, get_recent_readings

ML_MODEL_PATH = 'models/light_model.pkl'
ml_model = None
if os.path.exists(ML_MODEL_PATH):
    try:
        ml_model = joblib.load(ML_MODEL_PATH)
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

def predict_risk(current_glucose, last_meal_category=None):
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
            glucose_diff = prev['glucose'] - current_glucose 
            drop_rate_per_min = glucose_diff / time_diff_mins

    # Predict future (30 mins drop)
    predicted_30 = current_glucose
    if drop_rate_per_min > 0:
        predicted_30 = current_glucose - (drop_rate_per_min * 30)

    # ML Prediction
    ml_risk_flag = 0
    if ml_model is not None:
        try:
            # Features: ['glucose', 'drop_rate']
            pred = ml_model.predict([[current_glucose, drop_rate_per_min]])
            ml_risk_flag = int(pred[0])
        except:
            pass

    # Time to next dip calculation
    time_to_dip = "Stable"
    if drop_rate_per_min > 0:
        minutes_to_70 = (current_glucose - 70) / drop_rate_per_min
        if minutes_to_70 < 0:
            time_to_dip = "Already Low"
        else:
            time_to_dip = f"~{int(minutes_to_70)} mins"

    # Assess Risk
    risk_level = "LOW"
    explanation = "Everything looks stable."

    if current_glucose < 70:
        risk_level = "HIGH"
        explanation = "Immediate action required: Glucose is critically low now."
        return risk_level, explanation, predicted_30, time_to_dip
        
    if predicted_30 < 70 or ml_risk_flag == 1:
        risk_level = "HIGH"
        explanation = f"⚠️ High risk: Glucose projected to drop rapidly OR Machine Learning detected anomalous pattern."
        return risk_level, explanation, predicted_30, time_to_dip

    if current_glucose < 90 and hours_since_meal > 3.0:
        risk_level = "MEDIUM"
        explanation = f"It has been {hours_since_meal:.1f} hours since your last routine meal. Glucose is dropping low."
    elif drop_rate_per_min > 1.0:
        risk_level = "MEDIUM"
        explanation = "Glucose is dropping rapidly, keep an eye on it."

    return risk_level, explanation, predicted_30, time_to_dip
