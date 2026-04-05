from datetime import datetime, timedelta
import pandas as pd
import joblib
import os
import numpy as np
from database import get_user_profile, get_recent_readings, get_all_readings

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

def get_dynamic_drop_rate():
    """
    Computes average glucose drop rate per minute from the base ML dataset 
    and the user's personal SQLite history.
    """
    drop_rates = []
    
    # 1. Learn from Base Model Dataset
    try:
        base_df = pd.read_csv('data/glucose_dataset_1000_rows.csv')
        base_df['timestamp'] = pd.to_datetime(base_df['timestamp'])
        base_df = base_df.sort_values(by='timestamp').reset_index(drop=True)
        for i in range(1, len(base_df)):
            dt = (base_df.loc[i, 'timestamp'] - base_df.loc[i-1, 'timestamp']).total_seconds() / 60.0
            if dt > 0:
                rate = (base_df.loc[i-1, 'glucose'] - base_df.loc[i, 'glucose']) / dt
                if rate > 0: # Only count actual dropping phases
                    drop_rates.append(rate)
    except:
        pass
        
    # 2. Learn from User's SQLite History
    history = get_all_readings()
    if len(history) > 1:
        for i in range(1, len(history)):
            try:
                prev_time = pd.to_datetime(history[i-1]['timestamp']).tz_localize(None)
                curr_time = pd.to_datetime(history[i]['timestamp']).tz_localize(None)
                dt = (curr_time - prev_time).total_seconds() / 60.0
                if dt > 0:
                    rate = (history[i-1]['glucose'] - history[i]['glucose']) / dt
                    if rate > 0:
                        drop_rates.append(rate)
            except:
                continue
                
    if not drop_rates:
        return 0.25 # Fallback static 15 mg/dL per hour if no data exists
        
    # Remove extreme outliers and average
    q1 = np.percentile(drop_rates, 25)
    q3 = np.percentile(drop_rates, 75)
    iqr = q3 - q1
    valid_rates = [r for r in drop_rates if (q1 - 1.5*iqr) <= r <= (q3 + 1.5*iqr)]
    
    return np.mean(valid_rates) if valid_rates else 0.25

def predict_daily_trajectory(current_glucose, client_time_str=None):
    labels = []
    data_points = []
    
    if client_time_str:
        now = pd.to_datetime(client_time_str).tz_localize(None)
    else:
        now = datetime.now()
        
    profile = get_user_profile()
    meals = []
    if profile:
        today_str = now.strftime("%Y-%m-%d")
        for m in ['breakfast_time', 'lunch_time', 'dinner_time']:
            if profile[m]:
                dt = datetime.strptime(f"{today_str} {profile[m]}", "%Y-%m-%d %H:%M")
                meals.append(dt)
                meals.append(dt + pd.Timedelta(days=1)) # Tomorrow's meals too
                meals.append(dt - pd.Timedelta(days=1)) # Yesterday's meals
                
    # Biological Simulation:
    # 1. Base metabolism level that it wants to drop to without food
    base_level = 65.0 
    
    future_minutes = np.arange(0, 12 * 60, 30)
    simulated_curve = []
    
    # AI dynamically learned decay rate per minute
    learned_drop_rate = get_dynamic_drop_rate()
    
    for lead_time in future_minutes:
        pred_time = now + timedelta(minutes=float(lead_time))
        
        # Find the most recent meal BEFORE this exact pred_time
        past_meals = [m for m in meals if m <= pred_time]
        if past_meals:
            last_meal = max(past_meals)
            mins_since_meal = (pred_time - last_meal).total_seconds() / 60.0
            
            # Spike mechanics: Peaks at 60 mins (+50 mg/dL), then drops dynamically
            if mins_since_meal < 60:
                # Rising
                sugar = base_level + 30 + (mins_since_meal / 60.0) * 50
            else:
                # Falling: based on the AI learned drop rate from the Base Model and SQLite history!
                peak = base_level + 80
                sugar = peak - ((mins_since_meal - 60) * learned_drop_rate)
                if sugar < base_level:
                    sugar = base_level
        else:
            sugar = base_level
            
        simulated_curve.append(sugar)
        
    # Calibration Offset:
    # Anchor the simulated curve to the User's actual current glucose
    # But slowly blend the offset out over 4 hours so it returns to the natural simulated curve
    offset_start = current_glucose - simulated_curve[0]
    
    calibrated_predictions = []
    for i, lead_time in enumerate(future_minutes):
        # Blend out the offset linearly over 240 mins (4 hours)
        blend_factor = max(0, 1.0 - (lead_time / 240.0))
        current_offset = offset_start * blend_factor
        
        final_val = simulated_curve[i] + current_offset
        calibrated_predictions.append(final_val)
    
    time_to_dip = "Stable"
    found_dip = False
    
    for i, lead_time in enumerate(future_minutes):
        pred_g = calibrated_predictions[i]
        pred_time = now + timedelta(minutes=float(lead_time))
        
        labels.append(pred_time.strftime("%H:%M"))
        data_points.append(round(pred_g, 1))
        
        if not found_dip and pred_g <= 70 and lead_time > 0:
            found_dip = True
            time_to_dip = f"At {pred_time.strftime('%I:%M %p')}"
            
    return labels, data_points, time_to_dip

def predict_risk(current_glucose, activity='Normal', client_time=None):
    if client_time:
        now = pd.to_datetime(client_time).tz_localize(None)
    else:
        now = datetime.now()
        
    hours_since_meal = get_time_since_meal(now)
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

    if current_glucose <= 70:
        risk_level = "HIGH"
        explanation = "Immediate action required: Glucose is critically low now."
    elif predicted_30 <= 70 or ml_risk_flag == 1:
        risk_level = "HIGH"
        explanation = "⚠️ High risk: Rapid mathematical drop OR Machine Learning detected anomalous pattern."
    elif current_glucose < 90 and hours_since_meal > 3.0:
        risk_level = "MEDIUM"
        explanation = f"It's been {hours_since_meal:.1f} hours since your last routine meal. Avoid dropping lower."
    elif drop_rate_per_min > 1.0:
        risk_level = "MEDIUM"
        explanation = "Glucose is dropping rapidly, keep an eye on it."

    return risk_level, explanation, predicted_30
