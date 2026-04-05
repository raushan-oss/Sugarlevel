from datetime import datetime
import pandas as pd
from database import get_user_profile, get_recent_readings

def get_time_since_meal(current_time=None):
    """
    Calculates time since the last routine meal dynamically to minimize user input.
    """
    if current_time is None:
        current_time = datetime.now()
        
    profile = get_user_profile()
    if not profile:
        return 0 # No profile, can't auto-calculate
        
    # Convert profile times to today's datetime objects
    today_str = current_time.strftime("%Y-%m-%d")
    
    meals = []
    for m in ['breakfast_time', 'lunch_time', 'dinner_time']:
        if profile[m]:
            dt = datetime.strptime(f"{today_str} {profile[m]}", "%Y-%m-%d %H:%M")
            # If meal time was yesterday relative to am/pm limits (simple fix: if time is in future, it was yesterday)
            if dt > current_time:
                dt = dt - pd.Timedelta(days=1)
            meals.append(dt)
            
    if not meals:
        return 0
        
    # Find the most recent meal that has passed
    last_meal = max(meals)
    delta_hours = (current_time - last_meal).total_seconds() / 3600.0
    return delta_hours

def predict_risk(current_glucose, last_meal_category=None):
    """
    Hybrid Predictive Engine
    Returns: Risk Level, Explanation, Predictive Projected Value
    """
    # 1. Gather context
    hours_since_meal = get_time_since_meal()
    readings = get_recent_readings(limit=2)
    
    # 2. Extract History Drop Rate
    drop_rate_per_min = 0.0
    if len(readings) == 2:
        prev = readings[0]
        curr = readings[1]
        
        # Parse timestamps (SQLite defaults to string "YYYY-MM-DD HH:MM:SS")
        try:
            prev_time = datetime.strptime(prev['timestamp'], "%Y-%m-%d %H:%M:%S")
            curr_time = datetime.strptime(curr['timestamp'], "%Y-%m-%d %H:%M:%S")
        except:
            # Fallback for manual tests/ISO formats
            prev_time = pd.to_datetime(prev['timestamp'])
            curr_time = pd.to_datetime(curr['timestamp'])

        time_diff_mins = (curr_time - prev_time).total_seconds() / 60.0
        
        if time_diff_mins > 0:
            glucose_diff = prev['glucose'] - current_glucose 
            drop_rate_per_min = glucose_diff / time_diff_mins

    # 3. Predict future (30 mins drop)
    predicted_30 = current_glucose
    if drop_rate_per_min > 0: # It's dropping
        predicted_30 = current_glucose - (drop_rate_per_min * 30)

    # 4. Assess Risk (Hybrid Logic)
    risk_level = "LOW"
    explanation = "Everything looks stable."

    # A. Direct Rule-based Safety
    if current_glucose < 70:
        risk_level = "HIGH"
        explanation = "Immediate action required: Glucose is critically low now."
        return risk_level, explanation, predicted_30
        
    # B. Trend Calculation High Risk
    if predicted_30 < 70:
        risk_level = "HIGH"
        explanation = f"⚠️ High risk: Based on recent drop rate, glucose projected to hit {predicted_30:.1f} mg/dL in 30 mins."
        return risk_level, explanation, predicted_30

    # C. Medium Risk Conditions
    if current_glucose < 90 and hours_since_meal > 3.0:
        risk_level = "MEDIUM"
        explanation = f"It has been {hours_since_meal:.1f} hours since your routine meal. Glucose is dropping low."
    elif drop_rate_per_min > 1.0: # Dropping faster than 1 mg/min
        risk_level = "MEDIUM"
        explanation = "Glucose is dropping rapidly, keep an eye on it."

    return risk_level, explanation, predicted_30
