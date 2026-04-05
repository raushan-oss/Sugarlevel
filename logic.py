from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from database import get_user_profile, get_recent_readings, get_all_readings

def calculate_lbgi_risk(glucose):
    """
    Clinically validated Low Blood Glucose Index (LBGI) formula.
    Transforms glucose (mg/dL) into a specialized risk value.
    """
    if glucose < 1: glucose = 1 # Avoid log(0)
    
    # fbg = 1.509 * (ln(G)^1.084 - 5.381)
    fbg = 1.509 * (np.power(np.log(glucose), 1.084) - 5.381)
    
    # Only readings below the center contribute to LBGI risk
    risk_value = np.minimum(0, fbg)
    
    # LBGI component: 10 * risk^2
    return 10 * (risk_value**2)

def get_dynamic_drop_rate():
    """
    Learns drop rate from baseline and SQLite history.
    """
    drop_rates = []
    try:
        base_df = pd.read_csv('data/glucose_dataset_1000_rows.csv')
        base_df['timestamp'] = pd.to_datetime(base_df['timestamp'])
        base_df = base_df.sort_values(by='timestamp').reset_index(drop=True)
        for i in range(1, len(base_df)):
            dt = (base_df.loc[i, 'timestamp'] - base_df.loc[i-1, 'timestamp']).total_seconds() / 60.0
            if dt > 0:
                rate = (base_df.loc[i-1, 'glucose'] - base_df.loc[i, 'glucose']) / dt
                if rate > 0: drop_rates.append(rate)
    except: pass
        
    history = get_all_readings()
    if len(history) > 1:
        for i in range(1, len(history)):
            try:
                prev_time = pd.to_datetime(history[i-1]['timestamp']).tz_localize(None)
                curr_time = pd.to_datetime(history[i]['timestamp']).tz_localize(None)
                dt = (curr_time - prev_time).total_seconds() / 60.0
                if dt > 0:
                    rate = (history[i-1]['glucose'] - history[i]['glucose']) / dt
                    if rate > 0: drop_rates.append(rate)
            except: continue
                
    if not drop_rates: return 0.25
    q1, q3 = np.percentile(drop_rates, [25, 75])
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
                meals.extend([dt - timedelta(days=1), dt, dt + timedelta(days=1)])
                
    base_level = 95.0 # Healthier clinical baseline
    learned_drop_rate = get_dynamic_drop_rate()
    future_minutes = np.arange(0, 12 * 60, 30)
    simulated_curve = []
    
    for lead_time in future_minutes:
        pred_time = now + timedelta(minutes=float(lead_time))
        past_meals = [m for m in meals if m <= pred_time]
        if past_meals:
            last_meal = max(past_meals)
            mins_since_meal = (pred_time - last_meal).total_seconds() / 60.0
            
            # Spike Logic: 60 min rise, then decay
            if mins_since_meal < 60:
                sugar = base_level + (mins_since_meal / 60.0) * 60
            else:
                peak = base_level + 60
                sugar = peak - ((mins_since_meal - 60) * learned_drop_rate)
                if sugar < 65.0: sugar = 65.0 # Body base
        else:
            sugar = base_level
        simulated_curve.append(sugar)
        
    offset_start = current_glucose - simulated_curve[0]
    calibrated_predictions = []
    for i, lead_time in enumerate(future_minutes):
        # High correction (blend), Low correction (keep until meal)
        if offset_start > 0:
            blend = max(0, 1.0 - (lead_time / 240.0))
            current_offset = offset_start * blend
        else:
            current_offset = offset_start
        calibrated_predictions.append(simulated_curve[i] + current_offset)
    
    time_to_dip = "Stable"
    found_dip = False
    if current_glucose <= 70:
        time_to_dip = "Already Low"
        found_dip = True
    
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
    # 1. Clinical LBGI Risk
    lbgi_score = calculate_lbgi_risk(current_glucose)
    
    # 2. Trend Risk
    readings = get_recent_readings(limit=2)
    drop_rate = 0.0
    if len(readings) == 2:
        try:
            p_time = pd.to_datetime(readings[0]['timestamp'])
            c_time = pd.to_datetime(readings[1]['timestamp'])
            dt = (c_time - p_time).total_seconds() / 60.0
            if dt > 0: drop_rate = (readings[0]['glucose'] - current_glucose) / dt
        except: pass
        
    predicted_30 = current_glucose - (drop_rate * 30)
    
    risk_level = "LOW"
    if lbgi_score >= 5.0 or current_glucose <= 70 or predicted_30 <= 70:
        risk_level = "HIGH"
        explanation = "⚠️ Clinical high risk. LBGI score indicates severe instability."
    elif lbgi_score >= 2.5 or drop_rate > 1.0:
        risk_level = "MEDIUM"
        explanation = "Moderate risk. Trend or biological scoring shows movement toward low range."
    else:
        explanation = "Clinical indicators are stable."
        
    return risk_level, explanation, lbgi_score
