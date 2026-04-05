import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from database import get_db_connection

def retrain_model(baseline_path='data/glucose_dataset_1000_rows.csv', save_path='models/trajectory_model.pkl'):
    """
    Fuses baseline CSV data with new personal SQLite data and trains a Trajectory Regressor.
    """
    try:
        baseline_df = pd.read_csv(baseline_path)
        baseline_df['timestamp'] = pd.to_datetime(baseline_df['timestamp'])
    except Exception as e:
        print(f"Error loading baseline: {e}")
        return False

    # Extract user history
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT timestamp, glucose FROM readings ORDER BY timestamp ASC')
    user_rows = c.fetchall()
    conn.close()

    user_df = pd.DataFrame([dict(row) for row in user_rows])
    if not user_df.empty:
        user_df['timestamp'] = pd.to_datetime(user_df['timestamp'])
        # Simplified assumption for auto-retrain: We will just merge glucose levels 
        # based on time of day vs meal gaps if they logged it. 
        # For a robust merge we assume their daily input follows a similar pattern.
        combined_df = pd.concat([baseline_df[['timestamp', 'glucose']], user_df[['timestamp', 'glucose']]], ignore_index=True)
    else:
        combined_df = baseline_df

    combined_df = combined_df.sort_values(by='timestamp').reset_index(drop=True)

    # Feature Engineering for Full-Day Trajectory
    # We predict continuous 'glucose' based on 'minute_of_day'
    # In a real medical model, we'd calculate exact time since past meals.
    # To keep it memory-light, we'll map absolute minute of day.
    
    combined_df['minute_of_day'] = combined_df['timestamp'].dt.hour * 60 + combined_df['timestamp'].dt.minute
    
    # We want to train the model to output a glucose value for any given minute
    X = combined_df[['minute_of_day']]
    y = combined_df['glucose']

    # Using polynomial features internally or simple regression (Linear is too straight, 
    # we'll use DecisionTreeRegressor to capture spikes better without OOMing Render)
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X, y)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print("Auto-Retrain Complete. Model updated.")
    return True

if __name__ == '__main__':
    retrain_model()
