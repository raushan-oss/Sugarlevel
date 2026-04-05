import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Preprocess the raw data: clean, impute, and generate derived features.
    Expected columns: ['timestamp', 'glucose', 'carbs', 'insulin', 'activity', 'sleep']
    """
    if df.empty:
        return df

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Impute missing values (simple forward fill for glucose, 0 for carbs/insulin)
    if 'glucose' in df.columns:
        df['glucose'] = df['glucose'].ffill().bfill()
    if 'carbs' in df.columns:
        df['carbs'] = df['carbs'].fillna(0)
    if 'insulin' in df.columns:
        df['insulin'] = df['insulin'].fillna(0)
    if 'sleep' in df.columns:
        df['sleep'] = df['sleep'].ffill().bfill().fillna(0)
    
    # Activity mapping (Low: 1, Medium: 2, High: 3)
    if 'activity' in df.columns:
        activity_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'low': 1, 'medium': 2, 'high': 3}
        df['activity_level'] = df['activity'].map(activity_mapping).fillna(1)
        
    # Derived features
    # 1. Rate of glucose change (diff over 1 step)
    df['glucose_roc'] = df['glucose'].diff().fillna(0)
    
    # 2. Rolling average (last 3 readings)
    df['glucose_rolling_avg'] = df['glucose'].rolling(window=3, min_periods=1).mean()
    
    # 3. Time variables
    df['hour'] = df['timestamp'].dt.hour
    
    # Time of day encoded (Morning: 0, Afternoon: 1, Evening: 2, Night: 3)
    def categorize_time_of_day(hour):
        if 6 <= hour < 12: return 0 # Morning
        elif 12 <= hour < 18: return 1 # Afternoon
        elif 18 <= hour < 22: return 2 # Evening
        else: return 3 # Night
    df['time_of_day'] = df['hour'].apply(categorize_time_of_day)

    # 4. Time since last meal and insulin
    # To compute this, we need to track the last time carbs > 0 or insulin > 0
    # For simplicity in this vectorized approach, we will calculate the time difference
    # Assuming rows are frequent enough. If just one row is passed (for prediction), this requires state.
    # We will simulate state or keep it simple.
    
    # For training data (multi-row):
    if len(df) > 1:
        meal_times = df[df['carbs'] > 0]['timestamp']
        insulin_times = df[df['insulin'] > 0]['timestamp']
        
        # A simple approach is forward filling the time of the last event
        df['last_meal_time'] = np.where(df['carbs'] > 0, df['timestamp'], pd.NaT)
        df['last_meal_time'] = pd.to_datetime(df['last_meal_time']).ffill()
        df['time_since_meal_mins'] = (df['timestamp'] - df['last_meal_time']).dt.total_seconds() / 60.0
        df['time_since_meal_mins'] = df['time_since_meal_mins'].fillna(240) # default 4 hours
        
        df['last_insulin_time'] = np.where(df['insulin'] > 0, df['timestamp'], pd.NaT)
        df['last_insulin_time'] = pd.to_datetime(df['last_insulin_time']).ffill()
        df['time_since_insulin_mins'] = (df['timestamp'] - df['last_insulin_time']).dt.total_seconds() / 60.0
        df['time_since_insulin_mins'] = df['time_since_insulin_mins'].fillna(240) # default 4 hours
        
        # Drop temporary columns
        df = df.drop(columns=['last_meal_time', 'last_insulin_time'])
    else:
        # If single row (from API), require these to be calculated by the app or default them.
        # We will assume API provides 'time_since_meal_mins' and 'time_since_insulin_mins' 
        # or we will default them if not present.
        if 'time_since_meal_mins' not in df.columns:
            df['time_since_meal_mins'] = 240
        if 'time_since_insulin_mins' not in df.columns:
            df['time_since_insulin_mins'] = 240

    # Ensure required numerical features are returned
    features = [
        'glucose', 'carbs', 'insulin', 'activity_level', 'sleep',
        'glucose_roc', 'glucose_rolling_avg', 'time_of_day',
        'time_since_meal_mins', 'time_since_insulin_mins'
    ]
    
    return df[features]

def create_target(df, horizon_mins=30, threshold=70):
    """
    Create target variable for historical dataset: 
    Will glucose drop below threshold in the next `horizon_mins` minutes?
    """
    # Assuming data frequency is known. Without an explicit frequency, 
    # we'll look ahead in the dataframe within a time window.
    df['target'] = 0
    # Sort by timestamp
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    for i in range(len(df)):
        current_time = df.loc[i, 'timestamp']
        end_time = current_time + pd.Timedelta(minutes=horizon_mins)
        # Look at the window (current_time, end_time]
        future_window = df[(df['timestamp'] > current_time) & (df['timestamp'] <= end_time)]
        if not future_window.empty:
            if any(future_window['glucose'] < threshold):
                df.loc[i, 'target'] = 1
                
    return df
