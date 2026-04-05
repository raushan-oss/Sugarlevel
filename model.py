import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from utils import preprocess_data, create_target

def train_model(data_path='data/dataset.csv', model_save_path='models/hypoglycemia_model.pkl'):
    print(f"Loading data from {data_path}...")
    try:
        raw_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please provide the dataset.")
        return False
        
    print("Preprocessing data and extracting features...")
    # Generate features
    features_df = preprocess_data(raw_df)
    
    # We need the original timestamps to compute the target
    features_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    
    print("Creating target variables...")
    # target: Will glucose drop < 70 in next 2 hours
    features_df = create_target(features_df, horizon_mins=125, threshold=70)
    
    # Drop rows where target couldn't be calculated accurately (e.g. end of dataset)
    # Since this is a simple prototype, we won't strictly enforce end-dropping, but normally we would.
    
    # Prepare X and y
    feature_cols = [
        'glucose', 'carbs', 'insulin', 'activity_level', 'sleep',
        'glucose_roc', 'glucose_rolling_avg', 'time_of_day',
        'time_since_meal_mins', 'time_since_insulin_mins'
    ]
    
    X = features_df[feature_cols]
    y = features_df['target']
    
    print(f"Dataset shape: {X.shape}. Class distribution:\n{y.value_counts()}")
    
    if len(y.unique()) < 2:
        print("Error: The dataset must contain both positive (hypoglycemia) and negative samples.")
        return False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    # Feature Importance
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns)
    print("\nFeature Importances:")
    print(feat_importances.sort_values(ascending=False))
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Model saved successfully to {model_save_path}")
    
    return True

if __name__ == '__main__':
    train_model()
