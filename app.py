from flask import Flask, render_template, request, jsonify, redirect, url_for
from database import get_user_profile, save_user_profile, add_reading, get_recent_readings
from logic import predict_risk
import datetime

app = Flask(__name__)

@app.route('/')
def index():
    # If no profile, force setup
    profile = get_user_profile()
    if not profile:
        return redirect(url_for('setup'))
    return render_template('index.html')

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    if request.method == 'POST':
        data = request.form
        save_user_profile({
            'wake_up_time': data.get('wake_up_time', '07:00'),
            'breakfast_time': data.get('breakfast_time', '08:00'),
            'lunch_time': data.get('lunch_time', '13:00'),
            'dinner_time': data.get('dinner_time', '19:00'),
            'activity_level': data.get('activity_level', 'Medium')
        })
        return redirect(url_for('index'))
        
    profile = get_user_profile() or {}
    return render_template('setup.html', profile=profile)

@app.route('/dashboard')
def dashboard():
    # Pass profile status for UI toggles
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    from logic import predict_daily_trajectory
    try:
        data = request.json
        glucose = float(data.get('glucose'))
        activity = data.get('activity', 'Normal')
        client_time = data.get('client_time')
        
        add_reading(glucose=glucose, activity=activity)
        
        risk_level, explanation, projected_30 = predict_risk(glucose, client_time=client_time)
        
        # Build 12-Hour Forecast using the local browser time
        labels, trajectory_data, time_to_dip = predict_daily_trajectory(glucose, client_time_str=client_time)
        
        return jsonify({
            'risk_level': risk_level,
            'explanation': explanation,
            'projected_30': round(projected_30, 2),
            'time_to_dip': time_to_dip,
            'glucose': glucose,
            'trajectory_labels': labels,
            'trajectory_data': trajectory_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    from retrain import retrain_model
    success = retrain_model()
    if success:
        return jsonify({'status': 'success', 'message': 'AI profile successfully updated!'})
    return jsonify({'status': 'error', 'message': 'Failed to compile new AI weights.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
