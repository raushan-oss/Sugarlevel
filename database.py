import sqlite3
from datetime import datetime
import os

DB_PATH = 'app.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Table for one-time user setup
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wake_up_time TEXT,
            breakfast_time TEXT,
            lunch_time TEXT,
            dinner_time TEXT,
            activity_level TEXT
        )
    ''')
    
    # Table for daily readings
    c.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            glucose REAL,
            last_meal_category TEXT,
            activity TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_user_profile(data):
    conn = get_db_connection()
    c = conn.cursor()
    # Check if a profile exists, update it if so, else insert
    c.execute('SELECT id FROM user_profile LIMIT 1')
    row = c.fetchone()
    
    if row:
        c.execute('''
            UPDATE user_profile
            SET wake_up_time=?, breakfast_time=?, lunch_time=?, dinner_time=?, activity_level=?
            WHERE id=?
        ''', (data['wake_up_time'], data['breakfast_time'], data['lunch_time'], data['dinner_time'], data['activity_level'], row['id']))
    else:
        c.execute('''
            INSERT INTO user_profile (wake_up_time, breakfast_time, lunch_time, dinner_time, activity_level)
            VALUES (?, ?, ?, ?, ?)
        ''', (data['wake_up_time'], data['breakfast_time'], data['lunch_time'], data['dinner_time'], data['activity_level']))
    
    conn.commit()
    conn.close()

def get_user_profile():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM user_profile LIMIT 1')
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def add_reading(glucose, last_meal_category=None, activity=None):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO readings (glucose, last_meal_category, activity)
        VALUES (?, ?, ?)
    ''', (glucose, last_meal_category, activity))
    conn.commit()
    conn.close()

def get_recent_readings(limit=5):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM readings ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows][::-1] # return chronological

def get_todays_readings():
    conn = get_db_connection()
    c = conn.cursor()
    # SQLite date('now', 'localtime') grabs today's records
    c.execute("SELECT timestamp, glucose FROM readings WHERE date(timestamp) = date('now', 'localtime') ORDER BY timestamp ASC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_all_readings():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT timestamp, glucose FROM readings ORDER BY timestamp ASC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Initialize on import
if not os.path.exists(DB_PATH):
    init_db()
