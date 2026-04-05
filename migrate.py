import sqlite3
import os

DB_PATH = 'app.db'

def migrate():
    if not os.path.exists(DB_PATH):
        print("DB does not exist. No migration needed.")
        return
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check for missing columns in user_profile
    c.execute("PRAGMA table_info(user_profile)")
    columns = [row[1] for row in c.fetchall()]
    
    if 'insulin_sensitivity' not in columns:
        print("Adding insulin_sensitivity column...")
        c.execute("ALTER TABLE user_profile ADD COLUMN insulin_sensitivity REAL DEFAULT 15")
        
    if 'carb_ratio' not in columns:
        print("Adding carb_ratio column...")
        c.execute("ALTER TABLE user_profile ADD COLUMN carb_ratio REAL DEFAULT 10")
        
    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == '__main__':
    migrate()
