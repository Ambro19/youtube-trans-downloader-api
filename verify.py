# Create verify.py
import sqlite3
conn = sqlite3.connect('youtube_trans_downloader.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(users)")
columns = [col[1] for col in cursor.fetchall()]
print("All columns in users table:")
for col in columns:
    print(f"  - {col}")
conn.close()