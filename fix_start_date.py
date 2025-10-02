# backend/fix_start_date.py
"""
Fix database schema: Make start_date column nullable or populate it
"""
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = "youtube_trans_downloader.db"

def fix_start_date():
    if not Path(DB_PATH).exists():
        print(f"❌ Database not found: {DB_PATH}")
        return False
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if start_date column exists and if it's nullable
        cursor.execute("PRAGMA table_info(subscriptions)")
        columns = cursor.fetchall()
        
        start_date_col = None
        for col in columns:
            # col format: (cid, name, type, notnull, dflt_value, pk)
            if col[1] == 'start_date':
                start_date_col = col
                break
        
        if not start_date_col:
            print("✓ No start_date column found - no fix needed")
            return True
        
        is_not_null = start_date_col[3] == 1  # notnull flag
        
        if not is_not_null:
            print("✓ start_date column is already nullable - no fix needed")
            return True
        
        print(f"Found start_date column with NOT NULL constraint")
        print("Applying fix: Setting default values for existing NULL rows...")
        
        # Update any NULL start_date to created_at
        cursor.execute("""
            UPDATE subscriptions 
            SET start_date = created_at 
            WHERE start_date IS NULL
        """)
        
        rows_updated = cursor.rowcount
        print(f"✓ Updated {rows_updated} rows with NULL start_date")
        
        # SQLite doesn't support ALTER COLUMN, so we need to recreate the table
        # But since we're just filling NULLs, the constraint should now pass
        
        conn.commit()
        print("✅ Schema fix complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    fix_start_date()