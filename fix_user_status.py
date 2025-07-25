#!/usr/bin/env python3
"""
Fix user active status in database
"""

import sqlite3
from pathlib import Path

def fix_user_status():
    """Fix the is_active status for existing users"""
    db_path = "youtube_trans_downloader.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found at {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Update all users to have is_active = True and email_verified = True
        cursor.execute("""
            UPDATE users 
            SET is_active = 1, email_verified = 1 
            WHERE is_active IS NULL OR email_verified IS NULL
        """)
        
        updated_count = cursor.rowcount
        conn.commit()
        
        # Check the results
        cursor.execute("SELECT id, username, email, is_active, email_verified FROM users")
        users = cursor.fetchall()
        
        print(f"✅ Updated {updated_count} users")
        print("\n👥 Current user status:")
        for user in users:
            print(f"  ID: {user[0]}, Username: {user[1]}, Email: {user[2]}, Active: {user[3]}, Verified: {user[4]}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error fixing user status: {e}")

if __name__ == "__main__":
    print("🔧 Fixing User Active Status")
    print("=" * 30)
    fix_user_status()