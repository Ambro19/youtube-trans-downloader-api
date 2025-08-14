#!/usr/bin/env python3
"""
🔧 DATABASE SCHEMA FIX SCRIPT
===============================
This script adds missing columns to your existing database
without losing any existing users or data.

Run this ONCE to fix the schema mismatch issue.
"""

import sqlite3
import os
from pathlib import Path

def fix_database_schema():
    """Add missing columns to existing database"""
    
    # Find the database file
    possible_db_paths = [
        "users.db",
        "database.db", 
        "youtube_downloader.db",
        "backend/users.db",
        "backend/database.db"
    ]
    
    db_path = None
    for path in possible_db_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if not db_path:
        print("❌ No database file found!")
        print("Looking for database files...")
        # Find any .db files in current directory
        db_files = list(Path(".").glob("*.db")) + list(Path("backend").glob("*.db"))
        if db_files:
            print(f"Found database files: {[str(f) for f in db_files]}")
            db_path = str(db_files[0])
            print(f"Using: {db_path}")
        else:
            print("No .db files found. Database might need to be recreated.")
            return False
    
    print(f"🔥 Found database: {db_path}")
    
    # Create backup first
    backup_path = f"{db_path}.backup"
    import shutil
    shutil.copy2(db_path, backup_path)
    print(f"✅ Backup created: {backup_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check current schema
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        existing_columns = [col[1] for col in columns]
        
        print(f"🔍 Current columns: {existing_columns}")
        
        # Add missing columns if they don't exist
        columns_to_add = {
            'avatar_url': 'TEXT',
            'full_name': 'TEXT',
            'updated_at': 'TIMESTAMP',
            'is_active': 'BOOLEAN DEFAULT 1',
            'is_verified': 'BOOLEAN DEFAULT 0',
            'stripe_customer_id': 'TEXT'
        }
        
        added_columns = []
        for column_name, column_type in columns_to_add.items():
            if column_name not in existing_columns:
                try:
                    sql = f"ALTER TABLE users ADD COLUMN {column_name} {column_type}"
                    cursor.execute(sql)
                    added_columns.append(column_name)
                    print(f"✅ Added column: {column_name}")
                except sqlite3.Error as e:
                    print(f"⚠️  Could not add {column_name}: {e}")
        
        # Commit changes
        conn.commit()
        
        # Verify final schema
        cursor.execute("PRAGMA table_info(users)")
        final_columns = cursor.fetchall()
        final_column_names = [col[1] for col in final_columns]
        
        print(f"🎯 Final columns: {final_column_names}")
        
        # Check existing users
        cursor.execute("SELECT id, username, email FROM users")
        users = cursor.fetchall()
        
        print(f"✅ DATABASE FIXED SUCCESSFULLY!")
        print(f"📊 Found {len(users)} existing users:")
        for user in users:
            print(f"   - ID: {user[0]}, Username: {user[1]}, Email: {user[2]}")
        
        if added_columns:
            print(f"🆕 Added columns: {added_columns}")
        else:
            print("ℹ️  No new columns needed to be added")
            
        return True
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        # Restore backup if something went wrong
        shutil.copy2(backup_path, db_path)
        print(f"🔄 Restored backup due to error")
        return False
        
    finally:
        conn.close()

def test_database_connection():
    """Test if the database works after fixing"""
    try:
        import sqlite3
        
        # Find database
        possible_paths = ["users.db", "database.db", "youtube_downloader.db", "backend/users.db"]
        db_path = None
        for path in possible_paths:
            if os.path.exists(path):
                db_path = path
                break
        
        if not db_path:
            print("❌ No database found for testing")
            return False
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Test query that was failing
        cursor.execute("""
            SELECT id, username, email, avatar_url, full_name 
            FROM users 
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        print("✅ Database test PASSED - schema is now compatible!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 FIXING DATABASE SCHEMA...")
    print("=" * 50)
    
    success = fix_database_schema()
    
    if success:
        print("\n🧪 TESTING DATABASE...")
        test_success = test_database_connection()
        
        if test_success:
            print("\n🎉 SUCCESS! Your database is now fixed!")
            print("🚀 You can now restart your server and login with existing users:")
            print("   - Xiggy")
            print("   - OneTechly") 
            print("   - LovePets")
            print("\n⚡ Run: python run.py")
        else:
            print("\n⚠️  Database fixed but needs further testing")
    else:
        print("\n❌ Could not fix database. You may need to recreate it.")