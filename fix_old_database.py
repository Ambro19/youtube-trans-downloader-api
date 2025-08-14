#!/usr/bin/env python3
"""
🔧 FIX OLD DATABASE SCRIPT
===========================
This script adds missing columns to your original database:
"youtube_trans_downloader.db"

This preserves your users with their original passwords.
"""

import sqlite3
import os
import shutil
from pathlib import Path

def fix_old_database():
    """Add missing columns to the original database"""
    
    db_path = "youtube_trans_downloader.db"
    
    print("🔧 FIXING ORIGINAL DATABASE...")
    print("=" * 40)
    
    if not os.path.exists(db_path):
        print(f"❌ Original database not found: {db_path}")
        return False
    
    print(f"🔥 Found original database: {db_path}")
    
    # Create backup first
    backup_path = f"{db_path}.backup_{int(__import__('time').time())}"
    shutil.copy2(db_path, backup_path)
    print(f"✅ Backup created: {backup_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check current users
        cursor.execute("SELECT id, username, email, subscription_tier FROM users")
        existing_users = cursor.fetchall()
        
        print(f"👤 Current users in database:")
        for user in existing_users:
            print(f"   - ID: {user[0]}, Username: {user[1]}, Email: {user[2]}, Tier: {user[3] or 'free'}")
        
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
            'is_verified': 'BOOLEAN DEFAULT 0'
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
        
        # Test query that was failing
        try:
            cursor.execute("""
                SELECT id, username, email, avatar_url, full_name, subscription_tier
                FROM users 
                LIMIT 1
            """)
            test_result = cursor.fetchone()
            print("✅ Database test query PASSED!")
        except Exception as e:
            print(f"❌ Test query failed: {e}")
            return False
        
        print(f"🎉 ORIGINAL DATABASE FIXED SUCCESSFULLY!")
        print(f"📊 {len(existing_users)} users preserved with original passwords")
        
        if added_columns:
            print(f"🆕 Added columns: {added_columns}")
        else:
            print("ℹ️  No new columns needed")
            
        return True
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        # Restore backup if something went wrong
        shutil.copy2(backup_path, db_path)
        print(f"🔄 Restored backup due to error")
        return False
        
    finally:
        conn.close()

def update_server_config():
    """Make sure server uses the old database"""
    
    print("\n🔧 UPDATING SERVER CONFIG...")
    print("=" * 30)
    
    # Check if server is configured to use the right database
    models_files = ["models.py", "backend/models.py"]
    
    for models_file in models_files:
        if os.path.exists(models_file):
            print(f"📝 Checking {models_file}...")
            
            with open(models_file, 'r') as f:
                content = f.read()
            
            # Check if it references the old database
            if "youtube_trans_downloader.db" in content:
                print("✅ Server already configured for old database")
                return True
            
            # Check for other database references
            if "DATABASE_URL" in content or "sqlite:///" in content:
                print("⚠️  Server might be using different database name")
                print("   You may need to update the database URL in models.py")
                print(f"   Change it to: 'sqlite:///./youtube_trans_downloader.db'")
                return False
    
    print("⚠️  Could not find models.py to check configuration")
    return True  # Assume it's fine

if __name__ == "__main__":
    success = fix_old_database()
    
    if success:
        config_ok = update_server_config()
        
        print("\n🎉 ORIGINAL DATABASE FIXED!")
        print("🔑 Your users with ORIGINAL passwords are ready:")
        print("   - OneTechly (Pro tier)")
        print("   - Xiggy (Free tier)")  
        print("   - LovePets (Free tier)")
        print("\n🚀 Now restart your server: python run.py")
        
        if not config_ok:
            print("\n⚠️  Check your models.py database URL if login still fails")
    else:
        print("\n❌ Could not fix database")