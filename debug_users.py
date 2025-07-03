#!/usr/bin/env python3
"""
Debug script to check and fix user database issues
"""

import sqlite3
import bcrypt
import sys
from pathlib import Path

def check_bcrypt():
    """Check if bcrypt is working properly"""
    print("🔧 Testing bcrypt functionality...")
    try:
        # Test password hashing
        password = "testpassword"
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Test password verification
        is_valid = bcrypt.checkpw(password.encode('utf-8'), hashed)
        
        print(f"✅ bcrypt is working: {is_valid}")
        return True
    except Exception as e:
        print(f"❌ bcrypt error: {e}")
        return False

def check_users_in_db():
    """Check what users exist in the database"""
    db_path = "instance/youtube_trans_downloader.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found at {db_path}")
        return
    
    print(f"🔍 Checking users in database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if users table exists and get schema
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='users';")
        schema = cursor.fetchone()
        if schema:
            print(f"📋 Users table schema: {schema[0]}")
        else:
            print("❌ Users table doesn't exist!")
            return
        
        # Get all users
        cursor.execute("SELECT id, username, email, created_at, is_active FROM users;")
        users = cursor.fetchall()
        
        print(f"\n👥 Found {len(users)} users:")
        for user in users:
            print(f"  ID: {user[0]}, Username: {user[1]}, Email: {user[2]}, Created: {user[3]}, Active: {user[4]}")
        
        # Check for the specific email
        cursor.execute("SELECT * FROM users WHERE email = ?", ("onetechly@gmail.com",))
        specific_user = cursor.fetchone()
        
        if specific_user:
            print(f"\n🎯 Found user with email 'onetechly@gmail.com':")
            print(f"   Full record: {specific_user}")
        else:
            print("\n🔍 No user found with email 'onetechly@gmail.com'")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

def test_password_verification():
    """Test password verification for existing users"""
    db_path = "instance/youtube_trans_downloader.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found at {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get user with onetechly@gmail.com
        cursor.execute("SELECT username, email, hashed_password FROM users WHERE email = ?", ("onetechly@gmail.com",))
        user = cursor.fetchone()
        
        if user:
            username, email, hashed_password = user
            print(f"\n🔐 Testing password verification for {username} ({email})")
            
            # Try common test passwords
            test_passwords = ["password", "123456", "testpassword", "onetechly", "admin"]
            
            for pwd in test_passwords:
                try:
                    is_valid = bcrypt.checkpw(pwd.encode('utf-8'), hashed_password.encode('utf-8'))
                    if is_valid:
                        print(f"✅ Password '{pwd}' is CORRECT!")
                        break
                    else:
                        print(f"❌ Password '{pwd}' is incorrect")
                except Exception as e:
                    print(f"❌ Error testing password '{pwd}': {e}")
            else:
                print("🤔 None of the test passwords worked")
        else:
            print("🔍 No user found with that email for password testing")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error during password testing: {e}")

def delete_test_users():
    """Delete any test users to start fresh"""
    db_path = "instance/youtube_trans_downloader.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found at {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Delete users with test emails
        test_emails = ["onetechly@gmail.com", "test@test.com", "admin@test.com"]
        
        for email in test_emails:
            cursor.execute("DELETE FROM users WHERE email = ?", (email,))
            if cursor.rowcount > 0:
                print(f"🗑️ Deleted user with email: {email}")
        
        conn.commit()
        conn.close()
        print("✅ Cleanup complete!")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

def create_test_user():
    """Create a test user with known credentials"""
    db_path = "instance/youtube_trans_downloader.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found at {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create test user
        username = "TestUser"
        email = "test@example.com"
        password = "password123"
        
        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        cursor.execute("""
            INSERT INTO users (username, email, hashed_password, is_active, email_verified) 
            VALUES (?, ?, ?, ?, ?)
        """, (username, email, hashed_password.decode('utf-8'), True, True))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created test user:")
        print(f"   Username: {username}")
        print(f"   Email: {email}")
        print(f"   Password: {password}")
        
    except Exception as e:
        print(f"❌ Error creating test user: {e}")

def main():
    print("🔧 YouTube Trans Downloader - User Database Debug Tool")
    print("=" * 50)
    
    # Check bcrypt
    if not check_bcrypt():
        print("❌ bcrypt is not working properly!")
        return
    
    # Check users in database
    check_users_in_db()
    
    # Test password verification
    test_password_verification()
    
    print("\n" + "=" * 50)
    print("🛠️ Troubleshooting Options:")
    print("1. Type 'cleanup' to delete all test users")
    print("2. Type 'create' to create a test user")
    print("3. Type 'quit' to exit")
    
    while True:
        choice = input("\nWhat would you like to do? ").strip().lower()
        
        if choice == 'cleanup':
            delete_test_users()
            check_users_in_db()
        elif choice == 'create':
            create_test_user()
            check_users_in_db()
        elif choice == 'quit':
            break
        else:
            print("Please type 'cleanup', 'create', or 'quit'")

if __name__ == "__main__":
    main()