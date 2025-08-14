# #!/usr/bin/env python3
# """
# 🔍 CHECK AND RESTORE USERS SCRIPT
# =================================
# This script will:
# 1. Check both database files for existing users
# 2. Restore users if they exist in the other database
# 3. Create test users if none exist

# Run this to restore your test users.
# """

# import sqlite3
# import os
# from pathlib import Path
# from datetime import datetime

# def check_database_users(db_path):
#     """Check what users exist in a database"""
#     try:
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()
        
#         # Check if users table exists
#         cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
#         if not cursor.fetchone():
#             print(f"   ❌ No 'users' table in {db_path}")
#             conn.close()
#             return []
        
#         # Get all users
#         cursor.execute("SELECT id, username, email, created_at, subscription_tier FROM users")
#         users = cursor.fetchall()
#         conn.close()
        
#         print(f"   📊 {len(users)} users found in {db_path}")
#         for user in users:
#             print(f"      - ID: {user[0]}, Username: {user[1]}, Email: {user[2]}, Tier: {user[4] if user[4] else 'free'}")
        
#         return users
        
#     except Exception as e:
#         print(f"   ❌ Error checking {db_path}: {e}")
#         return []

# def copy_users_between_databases(source_db, target_db):
#     """Copy users from source database to target database"""
#     try:
#         # Get users from source
#         source_conn = sqlite3.connect(source_db)
#         source_cursor = source_conn.cursor()
        
#         source_cursor.execute("""
#             SELECT username, email, hashed_password, created_at, subscription_tier,
#                    stripe_customer_id, usage_clean_transcripts, usage_unclean_transcripts,
#                    usage_audio_downloads, usage_video_downloads, usage_reset_date
#             FROM users
#         """)
#         users_data = source_cursor.fetchall()
#         source_conn.close()
        
#         if not users_data:
#             print(f"   ⚠️  No users to copy from {source_db}")
#             return False
        
#         # Insert into target
#         target_conn = sqlite3.connect(target_db)
#         target_cursor = target_conn.cursor()
        
#         for user_data in users_data:
#             try:
#                 target_cursor.execute("""
#                     INSERT INTO users (
#                         username, email, hashed_password, created_at, subscription_tier,
#                         stripe_customer_id, usage_clean_transcripts, usage_unclean_transcripts,
#                         usage_audio_downloads, usage_video_downloads, usage_reset_date,
#                         is_active, is_verified
#                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0)
#                 """, user_data)
                
#                 print(f"   ✅ Copied user: {user_data[0]}")
                
#             except sqlite3.IntegrityError:
#                 print(f"   ⚠️  User {user_data[0]} already exists in target")
        
#         target_conn.commit()
#         target_conn.close()
        
#         print(f"   ✅ Successfully copied users from {source_db} to {target_db}")
#         return True
        
#     except Exception as e:
#         print(f"   ❌ Error copying users: {e}")
#         return False

# def create_test_users(db_path):
#     """Create the standard test users"""
#     try:
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()
        
#         # Import password hashing
#         from passlib.context import CryptContext
#         pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
#         test_users = [
#             {
#                 "username": "Xiggy",
#                 "email": "xiggy@example.com", 
#                 "password": "password123",
#                 "tier": "free"
#             },
#             {
#                 "username": "OneTechly",
#                 "email": "onetechtly@gmail.com",
#                 "password": "password123", 
#                 "tier": "free"
#             },
#             {
#                 "username": "LovePets",
#                 "email": "lovepets@gmail.com",
#                 "password": "password123",
#                 "tier": "pro"  # Set to pro so they can test premium upgrade
#             }
#         ]
        
#         created_users = []
#         for user_data in test_users:
#             try:
#                 hashed_password = pwd_context.hash(user_data["password"])
#                 current_time = datetime.utcnow()
                
#                 cursor.execute("""
#                     INSERT INTO users (
#                         username, email, hashed_password, created_at, subscription_tier,
#                         is_active, is_verified, usage_clean_transcripts, usage_unclean_transcripts,
#                         usage_audio_downloads, usage_video_downloads
#                     ) VALUES (?, ?, ?, ?, ?, 1, 0, 0, 0, 0, 0)
#                 """, (
#                     user_data["username"],
#                     user_data["email"], 
#                     hashed_password,
#                     current_time,
#                     user_data["tier"]
#                 ))
                
#                 created_users.append(user_data["username"])
#                 print(f"   ✅ Created user: {user_data['username']} ({user_data['tier']} tier)")
                
#             except sqlite3.IntegrityError:
#                 print(f"   ⚠️  User {user_data['username']} already exists")
        
#         conn.commit()
#         conn.close()
        
#         if created_users:
#             print(f"   🎉 Successfully created {len(created_users)} test users!")
#             print(f"   🔑 Default password for all users: 'password123'")
#             return True
#         else:
#             print(f"   ℹ️  All test users already existed")
#             return True
            
#     except Exception as e:
#         print(f"   ❌ Error creating test users: {e}")
#         return False

# def main():
#     print("🔍 CHECKING AND RESTORING USERS...")
#     print("=" * 50)
    
#     # Find database files
#     db_files = [
#         "youtube_transcript_downloader.db",
#         "youtube_trans_downloader.db"
#     ]
    
#     existing_dbs = []
#     for db_file in db_files:
#         if os.path.exists(db_file):
#             existing_dbs.append(db_file)
    
#     if not existing_dbs:
#         print("❌ No database files found!")
#         return
    
#     print(f"📂 Found database files: {existing_dbs}")
#     print()
    
#     # Check users in each database
#     db_users = {}
#     for db_path in existing_dbs:
#         print(f"🔍 Checking {db_path}:")
#         users = check_database_users(db_path)
#         db_users[db_path] = users
#         print()
    
#     # Determine which database to use as primary
#     main_db = "youtube_transcript_downloader.db"  # This is the fixed one
    
#     # Check if main database has users
#     if not db_users.get(main_db, []):
#         print(f"📋 Main database ({main_db}) has no users")
        
#         # Check if other database has users to copy
#         other_dbs = [db for db in existing_dbs if db != main_db]
#         users_copied = False
        
#         for other_db in other_dbs:
#             if db_users.get(other_db, []):
#                 print(f"🔄 Copying users from {other_db} to {main_db}...")
#                 if copy_users_between_databases(other_db, main_db):
#                     users_copied = True
#                     break
        
#         if not users_copied:
#             print(f"🆕 Creating new test users in {main_db}...")
#             create_test_users(main_db)
#     else:
#         print(f"✅ Main database ({main_db}) already has users!")
    
#     # Final verification
#     print("\n🧪 FINAL VERIFICATION:")
#     print("=" * 30)
#     final_users = check_database_users(main_db)
    
#     if final_users:
#         print("\n🎉 SUCCESS! Users are now available:")
#         print("👤 You can login with:")
#         print("   Username: Xiggy | Password: password123")
#         print("   Username: OneTechly | Password: password123")
#         print("   Username: LovePets | Password: password123")
#         print("\n🚀 Now restart your server: python run.py")
#     else:
#         print("\n❌ Something went wrong. Users still not found.")

# if __name__ == "__main__":
#     main()