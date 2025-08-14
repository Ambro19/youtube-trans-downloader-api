# # fix_reset_date.py - Fix the missing usage_reset_date column

# import sqlite3
# from datetime import datetime

# def fix_usage_reset_date():
#     """Add the missing usage_reset_date column and set default values"""
    
#     try:
#         conn = sqlite3.connect('youtube_trans_downloader.db')
#         cursor = conn.cursor()
        
#         print("🔄 Fixing usage_reset_date column...")
        
#         # Add the column without default value first
#         try:
#             cursor.execute("ALTER TABLE users ADD COLUMN usage_reset_date DATETIME")
#             print("✅ Added usage_reset_date column")
#         except sqlite3.OperationalError as e:
#             if "duplicate column name" in str(e).lower():
#                 print("⚠️  usage_reset_date column already exists")
#             else:
#                 print(f"❌ Error adding column: {e}")
#                 return False
        
#         # Set default values for all users
#         try:
#             current_time = datetime.now().isoformat()
#             cursor.execute("""
#                 UPDATE users 
#                 SET usage_reset_date = ? 
#                 WHERE usage_reset_date IS NULL
#             """, (current_time,))
            
#             updated_rows = cursor.rowcount
#             print(f"✅ Updated {updated_rows} users with default usage_reset_date")
#         except Exception as e:
#             print(f"❌ Error updating users: {e}")
#             return False
        
#         # Create the missing index
#         try:
#             cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_usage_reset_date ON users(usage_reset_date)")
#             print("✅ Created index for usage_reset_date")
#         except Exception as e:
#             print(f"⚠️  Index creation: {e}")
        
#         # Update any NULL values for subscription fields
#         try:
#             cursor.execute("""
#                 UPDATE users SET 
#                     subscription_tier = 'free',
#                     subscription_status = 'inactive',
#                     usage_clean_transcripts = 0,
#                     usage_unclean_transcripts = 0,
#                     usage_audio_downloads = 0,
#                     usage_video_downloads = 0
#                 WHERE subscription_tier IS NULL 
#                    OR usage_clean_transcripts IS NULL
#             """)
            
#             updated_rows = cursor.rowcount
#             if updated_rows > 0:
#                 print(f"✅ Updated {updated_rows} users with default subscription values")
#             else:
#                 print("✅ All users already have proper subscription values")
#         except Exception as e:
#             print(f"⚠️  Error updating subscription defaults: {e}")
        
#         # Commit all changes
#         conn.commit()
        
#         # Verify the fix worked
#         cursor.execute("PRAGMA table_info(users)")
#         columns = [col[1] for col in cursor.fetchall()]
        
#         if 'usage_reset_date' in columns:
#             print("✅ usage_reset_date column verified!")
#         else:
#             print("❌ usage_reset_date column still missing!")
#             return False
        
#         # Check how many users we have
#         cursor.execute("SELECT COUNT(*) FROM users")
#         user_count = cursor.fetchone()[0]
#         print(f"📊 Total users in database: {user_count}")
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Fix failed: {e}")
#         return False
        
#     finally:
#         if conn:
#             conn.close()

# if __name__ == "__main__":
#     print("🔧 YouTube Transcript Downloader - Fix usage_reset_date")
#     print("=" * 50)
    
#     success = fix_usage_reset_date()
    
#     if success:
#         print("\n🎉 Fix completed successfully!")
#         print("   The database is now ready for the subscription system.")
#     else:
#         print("\n💔 Fix failed. Please check the errors above.")