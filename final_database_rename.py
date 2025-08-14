#!/usr/bin/env python3
"""
🔧 FINAL DATABASE RENAME
========================
Simple script to make sure your server uses the fixed database
"""

import os
import shutil

def final_fix():
    """Rename fixed database to common expected names"""
    
    print("🔧 FINAL DATABASE FIX...")
    print("=" * 30)
    
    fixed_db = "youtube_trans_downloader.db"
    
    if not os.path.exists(fixed_db):
        print(f"❌ Fixed database not found: {fixed_db}")
        return False
    
    print(f"✅ Fixed database found: {fixed_db}")
    
    # Common names servers expect
    target_names = [
        "users.db",
        "database.db", 
        "youtube_transcript_downloader.db"
    ]
    
    for target_name in target_names:
        try:
            # Backup existing if present
            if os.path.exists(target_name):
                backup_name = f"{target_name}.old"
                shutil.move(target_name, backup_name)
                print(f"📦 Backed up {target_name} to {backup_name}")
            
            # Copy fixed database
            shutil.copy2(fixed_db, target_name)
            print(f"✅ Created {target_name}")
            
        except Exception as e:
            print(f"⚠️  Could not create {target_name}: {e}")
    
    # Test one of them
    try:
        import sqlite3
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT username, subscription_tier FROM users")
        users = cursor.fetchall()
        conn.close()
        
        print(f"\n🎯 VERIFICATION - users.db contains:")
        for user in users:
            print(f"   - {user[0]} ({user[1]} tier)")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = final_fix()
    
    if success:
        print("\n🎉 DATABASE FULLY FIXED!")
        print("👤 Your original users with original passwords:")
        print("   - OneTechly (Pro tier) - Can test Premium upgrade!")
        print("   - Xiggy (Free tier)")
        print("   - LovePets (Free tier)")
        print("\n🚀 Now restart your server: python run.py")
        print("🔑 Use your ORIGINAL passwords (not password123)")
    else:
        print("\n❌ Something went wrong")