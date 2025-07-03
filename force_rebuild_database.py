# force_rebuild_database.py - Force complete database recreation

import os
from database import create_tables, engine
from sqlalchemy import text

def force_rebuild_database():
    """
    Force complete database recreation with ALL enhanced fields
    This will delete the old database and create a fresh one
    """
    try:
        print("🔄 Force rebuilding YouTube Transcript Downloader database...")
        
        # Database file path
        db_path = "youtube_transcript.db"
        
        # Step 1: Close any existing connections
        engine.dispose()
        print("🔌 Closed existing database connections")
        
        # Step 2: Delete existing database file
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"🗑️  Deleted old database: {db_path}")
        else:
            print("📄 No existing database found")
        
        # Step 3: Create fresh database with ALL tables and columns
        print("🔨 Creating fresh database with enhanced structure...")
        success = create_tables()
        
        if success:
            print("🎉 Database rebuild completed successfully!")
            
            # Step 4: Verify ALL expected columns exist
            with engine.connect() as conn:
                # Check users table structure
                user_info = conn.execute(text("PRAGMA table_info(users);"))
                user_columns = [row[1] for row in user_info.fetchall()]
                print(f"👤 Users table columns ({len(user_columns)}): {user_columns}")
                
                # Verify all expected columns exist
                expected_user_columns = [
                    'id', 'username', 'email', 'hashed_password', 'created_at',
                    'stripe_customer_id', 'stripe_subscription_id', 'full_name',
                    'phone_number', 'is_active', 'email_verified', 'last_login'
                ]
                
                missing_columns = []
                for col in expected_user_columns:
                    if col not in user_columns:
                        missing_columns.append(col)
                
                if missing_columns:
                    print(f"❌ Missing columns: {missing_columns}")
                    return False
                else:
                    print("✅ All expected user columns present!")
                
                # Check other tables
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
                tables = [row[0] for row in result.fetchall()]
                print(f"📋 Created tables: {tables}")
                
                expected_tables = ['users', 'subscriptions', 'transcript_downloads', 'payment_history']
                missing_tables = [t for t in expected_tables if t not in tables]
                
                if missing_tables:
                    print(f"❌ Missing tables: {missing_tables}")
                    return False
                else:
                    print("✅ All expected tables present!")
                
                # Show subscriptions table structure
                if 'subscriptions' in tables:
                    sub_info = conn.execute(text("PRAGMA table_info(subscriptions);"))
                    sub_columns = [row[1] for row in sub_info.fetchall()]
                    print(f"📊 Subscriptions table columns ({len(sub_columns)}): {sub_columns}")
                
                # Show payment_history table structure  
                if 'payment_history' in tables:
                    pay_info = conn.execute(text("PRAGMA table_info(payment_history);"))
                    pay_columns = [row[1] for row in pay_info.fetchall()]
                    print(f"💳 Payment history table columns ({len(pay_columns)}): {pay_columns}")
            
            return True
        else:
            print("❌ Database rebuild failed!")
            return False
            
    except Exception as e:
        print(f"💥 Critical error during rebuild: {str(e)}")
        return False

if __name__ == "__main__":
    print("⚠️  This will DELETE your existing database and all data!")
    print("⚠️  All users will need to re-register!")
    
    confirm = input("Type 'YES' to continue with complete rebuild: ")
    
    if confirm.upper() == 'YES':
        success = force_rebuild_database()
        if success:
            print("\n🚀 Database rebuild complete!")
            print("🚀 Ready to start the application!")
            print("🚀 All users need to re-register with the new database structure")
            print("\nRun: python -m uvicorn main:app --reload --port 8000")
        else:
            print("\n❌ Database rebuild failed. Please check the errors above.")
    else:
        print("❌ Rebuild cancelled.")