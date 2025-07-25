#init_database.py
# init_database.py - Simple database initialization script

import os
from database import create_tables, engine
from sqlalchemy import text

def initialize_database():
    """
    Initialize the database with all required tables
    This creates a fresh database with all the enhanced fields
    """
    try:
        print("🔄 Initializing YouTube Transcript Downloader database...")
        
        # Check if database file exists
        db_path = "youtube_transcript.db"
        if os.path.exists(db_path):
            print(f"📁 Found existing database: {db_path}")
            
            # Check if tables exist
            with engine.connect() as conn:
                try:
                    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
                    tables = [row[0] for row in result.fetchall()]
                    print(f"📋 Existing tables: {tables}")
                    
                    if 'users' in tables:
                        # Check if users table has new columns
                        user_info = conn.execute(text("PRAGMA table_info(users);"))
                        user_columns = [row[1] for row in user_info.fetchall()]
                        print(f"👤 User table columns: {user_columns}")
                        
                        if 'stripe_customer_id' not in user_columns:
                            print("⚠️  Database needs to be updated with new Stripe fields")
                            print("🔄 Recreating database with enhanced structure...")
                            # Remove old database
                            conn.close()
                            os.remove(db_path)
                            print("🗑️  Old database removed")
                        else:
                            print("✅ Database is already up to date!")
                            return True
                except Exception as e:
                    print(f"📝 Database check failed: {e}")
        else:
            print("📄 No existing database found - creating new one")
        
        # Create fresh database with all tables
        success = create_tables()
        
        if success:
            print("🎉 Database initialization completed successfully!")
            
            # Verify tables were created
            with engine.connect() as conn:
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
                tables = [row[0] for row in result.fetchall()]
                print(f"✅ Created tables: {tables}")
                
                # Show users table structure
                if 'users' in tables:
                    user_info = conn.execute(text("PRAGMA table_info(users);"))
                    print("👤 Users table structure:")
                    for row in user_info.fetchall():
                        print(f"   - {row[1]} ({row[2]})")
                
                # Show payment_history table structure
                if 'payment_history' in tables:
                    payment_info = conn.execute(text("PRAGMA table_info(payment_history);"))
                    print("💳 Payment history table structure:")
                    for row in payment_info.fetchall():
                        print(f"   - {row[1]} ({row[2]})")
            
            return True
        else:
            print("❌ Database initialization failed!")
            return False
            
    except Exception as e:
        print(f"💥 Critical error during initialization: {str(e)}")
        return False

if __name__ == "__main__":
    success = initialize_database()
    if success:
        print("\n🚀 Ready to start the application!")
        print("Run: python -m uvicorn main:app --reload --port 8000")
    else:
        print("\n❌ Database initialization failed. Please check the errors above.")