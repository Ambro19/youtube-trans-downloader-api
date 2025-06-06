# backend/run_migration.py - Run database migrations

import sqlite3
import os
from pathlib import Path

def run_migration():
    """Run the subscription fields migration"""
    
    # Database path (adjust if your database is elsewhere)
    db_path = "youtube_trans_downloader.db"
    migration_path = "migrations/001_add_subscription_fields.sql"
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        print("Make sure you're in the backend directory and the database exists.")
        return False
    
    # Check if migration file exists
    if not os.path.exists(migration_path):
        print(f"❌ Migration file not found: {migration_path}")
        print("Make sure you created the migrations folder and placed the SQL file there.")
        return False
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🔄 Running subscription fields migration...")
        
        # Read and execute migration SQL
        with open(migration_path, 'r') as file:
            migration_sql = file.read()
        
        # Split by semicolon and execute each statement
        statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]
        
        for statement in statements:
            if statement:
                try:
                    cursor.execute(statement)
                    print(f"✅ Executed: {statement[:50]}...")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        print(f"⚠️  Column already exists (skipping): {statement[:50]}...")
                    else:
                        print(f"❌ Error: {e}")
                        print(f"   Statement: {statement}")
        
        # Commit changes
        conn.commit()
        
        # Verify the migration worked
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        
        subscription_columns = [
            'subscription_tier', 'subscription_status', 'subscription_id',
            'stripe_customer_id', 'usage_clean_transcripts'
        ]
        
        existing_columns = [col[1] for col in columns]
        added_columns = [col for col in subscription_columns if col in existing_columns]
        
        print(f"\n✅ Migration completed successfully!")
        print(f"📊 Added subscription columns: {', '.join(added_columns)}")
        print(f"📝 Total columns in users table: {len(existing_columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False
        
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("🚀 YouTube Transcript Downloader - Database Migration")
    print("=" * 50)
    
    # Change to backend directory if not already there
    if not os.path.exists("main.py"):
        print("⚠️  Please run this script from the backend directory")
        print("   Example: cd backend && python run_migration.py")
        exit(1)
    
    success = run_migration()
    
    if success:
        print("\n🎉 Migration completed! You can now:")
        print("   1. Update your models.py file")
        print("   2. Restart your FastAPI server")
        print("   3. Test the subscription features")
    else:
        print("\n💔 Migration failed. Please check the errors above.")