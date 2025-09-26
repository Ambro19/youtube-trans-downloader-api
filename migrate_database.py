# backend/migrate_database.py - FIXED: Use correct database name
import sqlite3
import os
from pathlib import Path

def migrate_transcript_downloads_table():
    """Add missing columns to transcript_downloads table"""
    
    # FIXED: Look for the correct database file name
    db_path = "youtube_trans_downloader.db"
    if not os.path.exists(db_path):
        db_path = "./youtube_trans_downloader.db"
    if not os.path.exists(db_path):
        db_path = Path(__file__).parent / "youtube_trans_downloader.db"
    
    print(f"🔍 Looking for database at: {os.path.abspath(db_path)}")
    
    if not os.path.exists(db_path):
        print("❌ Database file not found. Please check the path.")
        print("💡 Expected database name: youtube_trans_downloader.db")
        
        # List available .db files in current directory
        current_dir = Path(__file__).parent
        db_files = list(current_dir.glob("*.db"))
        if db_files:
            print(f"📁 Found these database files in {current_dir}:")
            for db_file in db_files:
                print(f"   - {db_file.name}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if transcript_downloads table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transcript_downloads'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            print("⚠️ transcript_downloads table doesn't exist. Creating it...")
            # This should be handled by your models.py initialization
            print("💡 Run 'python -c \"from models import initialize_database; initialize_database()\"' first")
            conn.close()
            return False
        
        # Check current table structure
        cursor.execute("PRAGMA table_info(transcript_downloads)")
        existing_columns = {row[1]: row[2] for row in cursor.fetchall()}
        print(f"📊 Current columns: {list(existing_columns.keys())}")
        
        # Define all expected columns from models.py
        expected_columns = {
            "id": "INTEGER PRIMARY KEY",
            "user_id": "INTEGER",
            "youtube_id": "VARCHAR",
            "transcript_type": "VARCHAR",
            "quality": "VARCHAR",
            "file_format": "VARCHAR",
            "file_size": "INTEGER",
            "file_path": "VARCHAR",
            "processing_time": "FLOAT",
            "status": "VARCHAR",
            "language": "VARCHAR",
            "created_at": "DATETIME",
            "error_message": "TEXT"
        }
        
        # Add missing columns
        added_columns = []
        for column_name, column_type in expected_columns.items():
            if column_name not in existing_columns:
                try:
                    # Simplify column type for ALTER TABLE
                    simple_type = "TEXT"
                    if "INTEGER" in column_type.upper():
                        simple_type = "INTEGER"
                    elif "FLOAT" in column_type.upper() or "REAL" in column_type.upper():
                        simple_type = "REAL"
                    elif "DATETIME" in column_type.upper():
                        simple_type = "DATETIME"
                    
                    # Add default values for certain columns
                    default_value = ""
                    if column_name == "status":
                        default_value = " DEFAULT 'completed'"
                    elif column_name == "language":
                        default_value = " DEFAULT 'en'"
                    elif column_name == "created_at":
                        default_value = " DEFAULT CURRENT_TIMESTAMP"
                    
                    sql = f"ALTER TABLE transcript_downloads ADD COLUMN {column_name} {simple_type}{default_value}"
                    cursor.execute(sql)
                    added_columns.append(column_name)
                    print(f"✅ Added column: {column_name} ({simple_type})")
                except sqlite3.Error as e:
                    print(f"⚠️ Could not add column {column_name}: {e}")
        
        # Verify the final structure
        cursor.execute("PRAGMA table_info(transcript_downloads)")
        final_columns = [row[1] for row in cursor.fetchall()]
        print(f"🎯 Final columns ({len(final_columns)}): {final_columns}")
        
        conn.commit()
        conn.close()
        
        if added_columns:
            print(f"✅ Database migration completed! Added {len(added_columns)} columns: {added_columns}")
        else:
            print("✅ Database is already up to date!")
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting database migration...")
    success = migrate_transcript_downloads_table()
    if success:
        print("🎉 Migration completed successfully!")
    else:
        print("💥 Migration failed!")