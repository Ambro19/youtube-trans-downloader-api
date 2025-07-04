#!/usr/bin/env python3
"""
Fix transcript_downloads table schema - Add missing columns
"""

import sqlite3
from pathlib import Path

def fix_transcript_downloads_table():
    """Add missing columns to transcript_downloads table"""
    db_path = "youtube_trans_downloader.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found at {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🔧 Fixing transcript_downloads table schema...")
        
        # Check existing columns first
        cursor.execute("PRAGMA table_info(transcript_downloads);")
        existing_columns = [column[1] for column in cursor.fetchall()]
        print(f"📋 Existing columns: {existing_columns}")
        
        # List of columns that should exist
        required_columns = [
            ('file_size', 'INTEGER'),
            ('processing_time', 'REAL'),
            ('download_method', 'VARCHAR(50)'),
            ('quality', 'VARCHAR(20)'),
            ('language', 'VARCHAR(10)')
        ]
        
        # Add missing columns
        for column_name, column_type in required_columns:
            if column_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE transcript_downloads ADD COLUMN {column_name} {column_type};")
                    print(f"✅ Added column: {column_name}")
                except Exception as e:
                    print(f"⚠️ Column {column_name} might already exist: {e}")
        
        conn.commit()
        
        # Verify final schema
        cursor.execute("PRAGMA table_info(transcript_downloads);")
        final_columns = [column[1] for column in cursor.fetchall()]
        print(f"\n📋 Final transcript_downloads columns: {final_columns}")
        
        print("✅ TranscriptDownloads table schema fixed successfully!")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error fixing transcript_downloads table: {e}")

if __name__ == "__main__":
    fix_transcript_downloads_table()