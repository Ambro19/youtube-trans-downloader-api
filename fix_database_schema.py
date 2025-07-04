#!/usr/bin/env python3
"""
Fix database schema - Add missing columns to existing database
"""

import sqlite3
from pathlib import Path

def fix_database_schema():
    """Add missing columns to existing database"""
    db_path = "youtube_trans_downloader.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found at {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🔧 Fixing database schema...")
        
        # Check existing columns first
        cursor.execute("PRAGMA table_info(subscriptions);")
        existing_columns = [column[1] for column in cursor.fetchall()]
        print(f"📋 Existing columns: {existing_columns}")
        
        # List of columns that should exist
        required_columns = [
            ('stripe_subscription_id', 'VARCHAR(255)'),
            ('stripe_price_id', 'VARCHAR(255)'),
            ('status', 'VARCHAR(50) DEFAULT "active"'),
            ('current_period_start', 'DATETIME'),
            ('current_period_end', 'DATETIME'),
            ('cancel_at_period_end', 'BOOLEAN DEFAULT 0')
        ]
        
        # Add missing columns
        for column_name, column_type in required_columns:
            if column_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE subscriptions ADD COLUMN {column_name} {column_type};")
                    print(f"✅ Added column: {column_name}")
                except Exception as e:
                    print(f"⚠️ Column {column_name} might already exist: {e}")
        
        # Check if users table has stripe_customer_id
        cursor.execute("PRAGMA table_info(users);")
        user_columns = [column[1] for column in cursor.fetchall()]
        print(f"📋 User columns: {user_columns}")
        
        if 'stripe_customer_id' not in user_columns:
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN stripe_customer_id VARCHAR(255);")
                print("✅ Added stripe_customer_id to users table")
            except Exception as e:
                print(f"⚠️ stripe_customer_id might already exist: {e}")
        
        # Create PaymentHistory table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS payment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                stripe_payment_intent_id VARCHAR(255),
                stripe_customer_id VARCHAR(255),
                amount INTEGER,
                currency VARCHAR(3),
                status VARCHAR(50),
                subscription_tier VARCHAR(50),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)
        print("✅ PaymentHistory table created/verified")
        
        conn.commit()
        
        # Verify final schema
        cursor.execute("PRAGMA table_info(subscriptions);")
        final_columns = [column[1] for column in cursor.fetchall()]
        print(f"\n📋 Final subscription columns: {final_columns}")
        
        # Set default values for existing records
        cursor.execute("UPDATE subscriptions SET status = 'active' WHERE status IS NULL;")
        cursor.execute("UPDATE subscriptions SET cancel_at_period_end = 0 WHERE cancel_at_period_end IS NULL;")
        conn.commit()
        
        print("✅ Database schema fixed successfully!")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error fixing database schema: {e}")

if __name__ == "__main__":
    fix_database_schema()