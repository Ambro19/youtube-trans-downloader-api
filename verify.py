# verify.py - Verify database migration

import sqlite3

try:
    conn = sqlite3.connect('youtube_trans_downloader.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]
    
    print("🔍 All columns in users table:")
    for i, col in enumerate(columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Check for subscription columns specifically
    subscription_columns = [
        'subscription_tier', 'subscription_status', 'stripe_customer_id',
        'usage_clean_transcripts', 'usage_unclean_transcripts'
    ]
    
    print(f"\n✅ Subscription columns check:")
    for col in subscription_columns:
        if col in columns:
            print(f"  ✅ {col} - Found")
        else:
            print(f"  ❌ {col} - Missing")
    
    conn.close()
    print(f"\n📊 Total columns: {len(columns)}")
    
except FileNotFoundError:
    print("❌ Database file 'youtube_trans_downloader.db' not found!")
    print("   Make sure you're running this from the backend directory.")
except Exception as e:
    print(f"❌ Error: {e}")