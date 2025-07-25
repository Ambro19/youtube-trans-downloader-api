#!/usr/bin/env python3
"""
Fix OneTechly's subscription status - they paid $9.99 but subscription isn't working
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

def fix_onetechly_subscription():
    """Fix OneTechly's subscription status"""
    db_path = "youtube_trans_downloader.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found at {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🔧 Fixing OneTechly's subscription...")
        
        # Find OneTechly's user ID
        cursor.execute("SELECT id, username, email FROM users WHERE email = ?", ("onetechly@gmail.com",))
        user = cursor.fetchone()
        
        if not user:
            print("❌ OneTechly user not found!")
            return
        
        user_id, username, email = user
        print(f"✅ Found user: {username} (ID: {user_id}, Email: {email})")
        
        # Check existing subscription
        cursor.execute("SELECT * FROM subscriptions WHERE user_id = ?", (user_id,))
        existing_sub = cursor.fetchone()
        
        if existing_sub:
            print(f"📋 Found existing subscription: {existing_sub}")
            
            # Update existing subscription to Pro
            cursor.execute("""
                UPDATE subscriptions 
                SET tier = 'pro',
                    start_date = ?,
                    expiry_date = ?,
                    auto_renew = 1,
                    payment_id = 'manual_fix_pro_plan'
                WHERE user_id = ?
            """, (
                datetime.now().isoformat(),
                (datetime.now() + timedelta(days=30)).isoformat(),
                user_id
            ))
            print("✅ Updated existing subscription to Pro")
        else:
            # Create new Pro subscription
            cursor.execute("""
                INSERT INTO subscriptions (user_id, tier, start_date, expiry_date, payment_id, auto_renew)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                'pro',
                datetime.now().isoformat(),
                (datetime.now() + timedelta(days=30)).isoformat(),
                'manual_fix_pro_plan',
                1
            ))
            print("✅ Created new Pro subscription")
        
        # Set additional columns if they exist
        try:
            cursor.execute("""
                UPDATE subscriptions 
                SET status = 'active',
                    current_period_start = ?,
                    current_period_end = ?,
                    cancel_at_period_end = 0
                WHERE user_id = ?
            """, (
                datetime.now().isoformat(),
                (datetime.now() + timedelta(days=30)).isoformat(),
                user_id
            ))
            print("✅ Set additional subscription fields")
        except Exception as e:
            print(f"⚠️ Some additional fields not available: {e}")
        
        # Update user's Stripe customer ID (from the Stripe dashboard image)
        cursor.execute("""
            UPDATE users 
            SET stripe_customer_id = ?
            WHERE id = ?
        """, ("cus_SbvGVk6j4YARC", user_id))
        print("✅ Updated Stripe customer ID")
        
        conn.commit()
        
        # Verify the fix
        cursor.execute("""
            SELECT u.username, u.email, s.tier, s.start_date, s.expiry_date, s.payment_id
            FROM users u
            LEFT JOIN subscriptions s ON u.id = s.user_id
            WHERE u.id = ?
        """, (user_id,))
        
        result = cursor.fetchone()
        if result:
            username, email, tier, start_date, expiry_date, payment_id = result
            print(f"\n🎉 OneTechly's subscription fixed!")
            print(f"   Username: {username}")
            print(f"   Email: {email}")
            print(f"   Tier: {tier}")
            print(f"   Start Date: {start_date}")
            print(f"   Expiry Date: {expiry_date}")
            print(f"   Payment ID: {payment_id}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error fixing OneTechly's subscription: {e}")

def fix_all_paid_users():
    """Fix all users who might have payment issues"""
    db_path = "youtube_trans_downloader.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("\n🔧 Checking all users for subscription issues...")
        
        # Get all users
        cursor.execute("SELECT id, username, email FROM users")
        users = cursor.fetchall()
        
        for user_id, username, email in users:
            # Check if they have a subscription
            cursor.execute("SELECT tier, expiry_date FROM subscriptions WHERE user_id = ?", (user_id,))
            subscription = cursor.fetchone()
            
            if not subscription:
                # Create default free subscription
                cursor.execute("""
                    INSERT INTO subscriptions (user_id, tier, start_date, expiry_date, payment_id, auto_renew)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    'free',
                    datetime.now().isoformat(),
                    (datetime.now() + timedelta(days=365)).isoformat(),  # Free for 1 year
                    'default_free',
                    0
                ))
                print(f"✅ Created free subscription for {username}")
        
        conn.commit()
        conn.close()
        print("✅ All users now have subscriptions")
        
    except Exception as e:
        print(f"❌ Error fixing user subscriptions: {e}")

if __name__ == "__main__":
    fix_onetechly_subscription()
    fix_all_paid_users()