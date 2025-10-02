# backend/fix_existing_users.py
import os
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv()
load_dotenv(dotenv_path=find_dotenv(".env.local"), override=True)
load_dotenv(dotenv_path=find_dotenv(".env"), override=False)

from models import SessionLocal
from subscription_sync import sync_all_users_with_stripe

def main():
    # Verify Stripe is configured
    stripe_key = os.getenv("STRIPE_SECRET_KEY")
    if not stripe_key:
        print("❌ ERROR: STRIPE_SECRET_KEY not found")
        return 1
    
    print(f"✓ Stripe configured: {stripe_key[:7]}...{stripe_key[-4:]}")
    print(f"✓ Environment: {os.getenv('APP_ENV', 'development')}")
    print("\n🔧 Starting subscription sync...\n")
    
    db = SessionLocal()
    try:
        result = sync_all_users_with_stripe(db)
        
        if "error" in result:
            print(f"❌ Sync failed: {result['error']}")
            return 1
        
        if "synced" in result:
            print(f"✅ Sync complete:")
            print(f"   • Updated: {result['synced']} users")
            print(f"   • Unchanged: {result.get('unchanged', 0)} users")
            print(f"   • Errors: {result.get('errors', 0)} users")
            
            if result['synced'] > 0:
                print("\n🎉 Successfully fixed subscription statuses!")
            elif result['unchanged'] > 0:
                print("\nℹ️  All users already have correct subscription status.")
            else:
                print("\nℹ️  No users with Stripe customer IDs found.")
            
            return 0
        else:
            print(f"⚠️  Unexpected result: {result}")
            return 1
            
    except Exception as e:
        print(f"❌ Error during sync: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        db.close()

if __name__ == "__main__":
    sys.exit(main())