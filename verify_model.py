# verify_model.py - Check what User model columns are defined

def verify_user_model():
    """Check what columns are defined in the User model"""
    try:
        from database import User
        
        print("🔍 Checking User model definition...")
        
        # Get all column names from the User model
        user_columns = []
        for column_name, column_obj in User.__table__.columns.items():
            user_columns.append(column_name)
        
        print(f"📋 User model has {len(user_columns)} columns:")
        for i, col in enumerate(user_columns, 1):
            print(f"   {i}. {col}")
        
        # Expected columns
        expected_columns = [
            'id', 'username', 'email', 'hashed_password', 'created_at',
            'stripe_customer_id', 'stripe_subscription_id', 'full_name',
            'phone_number', 'is_active', 'email_verified', 'last_login'
        ]
        
        print(f"\n📋 Expected {len(expected_columns)} columns:")
        for i, col in enumerate(expected_columns, 1):
            print(f"   {i}. {col}")
        
        # Find missing columns
        missing = [col for col in expected_columns if col not in user_columns]
        extra = [col for col in user_columns if col not in expected_columns]
        
        if missing:
            print(f"\n❌ Missing columns ({len(missing)}):")
            for col in missing:
                print(f"   - {col}")
        
        if extra:
            print(f"\n⚠️  Extra columns ({len(extra)}):")
            for col in extra:
                print(f"   + {col}")
        
        if not missing and not extra:
            print("\n✅ User model is complete and correct!")
            return True
        else:
            print(f"\n❌ User model needs to be updated!")
            return False
            
    except Exception as e:
        print(f"💥 Error checking User model: {str(e)}")
        return False

if __name__ == "__main__":
    verify_user_model()