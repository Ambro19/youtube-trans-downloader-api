# debug_database_creation.py - Debug why database creation fails

import os
import sys
import importlib

def debug_database_creation():
    """Debug the database creation process step by step"""
    try:
        print("🔍 Debugging database creation process...")
        
        # Step 1: Clear all Python cache
        print("\n🧹 Clearing Python cache...")
        
        # Remove __pycache__ directories
        for root, dirs, files in os.walk('.'):
            for dir in dirs[:]:  # Use slice to avoid modifying list during iteration
                if dir == '__pycache__':
                    import shutil
                    cache_path = os.path.join(root, dir)
                    shutil.rmtree(cache_path)
                    print(f"   Removed: {cache_path}")
        
        # Clear module cache
        modules_to_clear = [name for name in sys.modules.keys() if name.startswith('database')]
        for module in modules_to_clear:
            del sys.modules[module]
            print(f"   Cleared module: {module}")
        
        # Step 2: Fresh import of database module
        print("\n📥 Fresh import of database module...")
        import database
        importlib.reload(database)
        
        # Step 3: Verify model definition
        print("\n🔍 Verifying User model definition...")
        from database import User, Base, engine
        
        user_columns = list(User.__table__.columns.keys())
        print(f"📋 User model columns ({len(user_columns)}): {user_columns}")
        
        # Step 4: Show the actual SQL that will be generated
        print("\n🔍 Checking SQL generation...")
        from sqlalchemy import MetaData
        from sqlalchemy.schema import CreateTable
        
        # Get the CREATE TABLE SQL
        create_sql = str(CreateTable(User.__table__).compile(engine))
        print("📜 CREATE TABLE SQL:")
        print(create_sql)
        
        # Step 5: Force database deletion and recreation
        print("\n🗑️ Forcing complete database recreation...")
        db_path = "youtube_transcript.db"
        
        # Close all connections
        engine.dispose()
        
        # Delete database file
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"   Deleted: {db_path}")
        
        # Step 6: Create fresh database with verbose output
        print("\n🔨 Creating database with detailed tracking...")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("   ✅ Base.metadata.create_all() completed")
        
        # Step 7: Immediately verify what was actually created
        print("\n🔍 Verifying actual database structure...")
        
        from sqlalchemy import text
        with engine.connect() as conn:
            # Get actual table structure
            result = conn.execute(text("PRAGMA table_info(users);"))
            actual_columns = [row[1] for row in result.fetchall()]
            print(f"📋 Actual database columns ({len(actual_columns)}): {actual_columns}")
            
            # Show the discrepancy
            model_columns = list(User.__table__.columns.keys())
            missing_in_db = [col for col in model_columns if col not in actual_columns]
            extra_in_db = [col for col in actual_columns if col not in model_columns]
            
            if missing_in_db:
                print(f"❌ Missing from database: {missing_in_db}")
            if extra_in_db:
                print(f"⚠️  Extra in database: {extra_in_db}")
            
            if not missing_in_db and not extra_in_db:
                print("✅ Database structure matches model perfectly!")
                return True
            else:
                print("❌ Database structure does not match model!")
                
                # Step 8: Try manual column addition
                print("\n🔧 Attempting manual column addition...")
                for col_name in missing_in_db:
                    col_obj = User.__table__.columns[col_name]
                    col_type = str(col_obj.type)
                    
                    try:
                        if col_obj.nullable:
                            add_sql = f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"
                        else:
                            # For non-nullable columns, add with default
                            if col_obj.default:
                                default_val = col_obj.default.arg
                                add_sql = f"ALTER TABLE users ADD COLUMN {col_name} {col_type} DEFAULT {default_val}"
                            else:
                                add_sql = f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"
                        
                        print(f"   Executing: {add_sql}")
                        conn.execute(text(add_sql))
                        conn.commit()
                        print(f"   ✅ Added column: {col_name}")
                        
                    except Exception as col_error:
                        print(f"   ❌ Failed to add {col_name}: {str(col_error)}")
                
                # Verify again
                result = conn.execute(text("PRAGMA table_info(users);"))
                final_columns = [row[1] for row in result.fetchall()]
                print(f"\n📋 Final database columns ({len(final_columns)}): {final_columns}")
                
                final_missing = [col for col in model_columns if col not in final_columns]
                if not final_missing:
                    print("✅ All columns successfully added!")
                    return True
                else:
                    print(f"❌ Still missing: {final_missing}")
                    return False
                
    except Exception as e:
        print(f"💥 Critical error during debug: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting advanced database creation debug...")
    success = debug_database_creation()
    
    if success:
        print("\n🎉 Database creation successful!")
        print("🚀 Ready to test login!")
    else:
        print("\n❌ Database creation still failing.")
        print("🔍 Check the SQL output above for clues.")