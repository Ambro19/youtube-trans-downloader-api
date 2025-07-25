from sqlalchemy import create_engine, text

# CHANGE THIS if your DB file has a different name or path
engine = create_engine("sqlite:///./youtube_trans_downloader.db")

with engine.connect() as conn:
    try:
        conn.execute(text(
            "ALTER TABLE users ADD COLUMN usage_reset_date DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL"
        ))
        print("✅ Column 'usage_reset_date' added to users table.")
    except Exception as e:
        print("⚠️  Column may already exist or error occurred:", e)
