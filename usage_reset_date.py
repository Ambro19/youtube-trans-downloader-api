from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///./your_database_file.db")  # change path if needed

with engine.connect() as conn:
    conn.execute(text(
        "ALTER TABLE users ADD COLUMN usage_reset_date DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL"
    ))
