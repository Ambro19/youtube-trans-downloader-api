# fix_subscriptions_table.py
"""
One-off SQLite migration: add missing columns on subscriptions table.
Safe to re-run; only adds columns that are missing.
"""

from models import engine, initialize_database

def add_column(conn, table, ddl):
    # ddl example: "created_at DATETIME"
    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {ddl}")

def run():
    initialize_database()

    with engine.begin() as conn:
        cols = conn.exec_driver_sql("PRAGMA table_info(subscriptions)").fetchall()
        names = {c[1] for c in cols}

        added = []

        # Core columns your code expects
        if "created_at" not in names:
            add_column(conn, "subscriptions", "created_at DATETIME")
            added.append("created_at")
        if "updated_at" not in names:
            add_column(conn, "subscriptions", "updated_at DATETIME")
            added.append("updated_at")
        if "cancelled_at" not in names:
            add_column(conn, "subscriptions", "cancelled_at DATETIME")
            added.append("cancelled_at")
        if "expires_at" not in names:
            add_column(conn, "subscriptions", "expires_at DATETIME")
            added.append("expires_at")
        if "extra_data" not in names:
            add_column(conn, "subscriptions", "extra_data TEXT")
            added.append("extra_data")

        # Optional but nice to have
        if "stripe_subscription_id" not in names:
            add_column(conn, "subscriptions", "stripe_subscription_id TEXT")
            added.append("stripe_subscription_id")
        if "stripe_customer_id" not in names:
            add_column(conn, "subscriptions", "stripe_customer_id TEXT")
            added.append("stripe_customer_id")

    if added:
        print("✅ Added columns to subscriptions:", ", ".join(added))
    else:
        print("✅ Subscriptions table already has all required columns.")

if __name__ == "__main__":
    run()
