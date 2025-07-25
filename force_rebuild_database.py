# force_rebuild_database.py - Force complete database recreation

# force_rebuild_database.py
from database import Base, engine
from database import TranscriptDownload  # Ensure it's imported

Base.metadata.create_all(bind=engine)
print("✅ All tables created.")

