# restore_usage.py
#from models import SessionLocal, User 
from collections import Counter
from database import initialize_database, SessionLocal
from models import User, TranscriptDownload
from sqlalchemy import func

def restore_usage():
    # ✅ ensure schema/migrations are applied
    try:
        initialize_database()
    except Exception:
        pass

    db = SessionLocal()
    try:
        users = db.query(User).all()
        for u in users:
            # recompute usage from transcript_downloads
            clean = db.query(func.count(TranscriptDownload.id)).filter(
                TranscriptDownload.user_id == u.id,
                TranscriptDownload.transcript_type == "clean_transcripts"
            ).scalar() or 0
            unclean = db.query(func.count(TranscriptDownload.id)).filter(
                TranscriptDownload.user_id == u.id,
                TranscriptDownload.transcript_type == "unclean_transcripts"
            ).scalar() or 0
            audio = db.query(func.count(TranscriptDownload.id)).filter(
                TranscriptDownload.user_id == u.id,
                TranscriptDownload.transcript_type == "audio_downloads"
            ).scalar() or 0
            video = db.query(func.count(TranscriptDownload.id)).filter(
                TranscriptDownload.user_id == u.id,
                TranscriptDownload.transcript_type == "video_downloads"
            ).scalar() or 0

            u.usage_clean_transcripts = clean
            u.usage_unclean_transcripts = unclean
            u.usage_audio_downloads = audio
            u.usage_video_downloads = video

        db.commit()
        print("✅ Usage counters restored for all users.")
    finally:
        db.close()

if __name__ == "__main__":
    restore_usage()