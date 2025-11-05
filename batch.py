# backend/batch.py
"""
Batch processing for transcripts, audio, and video downloads.
Handles multiple videos at once with job tracking and status updates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth_deps import get_current_user
from models import User, get_db
from transcript_utils import (
    download_audio_with_ytdlp,
    download_video_with_ytdlp,
    get_transcript_with_ytdlp,
    validate_video_id,
    _clean_plain_blocks,
    _get_segment_value,
)

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

logger = logging.getLogger("batch")
router = APIRouter(prefix="/batch")

# Constants
DOWNLOADS_DIR = Path(os.getenv("DOWNLOADS_DIR", "./server_files"))
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
JOB_SNAPSHOT_FILE = DOWNLOADS_DIR / "batch_jobs_snapshot.json"
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_BATCH_JOBS", "3"))
JOB_RETENTION_HOURS = 48

# Job storage
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# Priority transcript fetching
_EN_PRIORITY = ["en", "en-US", "en-GB", "en-CA", "en-AU", "en-IE", "en-NZ"]


# ============================================================================
# Models
# ============================================================================
class BatchSubmitRequest(BaseModel):
    video_ids: List[str]
    job_type: str  # 'transcript' | 'audio' | 'video'
    quality: Optional[str] = "medium"  # For audio/video
    clean_transcript: bool = True  # For transcripts


class BatchJobStatus(BaseModel):
    job_id: str
    job_type: str
    status: str
    total: int
    completed: int
    failed: int
    results: List[Dict[str, Any]]
    created_at: str
    updated_at: str
    user_id: int


# ============================================================================
# Transcript Processing with Paragraph Formatting
# ============================================================================
def _format_timestamped(segments) -> str:
    """Format segments with timestamps."""
    lines = []
    try:
        for seg in segments:
            start = _get_segment_value(seg, "start", 0)
            text = _get_segment_value(seg, "text", "")
            
            if not text:
                continue
                
            t = int(start)
            text_clean = text.replace("\n", " ")
            lines.append(f"[{t // 60:02d}:{t % 60:02d}] {text_clean}")
            
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"_format_timestamped error: {e}")
        raise


def get_formatted_transcript(video_id: str, clean: bool = True) -> str:
    """
    Get transcript with proper paragraph formatting.
    This ensures batch transcripts look professional like the main download.
    
    Args:
        video_id: YouTube video ID
        clean: If True, return clean paragraphs. If False, return timestamped.
        
    Returns:
        Formatted transcript text
        
    Raises:
        Exception: If no transcript can be obtained
    """
    if not YouTubeTranscriptApi:
        raise RuntimeError("youtube_transcript_api not available")
    
    try:
        listing = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try priority English variants
        for code in _EN_PRIORITY:
            try:
                t = listing.find_transcript([code])
                segments = t.fetch()
                
                if clean:
                    # Extract text and format into paragraphs
                    texts = [
                        _get_segment_value(s, "text", "").replace("\n", " ")
                        for s in segments
                        if _get_segment_value(s, "text", "")
                    ]
                    return _clean_plain_blocks(texts)
                else:
                    return _format_timestamped(segments)
                    
            except Exception:
                continue
        
        # Try generated/auto-captions
        try:
            t = listing.find_generated_transcript(_EN_PRIORITY)
            segments = t.fetch()
            
            if clean:
                texts = [
                    _get_segment_value(s, "text", "").replace("\n", " ")
                    for s in segments
                    if _get_segment_value(s, "text", "")
                ]
                return _clean_plain_blocks(texts)
            else:
                return _format_timestamped(segments)
                
        except Exception:
            pass
        
        # Try translation
        for t in listing:
            try:
                translated = t.translate("en")
                segments = translated.fetch()
                
                if clean:
                    texts = [
                        _get_segment_value(s, "text", "").replace("\n", " ")
                        for s in segments
                        if _get_segment_value(s, "text", "")
                    ]
                    return _clean_plain_blocks(texts)
                else:
                    return _format_timestamped(segments)
                    
            except Exception:
                continue
                
    except Exception as e:
        logger.warning(f"YouTube Transcript API failed for {video_id}: {e}")
    
    # Fallback to yt-dlp
    try:
        fb = get_transcript_with_ytdlp(video_id, clean=clean)
        if fb:
            return fb
    except Exception as e:
        logger.warning(f"yt-dlp fallback failed for {video_id}: {e}")
    
    raise Exception("No transcript/captions found for this video")


# ============================================================================
# Job Management
# ============================================================================
def _generate_job_id() -> str:
    """Generate unique job ID."""
    return uuid.uuid4().hex[:16]


def _save_snapshot():
    """Save jobs to disk for persistence."""
    try:
        with _jobs_lock:
            # Only save jobs from last 48 hours
            cutoff = time.time() - (JOB_RETENTION_HOURS * 3600)
            active_jobs = {
                jid: job for jid, job in _jobs.items()
                if job.get("created_at", 0) > cutoff
            }
            
        with open(JOB_SNAPSHOT_FILE, "w") as f:
            json.dump(active_jobs, f)
    except Exception as e:
        logger.error(f"Failed to save job snapshot: {e}")


def _load_snapshot():
    """Load jobs from disk on startup."""
    if not JOB_SNAPSHOT_FILE.exists():
        return
    
    try:
        with open(JOB_SNAPSHOT_FILE, "r") as f:
            saved_jobs = json.load(f)
        
        with _jobs_lock:
            _jobs.update(saved_jobs)
        
        logger.info(f"Restored {len(saved_jobs)} batch jobs from snapshot")
    except Exception as e:
        logger.error(f"Failed to load job snapshot: {e}")


def _cleanup_old_jobs():
    """Remove jobs older than retention period."""
    cutoff = time.time() - (JOB_RETENTION_HOURS * 3600)
    
    with _jobs_lock:
        old_ids = [
            jid for jid, job in _jobs.items()
            if job.get("created_at", 0) < cutoff
        ]
        
        for jid in old_ids:
            _jobs.pop(jid, None)
    
    if old_ids:
        logger.info(f"Cleaned up {len(old_ids)} old batch jobs")
        _save_snapshot()


# ============================================================================
# Processing Functions
# ============================================================================
def process_transcript(video_id: str, clean: bool = True) -> Dict[str, Any]:
    """Process single transcript with proper formatting."""
    start = time.time()
    
    try:
        text = get_formatted_transcript(video_id, clean=clean)
        
        return {
            "success": True,
            "video_id": video_id,
            "transcript": text,
            "duration": round(time.time() - start, 2),
            "length": len(text),
        }
    except Exception as e:
        logger.error(f"Transcript failed for {video_id}: {e}")
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e),
            "duration": round(time.time() - start, 2),
        }


def process_audio(video_id: str, quality: str) -> Dict[str, Any]:
    """Process single audio download."""
    start = time.time()
    
    try:
        # Fixed call with all 3 required parameters
        path = download_audio_with_ytdlp(video_id, quality, str(DOWNLOADS_DIR))
        
        if not path or not os.path.exists(path):
            raise Exception("Audio download failed - no file created")
        
        file_size = os.path.getsize(path)
        filename = os.path.basename(path)
        
        return {
            "success": True,
            "video_id": video_id,
            "filename": filename,
            "file_size": file_size,
            "duration": round(time.time() - start, 2),
            "download_url": f"/files/{filename}",
        }
    except Exception as e:
        logger.error(f"Audio failed for {video_id}: {e}")
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e),
            "duration": round(time.time() - start, 2),
        }


def process_video(video_id: str, quality: str) -> Dict[str, Any]:
    """Process single video download."""
    start = time.time()
    
    try:
        # Fixed call with all 3 required parameters
        path = download_video_with_ytdlp(video_id, quality, str(DOWNLOADS_DIR))
        
        if not path or not os.path.exists(path):
            raise Exception("Video download failed - no file created")
        
        file_size = os.path.getsize(path)
        filename = os.path.basename(path)
        
        return {
            "success": True,
            "video_id": video_id,
            "filename": filename,
            "file_size": file_size,
            "duration": round(time.time() - start, 2),
            "download_url": f"/files/{filename}",
        }
    except Exception as e:
        logger.error(f"Video failed for {video_id}: {e}")
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e),
            "duration": round(time.time() - start, 2),
        }


# ============================================================================
# Batch Job Processor
# ============================================================================
def process_batch_job(job_id: str):
    """Process batch job in background thread."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        
        job["status"] = "processing"
        job["updated_at"] = time.time()
    
    try:
        job_type = job["job_type"]
        video_ids = job["video_ids"]
        quality = job.get("quality", "medium")
        clean = job.get("clean_transcript", True)
        
        logger.info(f"Starting batch processing job {job_id} ({job_type}) for user {job['username']}")
        
        results = []
        for video_id in video_ids:
            try:
                if job_type == "transcript":
                    result = process_transcript(video_id, clean=clean)
                elif job_type == "audio":
                    result = process_audio(video_id, quality)
                elif job_type == "video":
                    result = process_video(video_id, quality)
                else:
                    result = {"success": False, "video_id": video_id, "error": "Unknown job type"}
                
                results.append(result)
                
                # Update progress
                with _jobs_lock:
                    job["completed"] += 1
                    if result["success"]:
                        logger.info(f"Successfully processed {job_type} for {video_id} in {result['duration']}s")
                    else:
                        job["failed"] += 1
                        logger.error(f"Processing failed for {video_id} ({job_type}): {result.get('error', 'Unknown error')}")
                    
                    job["results"] = results
                    job["updated_at"] = time.time()
                
            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}")
                results.append({
                    "success": False,
                    "video_id": video_id,
                    "error": str(e),
                })
                
                with _jobs_lock:
                    job["completed"] += 1
                    job["failed"] += 1
                    job["results"] = results
                    job["updated_at"] = time.time()
        
        # Mark as complete
        with _jobs_lock:
            job["status"] = "completed"
            job["updated_at"] = time.time()
        
        logger.info(f"Job {job_id} completed: {job['completed'] - job['failed']}/{job['total']} successful ({job_type})")
        
    except Exception as e:
        logger.error(f"Batch job {job_id} failed: {e}")
        with _jobs_lock:
            job["status"] = "failed"
            job["error"] = str(e)
            job["updated_at"] = time.time()
    
    finally:
        _save_snapshot()


# ============================================================================
# API Endpoints
# ============================================================================
@router.post("/submit")
def submit_batch_job(
    req: BatchSubmitRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Submit a new batch job."""
    # Validate video IDs
    valid_ids = [vid.strip() for vid in req.video_ids if validate_video_id(vid.strip())]
    
    if not valid_ids:
        raise HTTPException(status_code=400, detail="No valid video IDs provided")
    
    if len(valid_ids) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 videos per batch")
    
    if req.job_type not in ["transcript", "audio", "video"]:
        raise HTTPException(status_code=400, detail="Invalid job type")
    
    # Create job
    job_id = _generate_job_id()
    job = {
        "job_id": job_id,
        "user_id": user.id,
        "username": user.username,
        "job_type": req.job_type,
        "video_ids": valid_ids,
        "quality": req.quality,
        "clean_transcript": req.clean_transcript,
        "status": "queued",
        "total": len(valid_ids),
        "completed": 0,
        "failed": 0,
        "results": [],
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    
    with _jobs_lock:
        _jobs[job_id] = job
    
    logger.info(f"Created batch job {job_id} ({req.job_type}) with {len(valid_ids)} items for user {user.username}")
    
    # Start processing in background
    thread = threading.Thread(target=process_batch_job, args=(job_id,), daemon=True)
    thread.start()
    
    _save_snapshot()
    
    return {
        "job_id": job_id,
        "status": "queued",
        "total": len(valid_ids),
        "message": f"Batch job created with {len(valid_ids)} videos",
    }


@router.get("/jobs")
def list_jobs(user: User = Depends(get_current_user)):
    """List all jobs for current user."""
    with _jobs_lock:
        user_jobs = [
            {
                "job_id": job["job_id"],
                "job_type": job["job_type"],
                "status": job["status"],
                "total": job["total"],
                "completed": job["completed"],
                "failed": job["failed"],
                "created_at": datetime.fromtimestamp(job["created_at"]).isoformat(),
                "updated_at": datetime.fromtimestamp(job["updated_at"]).isoformat(),
            }
            for job in _jobs.values()
            if job["user_id"] == user.id
        ]
    
    return {"jobs": user_jobs, "total": len(user_jobs)}


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str, user: User = Depends(get_current_user)):
    """Get status of specific job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["user_id"] != user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "job_id": job["job_id"],
        "job_type": job["job_type"],
        "status": job["status"],
        "total": job["total"],
        "completed": job["completed"],
        "failed": job["failed"],
        "results": job["results"],
        "created_at": datetime.fromtimestamp(job["created_at"]).isoformat(),
        "updated_at": datetime.fromtimestamp(job["updated_at"]).isoformat(),
    }


@router.get("/view/{video_id}/{content_type}")
def view_batch_content(
    video_id: str,
    content_type: str,
    user: User = Depends(get_current_user),
):
    """View transcript content from batch job."""
    if content_type != "transcript":
        raise HTTPException(status_code=400, detail="Only transcripts can be viewed")
    
    # Find the transcript in any completed job
    with _jobs_lock:
        for job in _jobs.values():
            if job["user_id"] == user.id and job["job_type"] == "transcript":
                for result in job.get("results", []):
                    if result.get("video_id") == video_id and result.get("success"):
                        return {
                            "video_id": video_id,
                            "transcript": result.get("transcript", ""),
                            "length": result.get("length", 0),
                        }
    
    raise HTTPException(status_code=404, detail="Transcript not found")


@router.post("/jobs/{job_id}/retry_failed")
def retry_failed_items(job_id: str, user: User = Depends(get_current_user)):
    """Retry failed items in a job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["user_id"] != user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if job["failed"] == 0:
        return {"message": "No failed items to retry"}
    
    # Get failed video IDs
    failed_ids = [
        result["video_id"]
        for result in job["results"]
        if not result.get("success", False)
    ]
    
    if not failed_ids:
        return {"message": "No failed items found"}
    
    # Create new job for failed items
    new_job_id = _generate_job_id()
    new_job = {
        "job_id": new_job_id,
        "user_id": user.id,
        "username": user.username,
        "job_type": job["job_type"],
        "video_ids": failed_ids,
        "quality": job.get("quality", "medium"),
        "clean_transcript": job.get("clean_transcript", True),
        "status": "queued",
        "total": len(failed_ids),
        "completed": 0,
        "failed": 0,
        "results": [],
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    
    with _jobs_lock:
        _jobs[new_job_id] = new_job
    
    # Start processing
    thread = threading.Thread(target=process_batch_job, args=(new_job_id,), daemon=True)
    thread.start()
    
    _save_snapshot()
    
    return {
        "job_id": new_job_id,
        "retrying": len(failed_ids),
        "message": f"Created new job to retry {len(failed_ids)} failed items",
    }


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str, user: User = Depends(get_current_user)):
    """Delete a job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job["user_id"] != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        _jobs.pop(job_id)
    
    _save_snapshot()
    
    return {"message": "Job deleted successfully"}


# ============================================================================
# Startup/Cleanup
# ============================================================================
def startup():
    """Initialize batch processing."""
    _load_snapshot()
    
    # Start cleanup thread
    def cleanup_loop():
        while True:
            time.sleep(3600)  # Every hour
            _cleanup_old_jobs()
    
    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()


# Call startup when module is imported
startup()

##====================================
# # backend/batch.py
# # Enhanced Batch Jobs router with full transcript/audio/video support and usage tracking
# # - Supports all three download types: transcript, audio, video
# # - Proper usage tracking integration for all types
# # - Professional download handling with quality settings

# from __future__ import annotations

# import os
# import re
# import json
# import uuid
# import hmac
# import hashlib
# import asyncio
# import time
# from typing import List, Dict, Any, Optional
# from datetime import datetime
# from pathlib import Path

# from fastapi import APIRouter, Depends, HTTPException, Request, Body, BackgroundTasks
# from pydantic import BaseModel, Field, validator
# from sqlalchemy.orm import Session
# import jwt
# import logging

# # Optional deps (best-effort)
# try:
#     import requests  # type: ignore
# except Exception:  # pragma: no cover
#     requests = None

# try:
#     import boto3  # type: ignore
#     from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
# except Exception:  # pragma: no cover
#     boto3 = None
#     BotoCoreError = ClientError = Exception

# from models import User, get_db
# from transcript_utils import (
#     get_transcript_with_ytdlp,
#     download_audio_with_ytdlp, 
#     download_video_with_ytdlp,
#     get_video_info,
#     check_ytdlp_availability
# )

# log = logging.getLogger("batch")
# router = APIRouter(prefix="/batch", tags=["batch"])

# # ---- Auth config (match main.py/payment.py) --------------------------------
# SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
# ALGORITHM = os.getenv("ALGORITHM", "HS256")

# # ---- Downloads directory ----
# DOWNLOADS_DIR = Path(__file__).resolve().parent / "server_files"
# DOWNLOADS_DIR.mkdir(exist_ok=True)

# # ---- Caps ------------------------------------------------------------------
# PRO_MAX_LINKS = int(os.getenv("BATCH_PRO_MAX_LINKS", "3"))
# PREMIUM_MAX_LINKS = int(os.getenv("BATCH_PREMIUM_MAX_LINKS", "2000"))
# DEFAULT_MAX_LINKS = int(os.getenv("BATCH_DEFAULT_MAX_LINKS", "300"))  # safety

# # ---- Optional Integrations (Premium) ---------------------------------------
# WEBHOOK_URL = os.getenv("BATCH_WEBHOOK_URL")
# WEBHOOK_SECRET = os.getenv("BATCH_WEBHOOK_SECRET", "")
# SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
# S3_BUCKET = os.getenv("BATCH_S3_BUCKET")
# S3_PREFIX = os.getenv("BATCH_S3_PREFIX", "batch-results/")

# def _auth_user(request: Request, db: Session) -> User:
#     auth = request.headers.get("authorization") or ""
#     if not auth.startswith("Bearer "):
#         raise HTTPException(401, "Missing bearer token")
#     token = auth.split()[1]
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username = payload.get("sub")
#         if not username:
#             raise HTTPException(401, "Bad token")
#     except Exception:
#         raise HTTPException(401, "Invalid token")
#     user = db.query(User).filter(User.username == username).first()
#     if not user:
#         raise HTTPException(401, "User not found")
#     return user

# # ---- Import usage tracking functions from main.py (avoid circular imports) ----
# def increment_user_usage(db: Session, user: User, usage_type: str) -> int:
#     """Increment user usage counter (imported logic from main.py)"""
#     current = getattr(user, f"usage_{usage_type}", 0) or 0
#     new_val = current + 1
#     setattr(user, f"usage_{usage_type}", new_val)

#     now = datetime.utcnow()
#     if not getattr(user, "usage_reset_date", None):
#         user.usage_reset_date = now
#     elif user.usage_reset_date.month != now.month:
#         user.usage_clean_transcripts = 0
#         user.usage_unclean_transcripts = 0
#         user.usage_audio_downloads = 0
#         user.usage_video_downloads = 0
#         user.usage_reset_date = now
#         setattr(user, f"usage_{usage_type}", 1)
#         new_val = 1

#     try:
#         db.commit()
#         db.refresh(user)
#     except Exception as e:
#         log.error(f"Failed to increment usage for {user.username}: {e}")
#         db.rollback()
    
#     return new_val

# def check_usage_limit(user: User, usage_type: str) -> tuple[bool, int, int]:
#     """Check if user has reached usage limit (imported logic from main.py)"""
#     tier = getattr(user, "subscription_tier", "free")
#     limits = {
#         "free":   {"clean_transcripts": 5,  "unclean_transcripts": 3,  "audio_downloads": 2,  "video_downloads": 1},
#         "pro":    {"clean_transcripts": 100,"unclean_transcripts": 50, "audio_downloads": 50, "video_downloads": 20},
#         "premium":{"clean_transcripts": float("inf"), "unclean_transcripts": float("inf"),
#                    "audio_downloads": float("inf"), "video_downloads": float("inf")},
#     }
#     cur = getattr(user, f"usage_{usage_type}", 0) or 0
#     limit = limits.get(tier, limits["free"]).get(usage_type, 0)
#     return cur < limit, cur, limit

# def create_download_record(db: Session, user: User, kind: str, youtube_id: str, **kw):
#     """Create download record (imported logic from main.py)"""
#     try:
#         from models import TranscriptDownload
#         rec = TranscriptDownload(
#             user_id=user.id,
#             youtube_id=youtube_id,
#             transcript_type=kind,
#             quality=kw.get("quality", "default"),
#             file_format=kw.get("file_format", "txt"),
#             file_size=kw.get("file_size", 0),
#             processing_time=kw.get("processing_time", 0),
#             created_at=datetime.utcnow(),
#         )
#         db.add(rec)
#         db.commit()
#         db.refresh(rec)
#         return rec
#     except Exception as e:
#         log.error(f"Create download record failed: {e}")
#         db.rollback()
#         return None

# # ---- Simple ID extractor (no import from main.py to avoid circular deps) ---
# _YT_PATTERNS = [
#     re.compile(r"(?:youtube\.com\/watch\?v=)([^&\n?#]+)", re.I),
#     re.compile(r"(?:youtu\.be\/)([^&\n?#]+)", re.I),
#     re.compile(r"(?:youtube\.com\/embed\/)([^&\n?#]+)", re.I),
#     re.compile(r"(?:youtube\.com\/shorts\/)([^&\n?#]+)", re.I),
#     re.compile(r"[?&]v=([^&\n?#]+)", re.I),
# ]

# def extract_youtube_video_id(text: str) -> str:
#     s = (text or "").strip()
#     for p in _YT_PATTERNS:
#         m = p.search(s)
#         if m:
#             return m.group(1)[:11]
#     return s[:11]  # assume raw ID

# # ---- In-memory job store ----------------------------------------------------
# JOBS: Dict[str, Dict[str, Any]] = {}

# SNAPSHOT_FILE = os.getenv("BATCH_SNAPSHOT_FILE", "./.batch_jobs_snapshot.json")
# ENABLE_SNAPSHOT = os.getenv("ENVIRONMENT", "development") != "production"

# def _save_snapshot():
#     if not ENABLE_SNAPSHOT:
#         return
#     try:
#         data = {"jobs": JOBS, "saved_at": datetime.utcnow().isoformat()}
#         with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
#             json.dump(data, f, ensure_ascii=False, indent=2)
#     except Exception as e:  # pragma: no cover
#         log.debug("snapshot save skipped: %s", e)

# def _load_snapshot():
#     if not ENABLE_SNAPSHOT:
#         return
#     try:
#         if os.path.exists(SNAPSHOT_FILE):
#             with open(SNAPSHOT_FILE, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#                 JOBS.clear()
#                 JOBS.update(data.get("jobs", {}))
#                 log.info("Restored %d batch jobs from snapshot", len(JOBS))
#     except Exception as e:  # pragma: no cover
#         log.debug("snapshot load failed: %s", e)

# _load_snapshot()

# # ---- Pydantic models --------------------------------------------------------
# VALID_KINDS = {"transcript", "audio", "video"}

# class SubmitBody(BaseModel):
#     urls: Optional[List[str]] = Field(default=None, description="List of YouTube URLs or IDs")
#     csv_text: Optional[str] = Field(default=None, description="CSV or newline list of URLs/IDs")
#     kind: str = Field(default="transcript", description="transcript|audio|video")
#     quality: Optional[str] = Field(default=None, description="Quality setting for audio/video")

#     @validator("kind")
#     def _kind_ok(cls, v):
#         vv = (v or "").strip().lower()
#         if vv not in VALID_KINDS:
#             raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}")
#         return vv

#     def collect_ids(self) -> List[str]:
#         raw: List[str] = []
#         if self.urls:
#             raw.extend(self.urls)
#         if self.csv_text:
#             parts = re.split(r"[\s,;]+", self.csv_text.strip())
#             raw.extend([p for p in parts if p])
#         # de-dupe while preserving order
#         seen = set()
#         ids: List[str] = []
#         for entry in raw:
#             vid = extract_youtube_video_id(entry)
#             if len(vid) == 11 and vid not in seen:
#                 seen.add(vid)
#                 ids.append(vid)
#         # safety cap before tier rules
#         return ids[:DEFAULT_MAX_LINKS]

# class ItemOut(BaseModel):
#     idx: int
#     youtube_id: str
#     status: str
#     message: Optional[str] = None
#     result_type: str
#     result_url: Optional[str] = None
#     file_size: Optional[int] = None
#     file_size_mb: Optional[float] = None
#     video_title: Optional[str] = None
#     download_links: Optional[Dict[str, str]] = None

# class JobOut(BaseModel):
#     id: str
#     created_at: str
#     status: str
#     kind: str
#     total: int
#     completed: int
#     failed: int
#     queued: int
#     processing: int
#     items: List[ItemOut]

# # ---- Core processing functions with usage tracking -------------------------

# def _create_initial_item(idx: int, vid: str, kind: str) -> Dict[str, Any]:
#     """Create initial item with queued status for real processing."""
#     return {
#         "idx": idx,
#         "youtube_id": vid,
#         "status": "queued",
#         "message": "Waiting to process...",
#         "result_type": kind,
#         "result_url": None,
#         "file_size": None,
#         "file_size_mb": None,
#         "video_title": None,
#         "download_links": None,
#         "created_at": datetime.utcnow().isoformat(),
#     }

# def _process_item(item: Dict[str, Any], kind: str, user: User, db: Session, quality: Optional[str] = None) -> Dict[str, Any]:
#     """Process a single item with proper usage tracking integration for all types."""
#     vid = item["youtube_id"]
#     start_time = time.time()
    
#     try:
#         # Update to processing status
#         item["status"] = "processing"
#         item["message"] = f"Processing {kind}..."
        
#         # Check usage limits before processing based on kind
#         usage_type_map = {
#             "transcript": "clean_transcripts",  # Default to clean for batch
#             "audio": "audio_downloads",
#             "video": "video_downloads"
#         }
        
#         usage_type = usage_type_map.get(kind, "clean_transcripts")
#         can_process, current, limit = check_usage_limit(user, usage_type)
        
#         if not can_process:
#             item["status"] = "failed"
#             item["message"] = f"Monthly limit reached ({current}/{limit})"
#             return item
        
#         # Get video info first
#         video_info = get_video_info(vid)
#         if video_info:
#             item["video_title"] = video_info.get("title", "Unknown Title")
        
#         processing_time = 0
        
#         if kind == "transcript":
#             # Process transcript
#             transcript = get_transcript_with_ytdlp(vid, clean=True)
#             if transcript:
#                 item["status"] = "completed"
#                 item["message"] = "Transcript ready"
#                 item["file_size"] = len(transcript.encode('utf-8'))
#                 item["file_size_mb"] = round(item["file_size"] / (1024 * 1024), 2)
#                 item["download_links"] = {
#                     "transcript": f"/batch/download/{vid}/transcript",
#                     "view": f"/batch/view/{vid}/transcript"
#                 }
#                 # Store transcript content for download
#                 _store_transcript_result(vid, transcript)
                
#                 # Track usage and create record
#                 processing_time = time.time() - start_time
#                 increment_user_usage(db, user, usage_type)
#                 create_download_record(
#                     db=db, user=user, kind=usage_type, youtube_id=vid,
#                     file_format="txt", file_size=item["file_size"], 
#                     processing_time=processing_time
#                 )
                
#             else:
#                 raise Exception("No transcript available")
                
#         elif kind == "audio":
#             # Process audio download
#             if not check_ytdlp_availability():
#                 raise Exception("Audio processing unavailable")
            
#             audio_quality = quality or "medium"
#             file_path = download_audio_with_ytdlp(vid, audio_quality, str(DOWNLOADS_DIR))
#             if file_path and os.path.exists(file_path):
#                 file_size = os.path.getsize(file_path)
#                 filename = os.path.basename(file_path)
                
#                 item["status"] = "completed"
#                 item["message"] = "Audio ready for download"
#                 item["result_url"] = f"/files/{filename}"
#                 item["file_size"] = file_size
#                 item["file_size_mb"] = round(file_size / (1024 * 1024), 2)
#                 item["download_links"] = {
#                     "audio": f"/files/{filename}",
#                     "direct": f"/download-file/audio/{filename}"
#                 }
                
#                 # Track usage and create record
#                 processing_time = time.time() - start_time
#                 increment_user_usage(db, user, usage_type)
#                 create_download_record(
#                     db=db, user=user, kind=usage_type, youtube_id=vid,
#                     quality=audio_quality, file_format="mp3", file_size=file_size,
#                     processing_time=processing_time
#                 )
                
#             else:
#                 raise Exception("Audio download failed")
                
#         elif kind == "video":
#             # Process video download
#             if not check_ytdlp_availability():
#                 raise Exception("Video processing unavailable")
                
#             video_quality = quality or "720p"
#             file_path = download_video_with_ytdlp(vid, video_quality, str(DOWNLOADS_DIR))
#             if file_path and os.path.exists(file_path):
#                 file_size = os.path.getsize(file_path)
#                 filename = os.path.basename(file_path)
                
#                 item["status"] = "completed"
#                 item["message"] = "Video ready for download"
#                 item["result_url"] = f"/files/{filename}"
#                 item["file_size"] = file_size
#                 item["file_size_mb"] = round(file_size / (1024 * 1024), 2)
#                 item["download_links"] = {
#                     "video": f"/files/{filename}",
#                     "direct": f"/download-file/video/{filename}"
#                 }
                
#                 # Track usage and create record
#                 processing_time = time.time() - start_time
#                 increment_user_usage(db, user, usage_type)
#                 create_download_record(
#                     db=db, user=user, kind=usage_type, youtube_id=vid,
#                     quality=video_quality, file_format="mp4", file_size=file_size,
#                     processing_time=processing_time
#                 )
                
#             else:
#                 raise Exception("Video download failed")
        
#         item["processed_at"] = datetime.utcnow().isoformat()
#         log.info(f"Successfully processed {kind} for {vid} in {processing_time:.2f}s")
        
#     except Exception as e:
#         log.error(f"Processing failed for {vid} ({kind}): {e}")
#         item["status"] = "failed"
#         item["message"] = str(e)
#         item["result_url"] = None
#         item["download_links"] = None
#         item["failed_at"] = datetime.utcnow().isoformat()
    
#     return item

# def _store_transcript_result(vid: str, transcript: str):
#     """Store transcript content for later download."""
#     transcript_file = DOWNLOADS_DIR / f"{vid}_transcript.txt"
#     try:
#         transcript_file.write_text(transcript, encoding='utf-8')
#         log.debug(f"Stored transcript for {vid}: {len(transcript)} chars")
#     except Exception as e:
#         log.warning(f"Failed to store transcript for {vid}: {e}")

# async def _process_job_async(job_id: str, user_id: int, quality: Optional[str] = None):
#     """Process all items in a job asynchronously with usage tracking."""
#     if job_id not in JOBS:
#         log.warning(f"Job {job_id} not found for processing")
#         return
        
#     job = JOBS[job_id]
#     kind = job["kind"]
    
#     # Get database session and user for usage tracking
#     from models import get_db
#     with next(get_db()) as db:
#         user = db.query(User).filter(User.id == user_id).first()
#         if not user:
#             log.error(f"User {user_id} not found for job {job_id}")
#             return
    
#         # Update job status
#         job["status"] = "processing"
#         _save_snapshot()
        
#         log.info(f"Starting batch processing job {job_id} ({kind}) for user {user.username}")
        
#         # Process each queued item
#         for item in job["items"]:
#             if item["status"] == "queued":
#                 _process_item(item, kind, user, db, quality)
#                 _update_job_counts(job)
#                 _save_snapshot()
#                 # Small delay between items to avoid overwhelming the system
#                 await asyncio.sleep(1)
        
#         # Final job status update
#         counts = _calc_counts(job["items"])
#         job.update(counts)
        
#         if counts["failed"] == 0:
#             job["status"] = "completed"
#         elif counts["completed"] > 0:
#             job["status"] = "partial"
#         else:
#             job["status"] = "failed"
        
#         job["completed_at"] = datetime.utcnow().isoformat()
#         _save_snapshot()
        
#         log.info(f"Job {job_id} completed: {counts['completed']}/{counts['total']} successful ({kind})")

# def _calc_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
#     completed = sum(1 for it in items if it["status"] == "completed")
#     failed = sum(1 for it in items if it["status"] == "failed")
#     queued = sum(1 for it in items if it["status"] == "queued")
#     processing = sum(1 for it in items if it["status"] == "processing")
#     return {
#         "completed": completed, 
#         "failed": failed, 
#         "queued": queued,
#         "processing": processing,
#         "total": len(items)
#     }

# def _update_job_counts(job: Dict[str, Any]):
#     """Update job counts based on current item statuses."""
#     counts = _calc_counts(job["items"])
#     job.update(counts)

# def _own_job_or_404(job_id: str, user_id: int) -> Dict[str, Any]:
#     job = JOBS.get(job_id)
#     if not job or job["user_id"] != user_id:
#         raise HTTPException(404, "Job not found")
#     return job

# # ---- Integrations -----------------------------------------------------------

# def _sign(payload: bytes) -> str:
#     if not WEBHOOK_SECRET:
#         return ""
#     return hmac.new(WEBHOOK_SECRET.encode("utf-8"), payload, hashlib.sha256).hexdigest()

# def _notify_webhook(job_public: Dict[str, Any]):
#     if not WEBHOOK_URL or not requests:
#         return
#     try:
#         data = json.dumps({"type": "batch.completed", "job": job_public}, ensure_ascii=False).encode("utf-8")
#         headers = {"Content-Type": "application/json"}
#         sig = _sign(data)
#         if sig:
#             headers["X-Signature"] = sig
#         requests.post(WEBHOOK_URL, data=data, headers=headers, timeout=10)
#     except Exception as e:  # pragma: no cover
#         log.warning("webhook notify failed: %s", e)

# def _notify_slack(job_public: Dict[str, Any]):
#     if not SLACK_WEBHOOK_URL or not requests:
#         return
#     try:
#         completed = job_public.get("completed", 0)
#         total = job_public.get("total", 0)
#         kind = job_public.get("kind", "unknown")
#         text = f"Batch {job_public.get('id')} ({kind}) finished: {completed}/{total} succeeded."
#         requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=8)
#     except Exception as e:  # pragma: no cover
#         log.warning("slack notify failed: %s", e)

# def _upload_s3(job_public: Dict[str, Any]):
#     if not S3_BUCKET or not boto3:
#         return
#     try:
#         s3 = boto3.client("s3")
#         key = f"{S3_PREFIX}{job_public['id']}.json"
#         body = json.dumps(job_public, ensure_ascii=False).encode("utf-8")
#         s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType="application/json")
#         log.info("Uploaded batch summary to s3://%s/%s", S3_BUCKET, key)
#     except (BotoCoreError, ClientError, Exception) as e:  # pragma: no cover
#         log.warning("s3 upload failed: %s", e)

# # ---- Endpoints with usage validation for all types -------------------------

# @router.post("/submit", response_model=JobOut)
# async def submit_batch(
#     payload: SubmitBody = Body(...),
#     request: Request = None,
#     db: Session = Depends(get_db),
#     bg: BackgroundTasks = None,
# ):
#     user = _auth_user(request, db)
#     ids = payload.collect_ids()
#     if not ids:
#         raise HTTPException(400, "No valid YouTube IDs found")

#     # Enforce tier caps
#     tier = getattr(user, "subscription_tier", "free") or "free"
#     if tier == "pro":
#         ids = ids[:PRO_MAX_LINKS]
#     elif tier == "premium":
#         ids = ids[:PREMIUM_MAX_LINKS]
#     else:
#         ids = ids[:min(1, DEFAULT_MAX_LINKS)]  # Free has no batch feature

#     # Pre-validate usage limits for batch submission based on kind
#     usage_type_map = {
#         "transcript": "clean_transcripts",
#         "audio": "audio_downloads", 
#         "video": "video_downloads"
#     }
    
#     usage_type = usage_type_map.get(payload.kind, "clean_transcripts")
#     can_process, current, limit = check_usage_limit(user, usage_type)
    
#     # Check if user can process at least some items
#     remaining_capacity = max(0, limit - current) if limit != float("inf") else len(ids)
    
#     if remaining_capacity == 0:
#         raise HTTPException(
#             403, 
#             f"Monthly limit reached for {payload.kind} ({current}/{limit}). Please upgrade your plan."
#         )
    
#     # Limit batch size based on remaining capacity
#     if remaining_capacity < len(ids):
#         ids = ids[:remaining_capacity]
#         log.info(f"Limiting {payload.kind} batch for user {user.username} to {remaining_capacity} items due to usage limits")

#     job_id = uuid.uuid4().hex[:16]
#     now = datetime.utcnow().isoformat()

#     # Create initial items with queued status
#     items = [_create_initial_item(i, vid, payload.kind) for i, vid in enumerate(ids)]
#     counts = _calc_counts(items)

#     job = {
#         "id": job_id,
#         "user_id": user.id,
#         "created_at": now,
#         "status": "queued",
#         "kind": payload.kind,
#         "quality": payload.quality,
#         **counts,
#         "items": items,
#     }
#     JOBS[job_id] = job
#     _save_snapshot()

#     log.info(f"Created batch job {job_id} ({payload.kind}) with {len(ids)} items for user {user.username}")

#     # Start async processing with user context and quality settings
#     if bg:
#         bg.add_task(_process_job_async, job_id, user.id, payload.quality)

#     public_job = {k: v for k, v in job.items() if k != "user_id"}

#     # Premium integrations (fire-and-forget)
#     if tier == "premium" and bg is not None:
#         bg.add_task(_notify_webhook, public_job)
#         bg.add_task(_notify_slack, public_job)
#         bg.add_task(_upload_s3, public_job)

#     return JobOut(**public_job)

# @router.get("/jobs", response_model=List[JobOut])
# def list_jobs(request: Request, db: Session = Depends(get_db)):
#     user = _auth_user(request, db)
#     rows = sorted(
#         [j for j in JOBS.values() if j["user_id"] == user.id],
#         key=lambda j: j["created_at"],
#         reverse=True,
#     )
#     out = []
#     for j in rows:
#         out.append(JobOut(**{k: v for k, v in j.items() if k != "user_id"}))
#     return out

# @router.get("/jobs/{job_id}", response_model=JobOut)
# def get_job(job_id: str, request: Request, db: Session = Depends(get_db)):
#     user = _auth_user(request, db)
#     job = _own_job_or_404(job_id, user.id)
#     return JobOut(**{k: v for k, v in job.items() if k != "user_id"})

# @router.post("/jobs/{job_id}/retry_failed", response_model=JobOut)
# async def retry_failed(job_id: str, request: Request, db: Session = Depends(get_db), bg: BackgroundTasks = None):
#     user = _auth_user(request, db)
#     job = _own_job_or_404(job_id, user.id)
    
#     # Reset failed items to queued
#     changed = False
#     for item in job["items"]:
#         if item["status"] == "failed":
#             item["status"] = "queued"
#             item["message"] = "Retrying..."
#             item["result_url"] = None
#             item["download_links"] = None
#             changed = True

#     if changed:
#         counts = _calc_counts(job["items"])
#         job.update(counts)
#         job["status"] = "queued"
#         _save_snapshot()
        
#         log.info(f"Retrying failed items in job {job_id} ({job['kind']})")
        
#         # Start async processing for retry with user context and quality
#         if bg:
#             bg.add_task(_process_job_async, job_id, user.id, job.get("quality"))
    
#     return JobOut(**{k: v for k, v in job.items() if k != "user_id"})

# @router.delete("/jobs/{job_id}", response_model=dict)
# def delete_job(job_id: str, request: Request, db: Session = Depends(get_db)):
#     user = _auth_user(request, db)
#     job = _own_job_or_404(job_id, user.id)
#     JOBS.pop(job_id, None)
#     _save_snapshot()
#     log.info(f"Deleted job {job_id} ({job['kind']}) for user {user.username}")
#     return {"ok": True, "deleted": job_id}

# # ---- Download endpoints -----------------------------------------------------

# @router.get("/download/{video_id}/transcript")
# async def download_transcript(video_id: str, request: Request, db: Session = Depends(get_db)):
#     """Download transcript file for a completed batch item."""
#     user = _auth_user(request, db)
    
#     transcript_file = DOWNLOADS_DIR / f"{video_id}_transcript.txt"
#     if not transcript_file.exists():
#         raise HTTPException(404, "Transcript not found")
    
#     from fastapi.responses import FileResponse
#     return FileResponse(
#         path=str(transcript_file),
#         media_type="text/plain",
#         filename=f"{video_id}_transcript.txt"
#     )

# @router.get("/view/{video_id}/transcript")
# async def view_transcript(video_id: str, request: Request, db: Session = Depends(get_db)):
#     """View transcript content directly."""
#     user = _auth_user(request, db)
    
#     transcript_file = DOWNLOADS_DIR / f"{video_id}_transcript.txt"
#     if not transcript_file.exists():
#         raise HTTPException(404, "Transcript not found")
    
#     content = transcript_file.read_text(encoding='utf-8')
#     return {"video_id": video_id, "transcript": content}


# #------------End bach Module-----------