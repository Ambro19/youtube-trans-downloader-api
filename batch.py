# # backend/batch.py
# # Minimal Batch Jobs router with Pro vs Premium caps and optional integrations
# # - Auth: Bearer/JWT (same env keys as main.py/payment.py)
# # - Storage: in-memory + optional JSON snapshot (dev only)
# # - Endpoints:
# #     POST   /batch/submit
# #     GET    /batch/jobs
# #     GET    /batch/jobs/{job_id}
# #     POST   /batch/jobs/{job_id}/retry_failed
# #     DELETE /batch/jobs/{job_id}

# from __future__ import annotations

# import os
# import re
# import json
# import uuid
# import hmac
# import hashlib
# from typing import List, Dict, Any, Optional
# from datetime import datetime

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

# log = logging.getLogger("batch")
# router = APIRouter(prefix="/batch", tags=["batch"])

# # ---- Auth config (match main.py/payment.py) --------------------------------
# SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
# ALGORITHM = os.getenv("ALGORITHM", "HS256")

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


# class JobOut(BaseModel):
#     id: str
#     created_at: str
#     status: str
#     kind: str
#     total: int
#     completed: int
#     failed: int
#     items: List[ItemOut]


# # ---- Core helpers (stubbing) ------------------------------------------------

# def _stub_item(idx: int, vid: str, kind: str) -> Dict[str, Any]:
#     """Create a plausible finished item result for UI demo."""
#     # Make 1 out of 12 look failed to exercise UI states
#     if (idx + len(vid)) % 12 == 0:
#         return {
#             "idx": idx,
#             "youtube_id": vid,
#             "status": "failed",
#             "message": "stub: processing error",
#             "result_type": kind,
#             "result_url": None,
#         }
#     if kind == "audio":
#         return {
#             "idx": idx,
#             "youtube_id": vid,
#             "status": "completed",
#             "message": None,
#             "result_type": "audio",
#             "result_url": f"/files/{vid}_audio_medium.mp3",
#         }
#     if kind == "video":
#         return {
#             "idx": idx,
#             "youtube_id": vid,
#             "status": "completed",
#             "message": None,
#             "result_type": "video",
#             "result_url": f"/files/{vid}_video_720p.mp4",
#         }
#     # transcript
#     return {
#         "idx": idx,
#         "youtube_id": vid,
#         "status": "completed",
#         "message": None,
#         "result_type": "transcript",
#         "result_url": None,
#     }


# def _calc_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
#     completed = sum(1 for it in items if it["status"] == "completed")
#     failed = sum(1 for it in items if it["status"] == "failed")
#     return {"completed": completed, "failed": failed, "total": len(items)}


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
#         text = f"Batch {job_public.get('id')} finished: {completed}/{total} succeeded."
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


# # ---- Endpoints --------------------------------------------------------------

# @router.post("/submit", response_model=JobOut)
# def submit_batch(
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

#     job_id = uuid.uuid4().hex[:16]
#     now = datetime.utcnow().isoformat()

#     items = [_stub_item(i, vid, payload.kind) for i, vid in enumerate(ids)]
#     counts = _calc_counts(items)

#     job = {
#         "id": job_id,
#         "user_id": user.id,
#         "created_at": now,
#         "status": "completed" if counts["failed"] == 0 else "processing",
#         "kind": payload.kind,
#         **counts,
#         "items": items,
#     }
#     JOBS[job_id] = job
#     _save_snapshot()

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
# def retry_failed(job_id: str, request: Request, db: Session = Depends(get_db)):
#     user = _auth_user(request, db)
#     job = _own_job_or_404(job_id, user.id)
#     changed = False
#     for it in job["items"]:
#         if it["status"] == "failed":
#             kind = job["kind"]
#             vid = it["youtube_id"]
#             new_it = _stub_item(it["idx"], vid, kind)
#             if new_it["status"] == "failed":
#                 # force success on retry
#                 new_it["status"] = "completed"
#                 if kind == "audio":
#                     new_it["result_url"] = f"/files/{vid}_audio_medium.mp3"
#                 elif kind == "video":
#                     new_it["result_url"] = f"/files/{vid}_video_720p.mp4"
#                 else:
#                     new_it["result_url"] = None
#                 new_it["message"] = None
#             it.update(new_it)
#             changed = True

#     if changed:
#         counts = _calc_counts(job["items"])
#         job.update(counts)
#         job["status"] = "completed" if counts["failed"] == 0 else "processing"
#         _save_snapshot()
#     return JobOut(**{k: v for k, v in job.items() if k != "user_id"})


# @router.delete("/jobs/{job_id}", response_model=dict)
# def delete_job(job_id: str, request: Request, db: Session = Depends(get_db)):
#     user = _auth_user(request, db)
#     _own_job_or_404(job_id, user.id)
#     JOBS.pop(job_id, None)
#     _save_snapshot()
#     return {"ok": True, "deleted": job_id}

###########################################################
###########################################################

# backend/batch.py
# Enhanced Batch Jobs router with real processing and download links
# - Auth: Bearer/JWT (same env keys as main.py/payment.py)
# - Storage: in-memory + optional JSON snapshot (dev only)
# - Real processing with download links in results

from __future__ import annotations

import os
import re
import json
import uuid
import hmac
import hashlib
import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Body, BackgroundTasks
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
import jwt
import logging

# Optional deps (best-effort)
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

try:
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None
    BotoCoreError = ClientError = Exception

from models import User, get_db
from transcript_utils import (
    get_transcript_with_ytdlp,
    download_audio_with_ytdlp, 
    download_video_with_ytdlp,
    get_video_info,
    check_ytdlp_availability
)

log = logging.getLogger("batch")
router = APIRouter(prefix="/batch", tags=["batch"])

# ---- Auth config (match main.py/payment.py) --------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")

# ---- Downloads directory ----
DOWNLOADS_DIR = Path(__file__).resolve().parent / "server_files"
DOWNLOADS_DIR.mkdir(exist_ok=True)

# ---- Caps ------------------------------------------------------------------
PRO_MAX_LINKS = int(os.getenv("BATCH_PRO_MAX_LINKS", "3"))
PREMIUM_MAX_LINKS = int(os.getenv("BATCH_PREMIUM_MAX_LINKS", "2000"))
DEFAULT_MAX_LINKS = int(os.getenv("BATCH_DEFAULT_MAX_LINKS", "300"))  # safety

# ---- Optional Integrations (Premium) ---------------------------------------
WEBHOOK_URL = os.getenv("BATCH_WEBHOOK_URL")
WEBHOOK_SECRET = os.getenv("BATCH_WEBHOOK_SECRET", "")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
S3_BUCKET = os.getenv("BATCH_S3_BUCKET")
S3_PREFIX = os.getenv("BATCH_S3_PREFIX", "batch-results/")

def _auth_user(request: Request, db: Session) -> User:
    auth = request.headers.get("authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = auth.split()[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(401, "Bad token")
    except Exception:
        raise HTTPException(401, "Invalid token")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(401, "User not found")
    return user

# ---- Simple ID extractor (no import from main.py to avoid circular deps) ---
_YT_PATTERNS = [
    re.compile(r"(?:youtube\.com\/watch\?v=)([^&\n?#]+)", re.I),
    re.compile(r"(?:youtu\.be\/)([^&\n?#]+)", re.I),
    re.compile(r"(?:youtube\.com\/embed\/)([^&\n?#]+)", re.I),
    re.compile(r"(?:youtube\.com\/shorts\/)([^&\n?#]+)", re.I),
    re.compile(r"[?&]v=([^&\n?#]+)", re.I),
]

def extract_youtube_video_id(text: str) -> str:
    s = (text or "").strip()
    for p in _YT_PATTERNS:
        m = p.search(s)
        if m:
            return m.group(1)[:11]
    return s[:11]  # assume raw ID

# ---- In-memory job store ----------------------------------------------------
JOBS: Dict[str, Dict[str, Any]] = {}

SNAPSHOT_FILE = os.getenv("BATCH_SNAPSHOT_FILE", "./.batch_jobs_snapshot.json")
ENABLE_SNAPSHOT = os.getenv("ENVIRONMENT", "development") != "production"

def _save_snapshot():
    if not ENABLE_SNAPSHOT:
        return
    try:
        data = {"jobs": JOBS, "saved_at": datetime.utcnow().isoformat()}
        with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:  # pragma: no cover
        log.debug("snapshot save skipped: %s", e)

def _load_snapshot():
    if not ENABLE_SNAPSHOT:
        return
    try:
        if os.path.exists(SNAPSHOT_FILE):
            with open(SNAPSHOT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                JOBS.clear()
                JOBS.update(data.get("jobs", {}))
                log.info("Restored %d batch jobs from snapshot", len(JOBS))
    except Exception as e:  # pragma: no cover
        log.debug("snapshot load failed: %s", e)

_load_snapshot()

# ---- Pydantic models --------------------------------------------------------
VALID_KINDS = {"transcript", "audio", "video"}

class SubmitBody(BaseModel):
    urls: Optional[List[str]] = Field(default=None, description="List of YouTube URLs or IDs")
    csv_text: Optional[str] = Field(default=None, description="CSV or newline list of URLs/IDs")
    kind: str = Field(default="transcript", description="transcript|audio|video")

    @validator("kind")
    def _kind_ok(cls, v):
        vv = (v or "").strip().lower()
        if vv not in VALID_KINDS:
            raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}")
        return vv

    def collect_ids(self) -> List[str]:
        raw: List[str] = []
        if self.urls:
            raw.extend(self.urls)
        if self.csv_text:
            parts = re.split(r"[\s,;]+", self.csv_text.strip())
            raw.extend([p for p in parts if p])
        # de-dupe while preserving order
        seen = set()
        ids: List[str] = []
        for entry in raw:
            vid = extract_youtube_video_id(entry)
            if len(vid) == 11 and vid not in seen:
                seen.add(vid)
                ids.append(vid)
        # safety cap before tier rules
        return ids[:DEFAULT_MAX_LINKS]

class ItemOut(BaseModel):
    idx: int
    youtube_id: str
    status: str
    message: Optional[str] = None
    result_type: str
    result_url: Optional[str] = None
    file_size: Optional[int] = None
    file_size_mb: Optional[float] = None
    video_title: Optional[str] = None
    download_links: Optional[Dict[str, str]] = None

class JobOut(BaseModel):
    id: str
    created_at: str
    status: str
    kind: str
    total: int
    completed: int
    failed: int
    queued: int
    processing: int
    items: List[ItemOut]

# ---- Core processing functions ----------------------------------------------

def _create_initial_item(idx: int, vid: str, kind: str) -> Dict[str, Any]:
    """Create initial item with queued status for real processing."""
    return {
        "idx": idx,
        "youtube_id": vid,
        "status": "queued",
        "message": "Waiting to process...",
        "result_type": kind,
        "result_url": None,
        "file_size": None,
        "file_size_mb": None,
        "video_title": None,
        "download_links": None,
        "created_at": datetime.utcnow().isoformat(),
    }

def _process_item(item: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """Actually process a single item and return updated item data."""
    vid = item["youtube_id"]
    
    try:
        # Update to processing status
        item["status"] = "processing"
        item["message"] = "Processing..."
        
        # Get video info first
        video_info = get_video_info(vid)
        if video_info:
            item["video_title"] = video_info.get("title", "Unknown Title")
        
        if kind == "transcript":
            # Process transcript
            transcript = get_transcript_with_ytdlp(vid, clean=True)
            if transcript:
                item["status"] = "completed"
                item["message"] = "Transcript ready"
                item["file_size"] = len(transcript.encode('utf-8'))
                item["file_size_mb"] = round(item["file_size"] / (1024 * 1024), 2)
                item["download_links"] = {
                    "transcript": f"/batch/download/{vid}/transcript",
                    "view": f"/batch/view/{vid}/transcript"
                }
                # Store transcript content for download
                _store_transcript_result(vid, transcript)
            else:
                raise Exception("No transcript available")
                
        elif kind == "audio":
            # Process audio download
            if not check_ytdlp_availability():
                raise Exception("Audio processing unavailable")
            
            file_path = download_audio_with_ytdlp(vid, "medium", str(DOWNLOADS_DIR))
            if file_path and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                filename = os.path.basename(file_path)
                
                item["status"] = "completed"
                item["message"] = "Audio ready for download"
                item["result_url"] = f"/files/{filename}"
                item["file_size"] = file_size
                item["file_size_mb"] = round(file_size / (1024 * 1024), 2)
                item["download_links"] = {
                    "audio": f"/files/{filename}",
                    "direct": f"/download-file/audio/{filename}"
                }
            else:
                raise Exception("Audio download failed")
                
        elif kind == "video":
            # Process video download
            if not check_ytdlp_availability():
                raise Exception("Video processing unavailable")
                
            file_path = download_video_with_ytdlp(vid, "720p", str(DOWNLOADS_DIR))
            if file_path and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                filename = os.path.basename(file_path)
                
                item["status"] = "completed"
                item["message"] = "Video ready for download"
                item["result_url"] = f"/files/{filename}"
                item["file_size"] = file_size
                item["file_size_mb"] = round(file_size / (1024 * 1024), 2)
                item["download_links"] = {
                    "video": f"/files/{filename}",
                    "direct": f"/download-file/video/{filename}"
                }
            else:
                raise Exception("Video download failed")
        
        item["processed_at"] = datetime.utcnow().isoformat()
        
    except Exception as e:
        log.error(f"Processing failed for {vid}: {e}")
        item["status"] = "failed"
        item["message"] = str(e)
        item["result_url"] = None
        item["download_links"] = None
        item["failed_at"] = datetime.utcnow().isoformat()
    
    return item

def _store_transcript_result(vid: str, transcript: str):
    """Store transcript content for later download."""
    transcript_file = DOWNLOADS_DIR / f"{vid}_transcript.txt"
    try:
        transcript_file.write_text(transcript, encoding='utf-8')
    except Exception as e:
        log.warning(f"Failed to store transcript for {vid}: {e}")

async def _process_job_async(job_id: str):
    """Process all items in a job asynchronously."""
    if job_id not in JOBS:
        return
        
    job = JOBS[job_id]
    kind = job["kind"]
    
    # Update job status
    job["status"] = "processing"
    _save_snapshot()
    
    # Process each queued item
    for item in job["items"]:
        if item["status"] == "queued":
            _process_item(item, kind)
            _update_job_counts(job)
            _save_snapshot()
            # Small delay between items to avoid overwhelming the system
            await asyncio.sleep(1)
    
    # Final job status update
    counts = _calc_counts(job["items"])
    job.update(counts)
    
    if counts["failed"] == 0:
        job["status"] = "completed"
    elif counts["completed"] > 0:
        job["status"] = "partial"
    else:
        job["status"] = "failed"
    
    job["completed_at"] = datetime.utcnow().isoformat()
    _save_snapshot()
    
    log.info(f"Job {job_id} completed: {counts['completed']}/{counts['total']} successful")

def _calc_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    completed = sum(1 for it in items if it["status"] == "completed")
    failed = sum(1 for it in items if it["status"] == "failed")
    queued = sum(1 for it in items if it["status"] == "queued")
    processing = sum(1 for it in items if it["status"] == "processing")
    return {
        "completed": completed, 
        "failed": failed, 
        "queued": queued,
        "processing": processing,
        "total": len(items)
    }

def _update_job_counts(job: Dict[str, Any]):
    """Update job counts based on current item statuses."""
    counts = _calc_counts(job["items"])
    job.update(counts)

def _own_job_or_404(job_id: str, user_id: int) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job or job["user_id"] != user_id:
        raise HTTPException(404, "Job not found")
    return job

# ---- Integrations -----------------------------------------------------------

def _sign(payload: bytes) -> str:
    if not WEBHOOK_SECRET:
        return ""
    return hmac.new(WEBHOOK_SECRET.encode("utf-8"), payload, hashlib.sha256).hexdigest()

def _notify_webhook(job_public: Dict[str, Any]):
    if not WEBHOOK_URL or not requests:
        return
    try:
        data = json.dumps({"type": "batch.completed", "job": job_public}, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        sig = _sign(data)
        if sig:
            headers["X-Signature"] = sig
        requests.post(WEBHOOK_URL, data=data, headers=headers, timeout=10)
    except Exception as e:  # pragma: no cover
        log.warning("webhook notify failed: %s", e)

def _notify_slack(job_public: Dict[str, Any]):
    if not SLACK_WEBHOOK_URL or not requests:
        return
    try:
        completed = job_public.get("completed", 0)
        total = job_public.get("total", 0)
        text = f"Batch {job_public.get('id')} finished: {completed}/{total} succeeded."
        requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=8)
    except Exception as e:  # pragma: no cover
        log.warning("slack notify failed: %s", e)

def _upload_s3(job_public: Dict[str, Any]):
    if not S3_BUCKET or not boto3:
        return
    try:
        s3 = boto3.client("s3")
        key = f"{S3_PREFIX}{job_public['id']}.json"
        body = json.dumps(job_public, ensure_ascii=False).encode("utf-8")
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType="application/json")
        log.info("Uploaded batch summary to s3://%s/%s", S3_BUCKET, key)
    except (BotoCoreError, ClientError, Exception) as e:  # pragma: no cover
        log.warning("s3 upload failed: %s", e)

# ---- Endpoints --------------------------------------------------------------

@router.post("/submit", response_model=JobOut)
async def submit_batch(
    payload: SubmitBody = Body(...),
    request: Request = None,
    db: Session = Depends(get_db),
    bg: BackgroundTasks = None,
):
    user = _auth_user(request, db)
    ids = payload.collect_ids()
    if not ids:
        raise HTTPException(400, "No valid YouTube IDs found")

    # Enforce tier caps
    tier = getattr(user, "subscription_tier", "free") or "free"
    if tier == "pro":
        ids = ids[:PRO_MAX_LINKS]
    elif tier == "premium":
        ids = ids[:PREMIUM_MAX_LINKS]
    else:
        ids = ids[:min(1, DEFAULT_MAX_LINKS)]  # Free has no batch feature

    job_id = uuid.uuid4().hex[:16]
    now = datetime.utcnow().isoformat()

    # Create initial items with queued status
    items = [_create_initial_item(i, vid, payload.kind) for i, vid in enumerate(ids)]
    counts = _calc_counts(items)

    job = {
        "id": job_id,
        "user_id": user.id,
        "created_at": now,
        "status": "queued",
        "kind": payload.kind,
        **counts,
        "items": items,
    }
    JOBS[job_id] = job
    _save_snapshot()

    # Start async processing
    if bg:
        bg.add_task(_process_job_async, job_id)

    public_job = {k: v for k, v in job.items() if k != "user_id"}

    # Premium integrations (fire-and-forget)
    if tier == "premium" and bg is not None:
        bg.add_task(_notify_webhook, public_job)
        bg.add_task(_notify_slack, public_job)
        bg.add_task(_upload_s3, public_job)

    return JobOut(**public_job)

@router.get("/jobs", response_model=List[JobOut])
def list_jobs(request: Request, db: Session = Depends(get_db)):
    user = _auth_user(request, db)
    rows = sorted(
        [j for j in JOBS.values() if j["user_id"] == user.id],
        key=lambda j: j["created_at"],
        reverse=True,
    )
    out = []
    for j in rows:
        out.append(JobOut(**{k: v for k, v in j.items() if k != "user_id"}))
    return out

@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, request: Request, db: Session = Depends(get_db)):
    user = _auth_user(request, db)
    job = _own_job_or_404(job_id, user.id)
    return JobOut(**{k: v for k, v in job.items() if k != "user_id"})

@router.post("/jobs/{job_id}/retry_failed", response_model=JobOut)
async def retry_failed(job_id: str, request: Request, db: Session = Depends(get_db), bg: BackgroundTasks = None):
    user = _auth_user(request, db)
    job = _own_job_or_404(job_id, user.id)
    
    # Reset failed items to queued
    changed = False
    for item in job["items"]:
        if item["status"] == "failed":
            item["status"] = "queued"
            item["message"] = "Retrying..."
            item["result_url"] = None
            item["download_links"] = None
            changed = True

    if changed:
        counts = _calc_counts(job["items"])
        job.update(counts)
        job["status"] = "queued"
        _save_snapshot()
        
        # Start async processing for retry
        if bg:
            bg.add_task(_process_job_async, job_id)
    
    return JobOut(**{k: v for k, v in job.items() if k != "user_id"})

@router.delete("/jobs/{job_id}", response_model=dict)
def delete_job(job_id: str, request: Request, db: Session = Depends(get_db)):
    user = _auth_user(request, db)
    _own_job_or_404(job_id, user.id)
    JOBS.pop(job_id, None)
    _save_snapshot()
    return {"ok": True, "deleted": job_id}

# ---- Download endpoints -----------------------------------------------------

@router.get("/download/{video_id}/transcript")
async def download_transcript(video_id: str, request: Request, db: Session = Depends(get_db)):
    """Download transcript file for a completed batch item."""
    user = _auth_user(request, db)
    
    transcript_file = DOWNLOADS_DIR / f"{video_id}_transcript.txt"
    if not transcript_file.exists():
        raise HTTPException(404, "Transcript not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        path=str(transcript_file),
        media_type="text/plain",
        filename=f"{video_id}_transcript.txt"
    )

@router.get("/view/{video_id}/transcript")
async def view_transcript(video_id: str, request: Request, db: Session = Depends(get_db)):
    """View transcript content directly."""
    user = _auth_user(request, db)
    
    transcript_file = DOWNLOADS_DIR / f"{video_id}_transcript.txt"
    if not transcript_file.exists():
        raise HTTPException(404, "Transcript not found")
    
    content = transcript_file.read_text(encoding='utf-8')
    return {"video_id": video_id, "transcript": content}