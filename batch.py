# backend/batch.py
# Minimal Batch Jobs router (stubbed processing so the UI lights up immediately)
# - Auth: same Bearer/JWT flow as payment.py
# - Storage: in-memory (per-process) with optional JSON snapshot to disk for dev
# - Endpoints:
#     POST   /batch/submit                  -> create a new batch job
#     GET    /batch/jobs                    -> list my jobs
#     GET    /batch/jobs/{job_id}           -> fetch job details
#     POST   /batch/jobs/{job_id}/retry_failed -> retry failed items (re-stub)
#     DELETE /batch/jobs/{job_id}           -> delete a job

from __future__ import annotations

import os
import re
import json
import uuid
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Body
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
import jwt
import logging

from models import User, get_db

log = logging.getLogger("batch")
router = APIRouter(prefix="/batch", tags=["batch"])

# ---- Auth config (match payment.py) -----------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")

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

# ---- Simple ID extractor (no main.py import to avoid circular deps) ---------
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

# ---- In-memory job store (dev-friendly) ------------------------------------
# Schema (dicts for simplicity):
# JOBS[job_id] = {
#   "id": str, "user_id": int, "created_at": iso, "status": "completed|processing|queued",
#   "kind": "transcript|audio|video",
#   "total": int, "completed": int, "failed": int,
#   "items": [{"idx": int, "youtube_id": str, "status": "...", "message": str|None,
#              "result_type": str, "result_url": str|None}]
# }
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
    except Exception as e:
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
    except Exception as e:
        log.debug("snapshot load failed: %s", e)

_load_snapshot()

# ---- Pydantic models --------------------------------------------------------
class BatchMode(str):
    # declared as constants-not-enum for brevity in this stub
    pass

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
            # very lenient parser: split on commas/newlines/whitespace
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
        return ids[:300]  # safety cap

class ItemOut(BaseModel):
    idx: int
    youtube_id: str
    status: str
    message: Optional[str] = None
    result_type: str
    result_url: Optional[str] = None

class JobOut(BaseModel):
    id: str
    created_at: str
    status: str
    kind: str
    total: int
    completed: int
    failed: int
    items: List[ItemOut]

# ---- Core helpers (stubbing) ------------------------------------------------
def _stub_item(idx: int, vid: str, kind: str) -> Dict[str, Any]:
    """Create a plausible finished item result for UI demo."""
    # Make 1 out of 12 look failed to exercise UI states:
    if (idx + len(vid)) % 12 == 0:
        return {
            "idx": idx,
            "youtube_id": vid,
            "status": "failed",
            "message": "stub: processing error",
            "result_type": kind,
            "result_url": None,
        }
    if kind == "audio":
        return {
            "idx": idx,
            "youtube_id": vid,
            "status": "completed",
            "message": None,
            "result_type": "audio",
            "result_url": f"/files/{vid}_audio_medium.mp3",
        }
    if kind == "video":
        return {
            "idx": idx,
            "youtube_id": vid,
            "status": "completed",
            "message": None,
            "result_type": "video",
            "result_url": f"/files/{vid}_video_720p.mp4",
        }
    # transcript
    return {
        "idx": idx,
        "youtube_id": vid,
        "status": "completed",
        "message": None,
        "result_type": "transcript",
        "result_url": None,  # your UI can show "Download .txt" via frontend blob
    }

def _calc_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    completed = sum(1 for it in items if it["status"] == "completed")
    failed = sum(1 for it in items if it["status"] == "failed")
    return {"completed": completed, "failed": failed, "total": len(items)}

def _own_job_or_404(job_id: str, user_id: int) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job or job["user_id"] != user_id:
        raise HTTPException(404, "Job not found")
    return job

# ---- Endpoints --------------------------------------------------------------

@router.post("/submit", response_model=JobOut)
def submit_batch(
    payload: SubmitBody = Body(...),
    request: Request = None,
    db: Session = Depends(get_db),
):
    user = _auth_user(request, db)
    ids = payload.collect_ids()
    if not ids:
        raise HTTPException(400, "No valid YouTube IDs found")

    job_id = uuid.uuid4().hex[:16]
    now = datetime.utcnow().isoformat()

    items = [_stub_item(i, vid, payload.kind) for i, vid in enumerate(ids)]
    counts = _calc_counts(items)

    job = {
        "id": job_id,
        "user_id": user.id,
        "created_at": now,
        "status": "completed" if counts["failed"] == 0 else "processing",
        "kind": payload.kind,
        **counts,
        "items": items,
    }
    JOBS[job_id] = job
    _save_snapshot()
    return JobOut(**{k: v for k, v in job.items() if k != "user_id"})

@router.get("/jobs", response_model=List[JobOut])
def list_jobs(request: Request, db: Session = Depends(get_db)):
    user = _auth_user(request, db)
    # latest first
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
def retry_failed(job_id: str, request: Request, db: Session = Depends(get_db)):
    user = _auth_user(request, db)
    job = _own_job_or_404(job_id, user.id)
    changed = False
    for it in job["items"]:
        if it["status"] == "failed":
            # re-stub as success
            kind = job["kind"]
            vid = it["youtube_id"]
            new_it = _stub_item(it["idx"], vid, kind)
            if new_it["status"] == "failed":
                # force success on retry
                new_it["status"] = "completed"
                if kind == "audio":
                    new_it["result_url"] = f"/files/{vid}_audio_medium.mp3"
                elif kind == "video":
                    new_it["result_url"] = f"/files/{vid}_video_720p.mp4"
                else:
                    new_it["result_url"] = None
                new_it["message"] = None
            it.update(new_it)
            changed = True

    if changed:
        counts = _calc_counts(job["items"])
        job.update(counts)
        job["status"] = "completed" if counts["failed"] == 0 else "processing"
        _save_snapshot()
    return JobOut(**{k: v for k, v in job.items() if k != "user_id"})

@router.delete("/jobs/{job_id}", response_model=dict)
def delete_job(job_id: str, request: Request, db: Session = Depends(get_db)):
    user = _auth_user(request, db)
    job = _own_job_or_404(job_id, user.id)
    JOBS.pop(job_id, None)
    _save_snapshot()
    return {"ok": True, "deleted": job_id}
