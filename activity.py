# backend/activity.py - Activity tracking endpoints for real-time updates

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from models import get_db, User, TranscriptDownload
from main import get_current_user  # Import your auth function

log = logging.getLogger("activity")
router = APIRouter(prefix="/user", tags=["activity"])

@router.get("/recent-activity")
async def get_recent_activity(
    limit: int = 15,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get recent user activity from download records"""
    try:
        # Get recent downloads for this user
        recent_downloads = db.query(TranscriptDownload)\
            .filter(TranscriptDownload.user_id == current_user.id)\
            .filter(TranscriptDownload.status == 'completed')\
            .order_by(desc(TranscriptDownload.created_at))\
            .limit(limit)\
            .all()
        
        activities = []
        for download in recent_downloads:
            # Map transcript types to proper activity descriptions
            activity_type = download.transcript_type or 'content'
            file_format = download.file_format or 'txt'
            
            # Determine proper action and description based on format and type
            if file_format in ['srt', 'vtt']:
                action = f"Generated {file_format.upper()} Transcript"
                description = f"{file_format.upper()} transcript for video {download.youtube_id}"
                icon = "🕒"
                category = "transcript"
            elif file_format == 'txt':
                if 'unclean' in activity_type.lower():
                    action = "Generated Timestamped Transcript" 
                    description = f"Timestamped transcript for video {download.youtube_id}"
                    icon = "🕒"
                else:
                    action = "Generated Clean Transcript"
                    description = f"Clean text transcript for video {download.youtube_id}"
                    icon = "📄"
                category = "transcript"
            elif file_format in ['mp3', 'm4a', 'aac', 'wav']:
                action = "Downloaded Audio File"
                description = f"{file_format.upper()} file from video {download.youtube_id}"
                icon = "🎵"
                category = "audio"
            elif file_format in ['mp4', 'mkv', 'avi', 'mov']:
                action = "Downloaded Video File" 
                description = f"{file_format.upper()} file from video {download.youtube_id}"
                icon = "🎬"
                category = "video"
            else:
                action = "Downloaded Content"
                description = f"Content for video {download.youtube_id}"
                icon = "📁"
                category = "general"
            
            # Add quality and size info to description
            if download.quality and download.quality != 'default':
                description += f" ({download.quality})"
            if download.file_size:
                size_mb = download.file_size / (1024 * 1024)
                description += f" - {size_mb:.1f}MB"
            
            activity = {
                "id": download.id,
                "action": action,
                "description": description,
                "timestamp": download.created_at.isoformat(),
                "icon": icon,
                "type": activity_type,
                "video_id": download.youtube_id,
                "file_format": file_format,
                "file_size": download.file_size,
                "quality": download.quality,
                "status": download.status,
                "category": category,
                "processing_time": download.processing_time
            }
            activities.append(activity)
        
        return {
            "activities": activities,
            "total": len(activities),
            "user_id": current_user.id,
            "fetched_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        log.error(f"Error fetching recent activity for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch recent activity")

@router.get("/download-history")
async def get_download_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive download history"""
    try:
        # Get paginated download history
        downloads = db.query(TranscriptDownload)\
            .filter(TranscriptDownload.user_id == current_user.id)\
            .order_by(desc(TranscriptDownload.created_at))\
            .offset(offset)\
            .limit(limit)\
            .all()
        
        download_list = []
        for download in downloads:
            download_data = {
                "id": download.id,
                "type": download.transcript_type,
                "video_id": download.youtube_id,
                "file_format": download.file_format,
                "file_size": download.file_size,
                "quality": download.quality,
                "status": download.status,
                "downloaded_at": download.created_at.isoformat(),
                "processing_time": download.processing_time,
                "language": download.language,
                "error_message": download.error_message,
                "file_name": f"{download.transcript_type}_{download.youtube_id}.{download.file_format or 'txt'}"
            }
            download_list.append(download_data)
        
        # Get total count for pagination
        total_count = db.query(TranscriptDownload)\
            .filter(TranscriptDownload.user_id == current_user.id)\
            .count()
        
        return {
            "downloads": download_list,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "user_id": current_user.id,
            "fetched_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        log.error(f"Error fetching download history for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch download history")

@router.get("/activity-summary")
async def get_activity_summary(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get activity summary stats"""
    try:
        # Calculate date range
        since_date = datetime.utcnow() - timedelta(days=days)
        
        # Get counts by type
        downloads = db.query(TranscriptDownload)\
            .filter(TranscriptDownload.user_id == current_user.id)\
            .filter(TranscriptDownload.created_at >= since_date)\
            .filter(TranscriptDownload.status == 'completed')\
            .all()
        
        # Categorize downloads
        stats = {
            "total": len(downloads),
            "transcripts": 0,
            "audio": 0,
            "video": 0,
            "period_days": days,
            "from_date": since_date.isoformat(),
            "to_date": datetime.utcnow().isoformat()
        }
        
        for download in downloads:
            file_format = (download.file_format or '').lower()
            transcript_type = (download.transcript_type or '').lower()
            
            if file_format in ['txt', 'srt', 'vtt'] or 'transcript' in transcript_type:
                stats["transcripts"] += 1
            elif file_format in ['mp3', 'm4a', 'aac', 'wav'] or 'audio' in transcript_type:
                stats["audio"] += 1
            elif file_format in ['mp4', 'mkv', 'avi', 'mov'] or 'video' in transcript_type:
                stats["video"] += 1
        
        return stats
        
    except Exception as e:
        log.error(f"Error fetching activity summary for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch activity summary")

# Function to create activity record when downloads happen
def create_activity_record(
    db: Session, 
    user: User, 
    action_type: str, 
    youtube_id: str, 
    file_format: str = None,
    quality: str = None,
    file_size: int = None,
    processing_time: float = None,
    status: str = "completed"
):
    """Helper function to create activity records during downloads"""
    try:
        from models import TranscriptDownload
        
        record = TranscriptDownload(
            user_id=user.id,
            youtube_id=youtube_id,
            transcript_type=action_type,
            file_format=file_format or 'txt',
            quality=quality,
            file_size=file_size,
            processing_time=processing_time,
            status=status,
            created_at=datetime.utcnow()
        )
        
        db.add(record)
        db.commit()
        db.refresh(record)
        
        log.info(f"Created activity record: {action_type} for user {user.id}, video {youtube_id}")
        return record
        
    except Exception as e:
        log.error(f"Failed to create activity record: {e}")
        db.rollback()
        return None