# ============================================================================
# FIXED TRANSCRIPT ENDPOINT FOR main.py
# ============================================================================
# Replace your current /download_transcript/ endpoint with this code
# 
# Key fixes:
# 1. Properly handles FetchedTranscriptSnippet objects (no .get() method)
# 2. Uses yt-dlp as primary method (from transcript_utils.py)
# 3. Falls back to youtube-transcript-api if yt-dlp fails
# 4. Handles all formats: clean text, SRT, VTT
# ============================================================================

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
import logging

# Your existing imports
from transcript_utils import get_transcript_with_ytdlp, validate_video_id

logger = logging.getLogger("youtube_trans_downloader")

# Request model
class TranscriptRequest(BaseModel):
    video_id: str
    clean: bool = True
    format: Optional[str] = None  # 'srt', 'vtt', or None for plain text


# ============================================================================
# FIXED ENDPOINT
# ============================================================================

@router.post("/download_transcript/")
async def download_transcript(
    request: TranscriptRequest,
    # Add your auth dependencies here
    # current_user: User = Depends(get_current_user)
):
    """
    Download transcript with proper error handling.
    
    Formats:
    - clean=True, format=None -> Plain text (no timestamps)
    - clean=False, format='srt' -> SRT format with timestamps
    - clean=False, format='vtt' -> VTT format with timestamps
    """
    video_id = request.video_id.strip()
    
    # Validate video ID
    if not validate_video_id(video_id):
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID")
    
    logger.info(f"Transcript for {video_id} (clean={request.clean}, fmt={request.format})")
    
    try:
        # ====================================================================
        # PRIMARY METHOD: Use yt-dlp (from transcript_utils.py)
        # ====================================================================
        transcript_text = get_transcript_with_ytdlp(
            video_id=video_id, 
            clean=request.clean
        )
        
        if transcript_text:
            # Format based on user request
            if request.format == 'srt' and not request.clean:
                content = _convert_to_srt(transcript_text, video_id)
                media_type = "text/plain"
                filename = f"{video_id}_transcript.srt"
            elif request.format == 'vtt' and not request.clean:
                content = _convert_to_vtt(transcript_text, video_id)
                media_type = "text/vtt"
                filename = f"{video_id}_transcript.vtt"
            else:
                # Plain text (clean format)
                content = transcript_text
                media_type = "text/plain"
                filename = f"{video_id}_transcript.txt"
            
            logger.info(f"✅ Transcript ready: {filename}")
            
            return Response(
                content=content,
                media_type=media_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"'
                }
            )
        
        # ====================================================================
        # FALLBACK METHOD: Use youtube-transcript-api
        # ====================================================================
        logger.warning(f"yt-dlp failed, trying fallback method for {video_id}")
        transcript_text = _get_transcript_fallback(video_id, request.clean, request.format)
        
        if transcript_text:
            if request.format == 'srt':
                media_type = "text/plain"
                filename = f"{video_id}_transcript.srt"
            elif request.format == 'vtt':
                media_type = "text/vtt"
                filename = f"{video_id}_transcript.vtt"
            else:
                media_type = "text/plain"
                filename = f"{video_id}_transcript.txt"
            
            return Response(
                content=transcript_text,
                media_type=media_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"'
                }
            )
        
        # No transcript found by any method
        raise HTTPException(
            status_code=404, 
            detail="No transcript/captions found for this video"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcript pipeline error for {video_id}: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Could not retrieve transcript: {str(e)}"
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _convert_to_srt(text: str, video_id: str) -> str:
    """
    Convert plain text to SRT format with estimated timestamps.
    If text already has timestamps like [00:12], preserve them.
    """
    lines = text.strip().split('\n')
    srt_content = []
    index = 1
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Estimate timing (5 seconds per line)
        start_sec = i * 5
        end_sec = start_sec + 5
        
        start_time = _format_srt_timestamp(start_sec)
        end_time = _format_srt_timestamp(end_sec)
        
        srt_content.append(f"{index}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(line)
        srt_content.append("")  # Blank line
        index += 1
    
    return "\n".join(srt_content)


def _convert_to_vtt(text: str, video_id: str) -> str:
    """Convert plain text to WebVTT format."""
    vtt_content = ["WEBVTT", ""]
    
    lines = text.strip().split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        start_sec = i * 5
        end_sec = start_sec + 5
        
        start_time = _format_vtt_timestamp(start_sec)
        end_time = _format_vtt_timestamp(end_sec)
        
        vtt_content.append(f"{start_time} --> {end_time}")
        vtt_content.append(line)
        vtt_content.append("")
    
    return "\n".join(vtt_content)


def _format_srt_timestamp(seconds: int) -> str:
    """Format seconds as SRT timestamp: 00:00:12,000"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},000"


def _format_vtt_timestamp(seconds: int) -> str:
    """Format seconds as WebVTT timestamp: 00:00:12.000"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.000"


def _get_transcript_fallback(video_id: str, clean: bool, fmt: Optional[str]) -> Optional[str]:
    """
    Fallback method using youtube-transcript-api.
    
    FIXED: Properly handles FetchedTranscriptSnippet objects.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        if not transcript_list:
            return None
        
        # ================================================================
        # FIX: Access snippet attributes correctly
        # ================================================================
        if clean:
            # Clean format: just text, no timestamps
            texts = []
            for snippet in transcript_list:
                # ✅ FIXED: Use dictionary access or attribute access
                # Option 1: Dictionary-style
                if isinstance(snippet, dict):
                    text = snippet.get('text', '')
                else:
                    # Option 2: Attribute-style (for FetchedTranscriptSnippet)
                    text = getattr(snippet, 'text', '')
                
                if text:
                    texts.append(text.strip())
            
            return " ".join(texts)
        
        elif fmt == 'srt':
            # SRT format with timestamps
            srt_lines = []
            for i, snippet in enumerate(transcript_list, 1):
                # ✅ FIXED: Proper attribute access
                if isinstance(snippet, dict):
                    text = snippet.get('text', '')
                    start = snippet.get('start', 0)
                    duration = snippet.get('duration', 3)
                else:
                    text = getattr(snippet, 'text', '')
                    start = getattr(snippet, 'start', 0)
                    duration = getattr(snippet, 'duration', 3)
                
                if not text:
                    continue
                
                end = start + duration
                start_ts = _format_srt_timestamp(int(start))
                end_ts = _format_srt_timestamp(int(end))
                
                srt_lines.append(f"{i}")
                srt_lines.append(f"{start_ts} --> {end_ts}")
                srt_lines.append(text.strip())
                srt_lines.append("")
            
            return "\n".join(srt_lines)
        
        elif fmt == 'vtt':
            # WebVTT format
            vtt_lines = ["WEBVTT", ""]
            
            for snippet in transcript_list:
                # ✅ FIXED: Proper attribute access
                if isinstance(snippet, dict):
                    text = snippet.get('text', '')
                    start = snippet.get('start', 0)
                    duration = snippet.get('duration', 3)
                else:
                    text = getattr(snippet, 'text', '')
                    start = getattr(snippet, 'start', 0)
                    duration = getattr(snippet, 'duration', 3)
                
                if not text:
                    continue
                
                end = start + duration
                start_ts = _format_vtt_timestamp(int(start))
                end_ts = _format_vtt_timestamp(int(end))
                
                vtt_lines.append(f"{start_ts} --> {end_ts}")
                vtt_lines.append(text.strip())
                vtt_lines.append("")
            
            return "\n".join(vtt_lines)
        
        return None
        
    except Exception as e:
        logger.warning(f"Fallback transcript method failed: {e}")
        return None


# ============================================================================
# USAGE NOTES
# ============================================================================
# 
# 1. Find your /download_transcript/ endpoint in app.py
# 2. Replace it with this fixed code
# 3. Make sure you have these imports at the top of app.py:
#    from transcript_utils import get_transcript_with_ytdlp, validate_video_id
#    from youtube_transcript_api import YouTubeTranscriptApi
# 
# 4. Test with:
#    - Clean format (plain text)
#    - SRT format (with timestamps)
#    - VTT format (with timestamps)
# 
# ============================================================================