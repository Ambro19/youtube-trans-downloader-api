# backend/transcript_fetcher.py - Smart Transcript Fetching with Cloud Provider Workarounds
"""
Handles YouTube transcript fetching with multiple strategies to work around cloud provider IP blocks.

Strategies (in order):
1. Direct YouTube Transcript API (fast, but may be blocked on cloud)
2. yt-dlp fallback (more reliable on cloud providers)
3. Proxy-based retry (if proxies configured)

Usage:
    from transcript_fetcher import get_transcript_smart
    
    text = get_transcript_smart(video_id, clean=True, fmt="txt")
"""

import logging
import os
from typing import Optional, List, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

logger = logging.getLogger("youtube_trans_downloader")

# Proxy configuration (optional)
PROXY_LIST = os.getenv("YOUTUBE_PROXIES", "").split(",") if os.getenv("YOUTUBE_PROXIES") else []
USE_PROXIES = len(PROXY_LIST) > 0


def _get_segment_value(seg: Any, key: str, default: Any = None) -> Any:
    """Safely get value from segment (dict or object)."""
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def _sec_to_vtt(ts: float) -> str:
    """Format seconds as VTT timestamp."""
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    ms = int((ts - int(ts)) * 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def _sec_to_srt(ts: float) -> str:
    """Format seconds as SRT timestamp."""
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    ms = int((ts - int(ts)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def segments_to_vtt(segments) -> str:
    """Convert segments to VTT format."""
    lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    
    for seg in segments:
        start = _get_segment_value(seg, "start", 0)
        duration = _get_segment_value(seg, "duration", 0)
        text = _get_segment_value(seg, "text", "")
        
        if not text:
            continue
            
        start_ts = _sec_to_vtt(start)
        end_ts = _sec_to_vtt(start + duration)
        text_clean = text.replace("\n", " ").strip()
        
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text_clean)
        lines.append("")
    
    return "\n".join(lines)


def segments_to_srt(segments) -> str:
    """Convert segments to SRT format."""
    out = []
    
    for i, seg in enumerate(segments, start=1):
        start = _get_segment_value(seg, "start", 0)
        duration = _get_segment_value(seg, "duration", 0)
        text = _get_segment_value(seg, "text", "")
        
        if not text:
            continue
            
        start_ts = _sec_to_srt(start)
        end_ts = _sec_to_srt(start + duration)
        text_clean = text.replace("\n", " ").strip()
        
        out.append(str(i))
        out.append(f"{start_ts} --> {end_ts}")
        out.append(text_clean)
        out.append("")
    
    return "\n".join(out)


def _clean_plain_blocks(blocks: List[str]) -> str:
    """Format plain text into readable paragraphs."""
    out, cur, chars = [], [], 0
    for word in " ".join(blocks).split():
        cur.append(word)
        chars += len(word) + 1
        if chars > 400 and word.endswith((".", "!", "?")):
            out.append(" ".join(cur))
            cur, chars = [], 0
    if cur:
        out.append(" ".join(cur))
    return "\n\n".join(out)


def _format_timestamped(segments) -> str:
    """Format segments with timestamps."""
    lines = []
    for seg in segments:
        start = _get_segment_value(seg, "start", 0)
        text = _get_segment_value(seg, "text", "")
        
        if not text:
            continue
            
        t = int(start)
        text_clean = text.replace("\n", " ")
        lines.append(f"[{t // 60:02d}:{t % 60:02d}] {text_clean}")
    
    return "\n".join(lines)


_EN_PRIORITY = ["en", "en-US", "en-GB", "en-CA", "en-AU", "en-IE", "en-NZ"]


def try_youtube_api_direct(video_id: str, proxies: Optional[Dict] = None) -> Optional[List]:
    """
    Try getting transcript using YouTube Transcript API.
    
    Args:
        video_id: YouTube video ID
        proxies: Optional proxy dict for requests
        
    Returns:
        Segments list or None if failed
    """
    try:
        # Try direct get_transcript first (fastest)
        if proxies:
            segments = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=_EN_PRIORITY,
                proxies=proxies
            )
        else:
            segments = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=_EN_PRIORITY
            )
        
        if segments:
            logger.info(f"✅ Direct API success for {video_id}")
            return segments
            
    except Exception as e:
        logger.debug(f"Direct API failed for {video_id}: {e}")
    
    # Try list_transcripts approach
    try:
        if proxies:
            listing = YouTubeTranscriptApi.list_transcripts(video_id, proxies=proxies)
        else:
            listing = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try priority English variants
        for code in _EN_PRIORITY:
            try:
                t = listing.find_transcript([code])
                segments = t.fetch()
                if segments:
                    logger.info(f"✅ List API success for {video_id} (lang: {code})")
                    return segments
            except:
                continue
        
        # Try generated/auto-captions
        try:
            t = listing.find_generated_transcript(_EN_PRIORITY)
            segments = t.fetch()
            if segments:
                logger.info(f"✅ Generated transcript success for {video_id}")
                return segments
        except:
            pass
        
        # Try translation
        for t in listing:
            try:
                translated = t.translate("en")
                segments = translated.fetch()
                if segments:
                    logger.info(f"✅ Translated transcript success for {video_id}")
                    return segments
            except:
                continue
                
    except Exception as e:
        logger.debug(f"List API failed for {video_id}: {e}")
    
    return None

def try_ytdlp_fallback(video_id: str, clean: bool = True, fmt: str = None) -> Optional[str]:
    """
    Try getting transcript using yt-dlp fallback.
    Now supports SRT/VTT formats!
    """
    try:
        from transcript_utils import get_transcript_with_ytdlp
        
        result = get_transcript_with_ytdlp(video_id, clean=clean, fmt=fmt)  # ← Pass fmt!
        if result:
            logger.info(f"✅ yt-dlp fallback success for {video_id}")
            return result
    except Exception as e:
        logger.debug(f"yt-dlp fallback failed for {video_id}: {e}")
    
    return None

# def try_ytdlp_fallback(video_id: str, clean: bool = True) -> Optional[str]:
#     """
#     Try getting transcript using yt-dlp fallback.
    
#     Args:
#         video_id: YouTube video ID
#         clean: Whether to return clean text
        
#     Returns:
#         Transcript text or None if failed
#     """
#     try:
#         from transcript_utils import get_transcript_with_ytdlp
        
#         result = get_transcript_with_ytdlp(video_id, clean=clean)
#         if result:
#             logger.info(f"✅ yt-dlp fallback success for {video_id}")
#             return result
#     except Exception as e:
#         logger.debug(f"yt-dlp fallback failed for {video_id}: {e}")
    
#     return None


def get_transcript_smart(
    video_id: str, 
    clean: bool = True, 
    fmt: Optional[str] = None,
    use_proxies: bool = USE_PROXIES
) -> str:
    """
    Smart transcript fetching with multiple fallback strategies.
    
    This function is designed to work reliably on cloud providers (Render, AWS, etc.)
    where YouTube may block direct API access.
    
    Strategies (in order):
    1. YouTube Transcript API (direct)
    2. yt-dlp fallback (more reliable on cloud)
    3. Proxy-based retry (if configured)
    
    Args:
        video_id: YouTube video ID (11 characters)
        clean: Return clean text (no timestamps)
        fmt: Output format ('srt', 'vtt', or None for txt)
        use_proxies: Whether to try proxy-based requests
        
    Returns:
        Transcript text in requested format
        
    Raises:
        Exception: If all strategies fail
    """
    logger.info(f"Smart fetch for {video_id} (clean={clean}, fmt={fmt})")
    
    # Strategy 1: Try YouTube API directly (fastest if not blocked)
    segments = try_youtube_api_direct(video_id)
    
    if segments:
        # Format the segments
        if fmt == "srt":
            return segments_to_srt(segments)
        if fmt == "vtt":
            return segments_to_vtt(segments)
        if clean:
            texts = [_get_segment_value(s, "text", "").replace("\n", " ") for s in segments]
            return _clean_plain_blocks(texts)
        return _format_timestamped(segments)

    # Strategy 2: Try yt-dlp fallback (more reliable on cloud providers)
    # Note: Try yt-dlp fallback (now supports SRT/VTT!)
    if fmt not in ("srt", "vtt"):
        logger.info(f"Trying yt-dlp fallback for {video_id} (format: {fmt})")
        ytdlp_result = try_ytdlp_fallback(video_id, clean=clean, fmt=fmt)  # ← Pass fmt!
        
        if ytdlp_result:
            logger.info(f"✅ yt-dlp fallback successful for {video_id}")
            return ytdlp_result
    else:
    logger.debug(f"Skipping yt-dlp for {video_id} - format {fmt} not supported by yt-dlp")
    
    # Strategy 3: Try with proxies (if configured)
    if use_proxies and PROXY_LIST:
        logger.info(f"Trying proxy-based requests for {video_id}")
        for proxy_url in PROXY_LIST:
            try:
                proxies = {
                    "http": proxy_url,
                    "https": proxy_url,
                }
                segments = try_youtube_api_direct(video_id, proxies=proxies)
                
                if segments:
                    if fmt == "srt":
                        return segments_to_srt(segments)
                    if fmt == "vtt":
                        return segments_to_vtt(segments)
                    if clean:
                        texts = [_get_segment_value(s, "text", "").replace("\n", " ") for s in segments]
                        return _clean_plain_blocks(texts)
                    return _format_timestamped(segments)
            except Exception as e:
                logger.debug(f"Proxy {proxy_url} failed: {e}")
                continue
    
    # All strategies failed
    error_msg = (
        f"Could not retrieve transcript for {video_id}. "
        "This may be due to:\n"
        "1. Video has no captions/transcripts available\n"
        "2. YouTube is blocking cloud provider IPs (common on Render/AWS)\n"
        "3. Video is region-restricted or private"
    )
    
    logger.error(error_msg)
    raise Exception(error_msg)


# Backward compatibility wrapper
def get_transcript_youtube_api(video_id: str, clean: bool = True, fmt: Optional[str] = None) -> str:
    """
    Legacy function name for backward compatibility.
    Calls the new smart fetching function.
    """
    return get_transcript_smart(video_id, clean=clean, fmt=fmt)