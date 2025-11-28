# transcript_fetcher.py — PRODUCTION READY (FIXED)
"""
Smart transcript fetching with multiple strategies to work around cloud IP blocks.

Strategy order (IMPROVED):
  1) If cookies configured → yt-dlp first (BEST for bot/consent walls)
  2) Try YouTubeTranscriptApi (fast when public captions exist and IP not blocked)
  3) Retry with yt-dlp if API fails
  4) Optional: retry API with proxies (if configured)

IMPROVEMENTS:
- Better error differentiation between "no captions" vs "blocked"
- More resilient fallback logic
- Enhanced logging for debugging
- FIXED: Import _get_cookies_file instead of _resolve_cookies_path
"""

from __future__ import annotations
import logging
import os
import re
from typing import Optional, List, Dict, Any, Iterable

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)

# Utilities that handle cookies and extractor hints for yt-dlp:
# FIXED: Use _get_cookies_file instead of _resolve_cookies_path
from transcript_utils import _get_cookies_file, get_transcript_with_ytdlp

logger = logging.getLogger("youtube_trans_downloader")

# -----------------------
# Config
# -----------------------
PROXY_LIST = os.getenv("YOUTUBE_PROXIES", "").split(",") if os.getenv("YOUTUBE_PROXIES") else []
USE_PROXIES = len(PROXY_LIST) > 0
_EN_PRIORITY = ["en", "en-US", "en-GB", "en-CA", "en-AU", "en-IE", "en-NZ"]

# -----------------------
# Small helpers / formatters
# -----------------------
def _get(seg: Any, key: str, default: Any = None) -> Any:
    """Safely get value from segment (dict or object)."""
    return seg.get(key, default) if isinstance(seg, dict) else getattr(seg, key, default)


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


def segments_to_vtt(segments: List[Dict[str, Any]]) -> str:
    """Convert segments to WebVTT format."""
    out = ["WEBVTT", "Kind: captions", "Language: en", ""]
    for seg in segments or []:
        start = _get(seg, "start", 0.0)
        dur = _get(seg, "duration", 0.0)
        text = (_get(seg, "text", "") or "").strip()
        if not text:
            continue
        out.append(f"{_sec_to_vtt(start)} --> {_sec_to_vtt(start + dur)}")
        out.append(text.replace("\n", " "))
        out.append("")
    return "\n".join(out)


def segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    """Convert segments to SRT format."""
    out: List[str] = []
    idx = 1
    for seg in segments or []:
        start = _get(seg, "start", 0.0)
        dur = _get(seg, "duration", 0.0)
        text = (_get(seg, "text", "") or "").strip()
        if not text:
            continue
        out += [
            str(idx),
            f"{_sec_to_srt(start)} --> {_sec_to_srt(start + dur)}",
            text.replace("\n", " "),
            ""
        ]
        idx += 1
    return "\n".join(out)


def _clean_plain_blocks(blocks: List[str]) -> str:
    """Format plain text into readable paragraphs."""
    out, cur, chars = [], [], 0
    for w in " ".join(blocks).split():
        cur.append(w)
        chars += len(w) + 1
        if chars > 400 and w[-1:] in ".!?":
            out.append(" ".join(cur))
            cur, chars = [], 0
    if cur:
        out.append(" ".join(cur))
    return "\n\n".join(out)


def _format_timestamped(segments: List[Dict[str, Any]]) -> str:
    """Format segments with timestamps [MM:SS] text."""
    lines: List[str] = []
    for seg in segments or []:
        start = float(_get(seg, "start", 0.0))
        raw = (_get(seg, "text", "") or "").replace("\n", " ").strip()
        if not raw:
            continue
        t = int(start)
        lines.append(f"[{t // 60:02d}:{t % 60:02d}] {raw}")
    return "\n".join(lines)


def segments_to_text_timestamped(segments: Iterable[Any]) -> str:
    """Generic timestamped formatter for mixed dict/obj segments."""
    lines: list[str] = []
    for seg in segments or []:
        if isinstance(seg, dict):
            text = seg.get("text", "") or seg.get("utf8", "") or ""
            start = seg.get("start", seg.get("start_time", seg.get("tStartMs")))
        else:
            text = getattr(seg, "text", "") or getattr(seg, "utf8", "") or ""
            start = getattr(seg, "start", getattr(seg, "start_time", getattr(seg, "tStartMs", None)))
        
        # Normalize seconds
        if start is None:
            t_seconds = 0
        else:
            try:
                val = float(start)
                t_seconds = int(round(val / 1000.0)) if val > 10_000 else int(round(val))
            except Exception:
                t_seconds = 0
        
        safe = (text or "").replace("\n", " ").strip()
        if not safe:
            continue
        lines.append(f"[{t_seconds // 60:02d}:{t_seconds % 60:02d}] {safe}")
    return "\n".join(lines)


# -----------------------
# API Helper Functions
# -----------------------

def _try_youtube_api(video_id: str, proxies: Optional[Dict[str, str]] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Try YouTubeTranscriptApi with various strategies.
    Returns segments or None if failed.
    """
    try:
        # Clean video ID
        m = re.search(r"[A-Za-z0-9_-]{11}", video_id)
        bare_id = m.group(0) if m else video_id
        
        kw: Dict[str, Any] = {}
        if proxies:
            kw["proxies"] = proxies
        
        # Strategy 1: Direct get_transcript with language priority
        try:
            logger.debug("📡 Trying API direct get_transcript for %s", bare_id)
            seg = YouTubeTranscriptApi.get_transcript(bare_id, languages=_EN_PRIORITY, **kw)
            if seg:
                logger.info("✅ API direct get_transcript succeeded (%d segments)", len(seg))
                return seg
        except Exception as e:
            logger.debug("API direct failed: %s", e)
        
        # Strategy 2: List transcripts and pick best
        try:
            logger.debug("📡 Trying API list_transcripts for %s", bare_id)
            listing = YouTubeTranscriptApi.list_transcripts(bare_id, **kw)
            
            # Prefer manual English
            for lang in _EN_PRIORITY:
                try:
                    t = listing.find_manually_created_transcript([lang])
                    seg = t.fetch()
                    if seg:
                        logger.info("✅ API found manual %s transcript (%d segments)", lang, len(seg))
                        return seg
                except Exception:
                    pass
            
            # Try generated English
            try:
                t = listing.find_generated_transcript(_EN_PRIORITY)
                seg = t.fetch()
                if seg:
                    logger.info("✅ API found generated EN transcript (%d segments)", len(seg))
                    return seg
            except Exception:
                pass
            
            # Last resort: translate any available transcript
            for t in listing:
                try:
                    seg = t.translate("en").fetch()
                    if seg:
                        logger.info("✅ API translated %s→en (%d segments)", getattr(t, "language_code", "?"), len(seg))
                        return seg
                except Exception:
                    pass
                    
        except Exception as e:
            logger.debug("API list_transcripts failed: %s", e)
        
        return None
        
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        logger.debug("API: No transcript available for %s: %s", video_id, e)
        return None
    except CouldNotRetrieveTranscript as e:
        logger.warning("API: Could not retrieve transcript for %s: %s", video_id, e)
        return None
    except Exception as e:
        logger.error("API: Unexpected error for %s: %s", video_id, e)
        return None


# -----------------------
# Public entrypoint - IMPROVED
# -----------------------

def get_transcript_smart(
    video_id_or_url: str,
    clean: bool = True,
    fmt: Optional[str] = None,
    use_proxies: bool = USE_PROXIES
) -> str:
    """
    Smart transcript fetching with multiple fallback strategies.
    
    Returns one of:
      - Clean TXT (default, no timestamps)
      - Timestamped TXT (when clean=False and fmt is None)
      - SRT / VTT (when fmt == 'srt' or 'vtt')
    
    Raises Exception if nothing could be fetched.
    
    IMPROVED: Better strategy ordering and error messages.
    """
    vid = (video_id_or_url or "").strip()
    logger.info("🔍 Smart fetch %s (clean=%s, fmt=%s)", vid, clean, fmt)
    
    errors = []  # Track all errors for better debugging
    
    # Strategy 1: If cookies exist, prefer yt-dlp (handles bot walls best)
    # FIXED: Use _get_cookies_file() instead of _resolve_cookies_path()
    if _get_cookies_file():
        logger.info("🍪 Cookies available - trying yt-dlp first")
        try:
            text = get_transcript_with_ytdlp(vid, clean=clean, fmt=fmt)
            if text:
                logger.info("✅ yt-dlp succeeded (with cookies)")
                return text
            else:
                errors.append("yt-dlp: returned empty")
                logger.warning("⚠️  yt-dlp returned empty, trying API fallback")
        except Exception as e:
            errors.append(f"yt-dlp: {str(e)}")
            logger.warning("⚠️  yt-dlp failed: %s, trying API fallback", e)
    else:
        logger.info("⚠️  No cookies - skipping yt-dlp, trying API first")
    
    # Strategy 2: Try YouTube Transcript API (fast when it works)
    logger.info("📡 Trying YouTube Transcript API")
    seg = _try_youtube_api(vid)
    if seg:
        logger.info("✅ API succeeded, formatting response")
        # Format the response
        if fmt == "srt":
            return segments_to_srt(seg)
        if fmt == "vtt":
            return segments_to_vtt(seg)
        if clean:
            texts = [(_get(s, "text", "") or "").replace("\n", " ") for s in seg]
            return _clean_plain_blocks(texts)
        return _format_timestamped(seg)
    else:
        errors.append("API: no segments returned")
    
    # Strategy 3: Retry with proxies if configured
    if use_proxies and PROXY_LIST:
        logger.info("🔄 Trying API with proxies")
        for i, proxy_url in enumerate(PROXY_LIST, 1):
            proxies = {"http": proxy_url, "https": proxy_url}
            logger.debug("Trying proxy %d/%d: %s", i, len(PROXY_LIST), proxy_url)
            seg = _try_youtube_api(vid, proxies=proxies)
            if seg:
                logger.info("✅ API+proxy succeeded")
                if fmt == "srt":
                    return segments_to_srt(seg)
                if fmt == "vtt":
                    return segments_to_vtt(seg)
                if clean:
                    texts = [(_get(s, "text", "") or "").replace("\n", " ") for s in seg]
                    return _clean_plain_blocks(texts)
                return _format_timestamped(seg)
        errors.append(f"API+proxies: tried {len(PROXY_LIST)} proxies, all failed")
    
    # Strategy 4: Final attempt with yt-dlp (even without cookies)
    # FIXED: Use _get_cookies_file() instead of _resolve_cookies_path()
    if not _get_cookies_file():
        logger.info("🔄 Final attempt: yt-dlp without cookies")
        try:
            text = get_transcript_with_ytdlp(vid, clean=clean, fmt=fmt)
            if text:
                logger.info("✅ yt-dlp succeeded (without cookies)")
                return text
            errors.append("yt-dlp (no cookies): returned empty")
        except Exception as e:
            errors.append(f"yt-dlp (no cookies): {str(e)}")
            logger.warning("⚠️  yt-dlp (no cookies) failed: %s", e)
    
    # All strategies failed
    error_summary = "; ".join(errors)
    logger.error("❌ All strategies failed for %s. Errors: %s", vid, error_summary)
    
    # Provide helpful error message based on what failed
    if any("bot" in e.lower() or "sign in" in e.lower() for e in errors):
        raise Exception(
            "YouTube is blocking transcript access from cloud servers. "
            "This video may not have captions, or your cookies may be expired. "
            "Please try a different video."
        )
    elif any("no transcript" in e.lower() or "disabled" in e.lower() for e in errors):
        raise Exception(
            "This video does not have captions/transcripts available. "
            "Please try a different video."
        )
    else:
        raise Exception(
            "Could not retrieve transcript (no captions or YouTube blocked our requests)."
        )