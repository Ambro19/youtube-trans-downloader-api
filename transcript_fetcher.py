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
from transcript_utils import _resolve_cookies_path, get_transcript_with_ytdlp

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
    if _resolve_cookies_path():
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
    if not _resolve_cookies_path():
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


# #############################===============================###########################
# # transcript_fetcher.py — PRODUCTION READY
# """
# Smart transcript fetching with multiple strategies to work around cloud IP blocks.

# Strategy order:
#   1) If cookies are configured -> yt-dlp (via transcript_utils)  [BEST behind bot/consent walls]
#   2) Else try YouTubeTranscriptApi                            [FAST when public captions exist]
#   3) If proxies provided, retry API with proxies
# """

# from __future__ import annotations
# import logging, os, re
# from typing import Optional, List, Dict, Any, Iterable

# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript

# # Utilities that already handle cookies and extractor hints for yt-dlp:
# from transcript_utils import _resolve_cookies_path, get_transcript_with_ytdlp

# logger = logging.getLogger("youtube_trans_downloader")

# # -----------------------
# # Config
# # -----------------------
# PROXY_LIST = os.getenv("YOUTUBE_PROXIES", "").split(",") if os.getenv("YOUTUBE_PROXIES") else []
# USE_PROXIES = len(PROXY_LIST) > 0
# _EN_PRIORITY = ["en", "en-US", "en-GB", "en-CA", "en-AU", "en-IE", "en-NZ"]

# # -----------------------
# # Small helpers / formatters
# # -----------------------
# def _get(seg: Any, key: str, default: Any = None) -> Any:
#     return seg.get(key, default) if isinstance(seg, dict) else getattr(seg, key, default)

# def _sec_to_vtt(ts: float) -> str:
#     h = int(ts // 3600); m = int((ts % 3600) // 60); s = int(ts % 60); ms = int((ts - int(ts)) * 1000)
#     return f"{h:02}:{m:02}:{s:02}.{ms:03}"

# def _sec_to_srt(ts: float) -> str:
#     h = int(ts // 3600); m = int((ts % 3600) // 60); s = int(ts % 60); ms = int((ts - int(ts)) * 1000)
#     return f"{h:02}:{m:02}:{s:02},{ms:03}"

# def segments_to_vtt(segments: List[Dict[str, Any]]) -> str:
#     out = ["WEBVTT", "Kind: captions", "Language: en", ""]
#     for seg in segments or []:
#         start = _get(seg, "start", 0.0); dur = _get(seg, "duration", 0.0); text = (_get(seg, "text", "") or "").strip()
#         if not text: 
#             continue
#         out.append(f"{_sec_to_vtt(start)} --> {_sec_to_vtt(start+dur)}")
#         out.append(text.replace("\n", " "))
#         out.append("")
#     return "\n".join(out)

# def segments_to_srt(segments: List[Dict[str, Any]]) -> str:
#     out: List[str] = []
#     idx = 1
#     for seg in segments or []:
#         start = _get(seg, "start", 0.0); dur = _get(seg, "duration", 0.0); text = (_get(seg, "text", "") or "").strip()
#         if not text:
#             continue
#         out += [str(idx), f"{_sec_to_srt(start)} --> {_sec_to_srt(start+dur)}", text.replace("\n", " "), ""]
#         idx += 1
#     return "\n".join(out)

# def _clean_plain_blocks(blocks: List[str]) -> str:
#     out, cur, chars = [], [], 0
#     for w in " ".join(blocks).split():
#         cur.append(w); chars += len(w) + 1
#         if chars > 400 and w[-1:] in ".!?":
#             out.append(" ".join(cur)); cur, chars = [], 0
#     if cur: out.append(" ".join(cur))
#     return "\n\n".join(out)

# def _format_timestamped(segments: List[Dict[str, Any]]) -> str:
#     lines: List[str] = []
#     for seg in segments or []:
#         start = float(_get(seg, "start", 0.0))
#         raw = (_get(seg, "text", "") or "").replace("\n", " ").strip()
#         if not raw:
#             continue
#         t = int(start)
#         lines.append(f"[{t//60:02d}:{t%60:02d}] {raw}")
#     return "\n".join(lines)

# #------------------
# def _try_ytdlp_with_cookies(video_id: str, clean: bool, fmt: Optional[str]) -> Optional[str]:
#     """Try yt-dlp with cookies first"""
#     if _resolve_cookies_path():
#         return get_transcript_with_ytdlp(video_id, clean=clean, fmt=fmt)
#     return None


# #---------------------
# def _try_direct_api(video_id: str, clean: bool, fmt: Optional[str]) -> Optional[str]:
#     """Try direct YouTube API"""
#     seg = _api_try(video_id)
#     if seg:
#         return _format_segments(seg, clean, fmt)
#     return None
# #--------------------
# def _try_api_with_proxies(video_id: str, clean: bool, fmt: Optional[str]) -> Optional[str]:
#     """Try API with proxy rotation"""
#     for proxy_url in PROXY_LIST:
#         proxies = {"http": proxy_url, "https": proxy_url}
#         seg = _api_try(video_id, proxies=proxies)
#         if seg:
#             return _format_segments(seg, clean, fmt)
#     return None

# #---------------
# def _try_ytdlp_without_cookies(video_id: str, clean: bool, fmt: Optional[str]) -> Optional[str]:
#     """Final fallback: yt-dlp without cookies"""
#     return get_transcript_with_ytdlp(video_id, clean=clean, fmt=fmt)
# #-----------------

# def try_youtube_api_direct(video_id: str, proxies: Optional[Dict] = None):
#     try:
#         kw = {"languages": _EN_PRIORITY}
#         if proxies: kw["proxies"] = proxies
#         seg = YouTubeTranscriptApi.get_transcript(video_id, **kw)
#         if seg: return seg
#     except Exception as e:
#         logger.debug("get_transcript failed: %s", e)
#     try:
#         kw = {}
#         if proxies: kw["proxies"] = proxies
#         listing = YouTubeTranscriptApi.list_transcripts(video_id, **kw)
#         for code in _EN_PRIORITY:
#             try:
#                 t = listing.find_transcript([code]); seg = t.fetch()
#                 if seg: return seg
#             except: pass
#         try:
#             t = listing.find_generated_transcript(_EN_PRIORITY); seg = t.fetch()
#             if seg: return seg
#         except: pass
#         for t in listing:
#             try:
#                 seg = t.translate("en").fetch()
#                 if seg: return seg
#             except: pass
#     except Exception as e:
#         logger.debug("list_transcripts failed: %s", e)
#     return None
# #-------------
# def _try_ytdlp_with_cookies(video_id: str, clean: bool, fmt: Optional[str]) -> Optional[str]:
#     """Try yt-dlp with cookies first"""
#     if _resolve_cookies_path():
#         return get_transcript_with_ytdlp(video_id, clean=clean, fmt=fmt)
#     return None

# #------------------------
# def _try_ytdlp_without_cookies(video_id: str, clean: bool, fmt: Optional[str]) -> Optional[str]:
#     """Final fallback: yt-dlp without cookies"""
#     return get_transcript_with_ytdlp(video_id, clean=clean, fmt=fmt)
# #------------------------    

# def try_ytdlp_fallback(video_id: str, clean: bool, fmt: Optional[str]) -> Optional[str]:
#     try:
#         from transcript_utils import get_transcript_with_ytdlp
#         return get_transcript_with_ytdlp(video_id, clean=clean, fmt=fmt)
#     except Exception as e:
#         logger.debug("yt-dlp fallback exception: %s", e)
#         return None

# def _normalize_video_id_or_url(s: str) -> str:
#     """Return a canonical URL for yt-dlp; accept bare IDs, watch URLs, shorts URLs."""
#     s = s.strip()
#     if re.fullmatch(r"[A-Za-z0-9_-]{10,}", s):
#         return f"https://www.youtube.com/watch?v={s}"
#     return s

# def _write_cookies_tmp() -> Optional[str]:
#     """
#     Provide a path to a cookies file for yt-dlp if available.
#     Priority:
#       - YT_COOKIES_B64 (base64 of a Netscape/Chrome-exported cookies.txt)
#       - YT_COOKIES_FILE (path already on disk)
#     Returns the path or None.
#     """
#     b64 = os.getenv("YT_COOKIES_B64", "").strip()
#     if b64:
#         try:
#             raw = base64.b64decode(b64)
#             f = tempfile.NamedTemporaryFile(prefix="yt_cookies_", suffix=".txt", delete=False)
#             f.write(raw)
#             f.flush()
#             f.close()
#             return f.name
#         except Exception as e:
#             logger.warning("Failed to decode YT_COOKIES_B64: %s", e)

#     fpath = os.getenv("YT_COOKIES_FILE", "").strip()
#     if fpath and os.path.exists(fpath):
#         return fpath

#     return None

# #------------------
# def _format_segments(segments: List[Dict[str, Any]], clean: bool, fmt: Optional[str]) -> str:
#     """Format segments based on requested format"""
#     if fmt == "srt":
#         return segments_to_srt(segments)
#     if fmt == "vtt":
#         return segments_to_vtt(segments)
#     if clean:
#         return _clean_plain_blocks([(_get(s, "text", "") or "").replace("\n", " ") for s in segments])
#     return _format_timestamped(segments) 

# #-------------------
# def _format_segments_clean(segments: List[Dict[str, Any]]) -> str:
#     """Flatten segments into a single clean text blob (no timestamps)."""
#     parts = []
#     for seg in segments:
#         text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
#         cleaned = text.replace("\n", " ").strip()
#         if cleaned:
#             parts.append(cleaned)
#     return "\n".join(parts)

# def _format_segments_srt(segments: List[Dict[str, Any]]) -> str:
#     """Minimal SRT output from a list of {start, duration, text} dicts."""
#     def srt_time(t: float) -> str:
#         # HH:MM:SS,mmm
#         ms = int(round((t - int(t)) * 1000))
#         t = int(t)
#         h, t = divmod(t, 3600)
#         m, s = divmod(t, 60)
#         return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

#     out = []
#     for i, seg in enumerate(segments, start=1):
#         start = float(seg.get("start", 0.0))
#         dur = float(seg.get("duration", 0.0))
#         end = start + dur
#         text = seg.get("text", "")
#         cleaned = text.replace("\n", " ").strip()
#         if not cleaned:
#             continue
#         out.append(str(i))
#         out.append(f"{srt_time(start)} --> {srt_time(end)}")
#         out.append(cleaned)
#         out.append("")  # blank line between cues
#     return "\n".join(out).strip()

# def _yt_dlp_transcript(video: str, want_format: Optional[str]) -> Optional[str]:
#     """
#     Use yt-dlp to fetch subtitles. want_format in {None, 'srt', 'vtt'}.
#     Returns text content or None if unavailable.
#     """
#     url = _normalize_video_id_or_url(video)
#     cookies_path = _write_cookies_tmp()

#     # Decide sub-format flags for yt-dlp
#     sub_fmt = "srt" if want_format == "srt" else "vtt" if want_format == "vtt" else "vtt"

#     with tempfile.TemporaryDirectory(prefix="ytdlp_subs_") as tmpdir:
#         base = os.path.join(tmpdir, "out")
#         # yt-dlp arguments: auto subs if manual not present; convert to desired format
#         args = [
#             "yt-dlp",
#             "--skip-download",
#             "--no-warnings",
#             "--quiet",
#             "--no-call-home",
#             "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
#             "--write-auto-subs",
#             "--sub-langs", "en.*,en",
#             "--sub-format", sub_fmt,
#             "--convert-subs", sub_fmt,
#             "-o", f"{base}.%(ext)s",
#             url,
#         ]
#         if cookies_path:
#             args.extend(["--cookies", cookies_path])

#         logger.info("Trying yt-dlp fallback for %s (format: %s)", video, want_format)
#         try:
#             proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
#         except Exception as e:
#             logger.error("yt-dlp failed to run: %s", e)
#             return None

#         if proc.returncode != 0:
#             # Helpful for debugging rate limit or bot checks
#             if proc.stderr:
#                 logger.error("yt-dlp stderr: %s", proc.stderr.strip())
#             return None

#         # Look for produced file
#         for ext in [sub_fmt, sub_fmt.upper()]:
#             candidate = f"{base}.{ext}"
#             if os.path.exists(candidate):
#                 with open(candidate, "r", encoding="utf-8", errors="ignore") as f:
#                     return f.read()

#     return None

# def fallback_auto_subs_with_ytdlp(video_id: str, lang: str = "en") -> list[dict]:
#     import subprocess, json, tempfile, os
#     # Ask yt_dlp for automatic subtitles (no download)
#     cmd = [
#         "yt-dlp",
#         f"https://www.youtube.com/watch?v={video_id}",
#         "--skip-download",
#         "--write-auto-sub",
#         "--sub-langs", lang,
#         "--sub-format", "json3",
#         "-J"  # dump metadata as JSON (contains subtitles)
#     ]
#     proc = subprocess.run(cmd, capture_output=True, text=True)
#     if proc.returncode != 0:  # no auto subs available
#         return []
#     data = json.loads(proc.stdout)
#     tracks = (data.get("subtitles") or {}) or (data.get("automatic_captions") or {})
#     track = (tracks.get(lang) or tracks.get(f"{lang}-orig") or [])
#     # Pick first JSON3 track and parse events -> [{'text': ..., 'start': ...}]
#     url = next((t.get("url") for t in track if "json3" in (t.get("ext") or "")), None)
#     if not url:
#         return []
#     # Fetch JSON3
#     import urllib.request
#     with urllib.request.urlopen(url) as r:
#         j3 = r.read().decode("utf-8")
#     j3 = json.loads(j3)
#     out = []
#     for ev in j3.get("events", []):
#         segs = ev.get("segs") or []
#         text = "".join(s.get("utf8","") for s in segs).strip()
#         if not text:
#             continue
#         t = ev.get("tStartMs", 0) // 1000
#         out.append({"text": text, "start": t})
#     return out

# #----------------------------------***********************---------

# def segments_to_text_timestamped(segments: Iterable[Any]) -> str:
#     """Generic timestamped formatter for mixed dict/obj segments."""
#     lines: list[str] = []
#     for seg in segments or []:
#         if isinstance(seg, dict):
#             text = seg.get("text") or seg.get("utf8") or ""
#             start = seg.get("start", seg.get("start_time", seg.get("tStartMs")))
#         else:
#             text = getattr(seg, "text", "") or getattr(seg, "utf8", "")
#             start = getattr(seg, "start", getattr(seg, "start_time", getattr(seg, "tStartMs", None)))
#         # normalize seconds
#         if start is None:
#             t_seconds = 0
#         else:
#             try:
#                 val = float(start)
#                 t_seconds = int(round(val / 1000.0)) if val > 10_000 else int(round(val))
#             except Exception:
#                 t_seconds = 0
#         safe = (text or "").replace("\n", " ").strip()
#         if not safe:
#             continue
#         lines.append(f"[{t_seconds // 60:02d}:{t_seconds % 60:02d}] {safe}")
#     return "\n".join(lines)

# # -----------------------
# # Direct API path
# # -----------------------
# def _api_try(video_id: str, proxies: Optional[Dict[str, str]] = None) -> Optional[List[Dict[str, Any]]]:
#     """Try the YouTubeTranscriptApi once, returning segments or None."""
#     try:
#         kw: Dict[str, Any] = {}
#         if proxies:
#             kw["proxies"] = proxies

#         # Work with bare ID for API calls
#         m = re.search(r"[A-Za-z0-9_-]{10,}", video_id)
#         bare_id = m.group(0) if m else video_id

#         listing = YouTubeTranscriptApi.list_transcripts(bare_id, **kw)
#         # prefer manual English
#         try:
#             t = listing.find_manually_created_transcript(['en', 'en-US'])
#         except Exception:
#             try:
#                 t = listing.find_transcript(['en', 'en-US'])
#             except Exception:
#                 t = listing.find_generated_transcript(['en', 'en-US'])
#         seg = t.fetch()
#         return seg or None
#     except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
#         return None
#     except Exception as e:
#         logger.debug("API path failed: %s", e)
#         return None
# #---------------
# def _try_direct_api(video_id: str, clean: bool, fmt: Optional[str]) -> Optional[str]:
#     """Try direct YouTube API"""
#     seg = _api_try(video_id)
#     if seg:
#         return _format_segments(seg, clean, fmt)
#     return None
# #--------------


# # -----------------------
# # Public entrypoint
# # -----------------------
# def get_transcript_smart(
#     video_id_or_url: str,
#     clean: bool = True,
#     fmt: Optional[str] = None,
#     use_proxies: bool = USE_PROXIES
# ) -> str:
#     """
#     Enhanced with multiple fallback strategies
#     """
#     vid = (video_id_or_url or "").strip()
#     logger.info("Smart fetch %s (clean=%s, fmt=%s)", vid, clean, fmt)

#     strategies = [
#         ("yt-dlp with cookies", _try_ytdlp_with_cookies),
#         ("direct API", _try_direct_api),
#         ("yt-dlp without cookies", _try_ytdlp_without_cookies),
#     ]

#     if use_proxies and PROXY_LIST:
#         strategies.insert(1, ("API with proxies", _try_api_with_proxies))

#     for strategy_name, strategy_func in strategies:
#         try:
#             logger.info("Trying strategy: %s", strategy_name)
#             result = strategy_func(vid, clean, fmt)
#             if result:
#                 logger.info("Strategy %s succeeded", strategy_name)
#                 return result
#         except Exception as e:
#             logger.debug("Strategy %s failed: %s", strategy_name, e)
#             continue

#     raise Exception("All transcript fetch strategies failed")

# # ## ---------------End transcript_utils Module-----------------
