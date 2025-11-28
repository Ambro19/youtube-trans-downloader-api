# transcript_utils.py — WITH PROXY SUPPORT
"""
Enhanced with residential proxy support to bypass YouTube IP blocking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import timedelta
import logging
import os
import re
import tempfile
import io
import json
import base64

from yt_dlp import YoutubeDL

logger = logging.getLogger("youtube_trans_downloader")

# ======================================================
# Cookies helpers (unchanged)
# ======================================================

COOKIES_FILE_ENV = (os.getenv("YT_COOKIES_FILE") or "").strip()
COOKIES_B64_ENV = (os.getenv("YT_COOKIES_B64") or "").strip()
YTDLP_DIR_ENV = os.getenv("YT_DLP_DIR") or "/tmp/yt-dlp"

_COOKIES_CACHE: Optional[str] = None


def _get_cookies_file() -> Optional[str]:
    """Returns a readable cookies file path for yt-dlp, or None."""
    global _COOKIES_CACHE

    if _COOKIES_CACHE is not None:
        if _COOKIES_CACHE and os.path.exists(_COOKIES_CACHE):
            return _COOKIES_CACHE
        _COOKIES_CACHE = None

    # 1) Prefer base64 env var
    if COOKIES_B64_ENV:
        try:
            target_dir = Path(YTDLP_DIR_ENV)
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "cookies.txt"

            if target.exists():
                target.unlink()
            
            decoded = base64.b64decode(COOKIES_B64_ENV)
            decoded = decoded.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            
            with open(target, "wb") as f:
                f.write(decoded)
            
            if target.exists() and target.stat().st_size > 0:
                _COOKIES_CACHE = str(target)
                logger.info("✅ Decoded YT_COOKIES_B64 to %s (%d bytes)", target, target.stat().st_size)
                return _COOKIES_CACHE
            else:
                logger.warning("❌ Cookie file created but appears empty")
                
        except Exception as e:
            logger.error("Failed to decode YT_COOKIES_B64: %s", e, exc_info=True)

    # 2) Fallback to file
    if COOKIES_FILE_ENV:
        if os.path.exists(COOKIES_FILE_ENV) and os.access(COOKIES_FILE_ENV, os.R_OK):
            if COOKIES_FILE_ENV.startswith(("/etc/", "/run/")):
                try:
                    target_dir = Path(YTDLP_DIR_ENV)
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target = target_dir / "cookies.txt"
                    
                    import shutil
                    shutil.copyfile(COOKIES_FILE_ENV, target)
                    _COOKIES_CACHE = str(target)
                    logger.info("✅ Copied YT_COOKIES_FILE from %s to %s", COOKIES_FILE_ENV, target)
                    return _COOKIES_CACHE
                except Exception as e:
                    logger.warning("Could not copy YT_COOKIES_FILE to /tmp: %s", e)
            else:
                _COOKIES_CACHE = COOKIES_FILE_ENV
                logger.info("✅ Using cookies from YT_COOKIES_FILE=%s", COOKIES_FILE_ENV)
                return _COOKIES_CACHE
        else:
            logger.warning("YT_COOKIES_FILE=%s not readable or missing", COOKIES_FILE_ENV)

    logger.warning("⚠️  No cookies configured (YT_COOKIES_FILE / YT_COOKIES_B64).")
    _COOKIES_CACHE = None
    return None


# ======================================================
# ENHANCED: Apply proxy + cookie options
# ======================================================

def _apply_cookie_opts(opts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced to include proxy support (residential or mobile).
    
    CRITICAL: Proxies bypass YouTube's datacenter IP blocking.
    Mobile proxies recommended for YouTube (higher success rate).
    """
    cp = _get_cookies_file()
    if cp:
        opts["cookiefile"] = cp
        logger.info("🍪 Using cookies file: %s", cp)
    else:
        logger.warning("⚠️  No cookies available")

    # === PROXY CONFIGURATION (NEW) ===
    proxy_enabled = os.getenv("PROXY_ENABLED", "false").lower() == "true"
    if proxy_enabled:
        proxy_host = os.getenv("PROXY_HOST")
        proxy_port = os.getenv("PROXY_PORT")
        proxy_user = os.getenv("PROXY_USERNAME")
        proxy_pass = os.getenv("PROXY_PASSWORD")
        
        if all([proxy_host, proxy_port, proxy_user, proxy_pass]):
            # Build proxy URL
            proxy_url = f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"
            opts["proxy"] = proxy_url
            
            # Mask password in logs
            safe_user = proxy_user[:15] + "..." if len(proxy_user) > 15 else proxy_user
            logger.info("🌐 Using proxy: %s:****@%s:%s", 
                       safe_user, proxy_host, proxy_port)
        else:
            logger.warning("⚠️  PROXY_ENABLED=true but missing credentials")
            logger.warning("    Required: PROXY_HOST, PROXY_PORT, PROXY_USERNAME, PROXY_PASSWORD")
    else:
        logger.info("ℹ️  Proxy disabled (PROXY_ENABLED=false)")
    
    # Force IPv4
    if os.getenv("YTDLP_BIND_IPV4", "1").strip() == "1":
        opts["force_ipv4"] = True

    # Enhanced headers
    opts.setdefault("http_headers", {})
    opts["http_headers"].update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    })

    # Player client selection
    ea = opts.setdefault("extractor_args", {})
    youtube_args = ea.setdefault("youtube", {})
    
    # Use web client with proxy (better compatibility)
    if proxy_enabled:
        youtube_args["player_client"] = ["web", "android"]
    else:
        youtube_args["player_client"] = ["android", "web"]
    
    # PO Token support (optional)
    po_token_android = os.getenv("YT_PO_TOKEN_ANDROID")
    po_token_web = os.getenv("YT_PO_TOKEN_WEB")
    
    if po_token_android or po_token_web:
        if po_token_android:
            youtube_args["po_token"] = f"android.{po_token_android}"
            logger.info("🔐 Using Android PO token")
        elif po_token_web:
            youtube_args["po_token"] = f"web.{po_token_web}"
            logger.info("🔐 Using Web PO token")
    
    # Skip DASH
    youtube_args.setdefault("skip", ["dash", "hls"])
    
    # Resilience
    opts.setdefault("socket_timeout", 30)
    opts.setdefault("retries", 10)
    opts.setdefault("fragment_retries", 10)
    opts.setdefault("file_access_retries", 5)
    opts.setdefault("extractor_retries", 3)
    
    # Logging
    opts.setdefault("quiet", True)
    opts.setdefault("no_warnings", False)
    
    return opts


# ======================================================
# Rest of functions (unchanged)
# ======================================================

def _norm_youtube_url(video_id_or_url: str) -> str:
    """Accept ID or any YT URL."""
    s = (video_id_or_url or "").strip()
    m = re.search(r"(?:v=|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})", s)
    vid = m.group(1) if m else s
    return f"https://www.youtube.com/watch?v={vid}"


def _ensure_ffmpeg_location() -> Optional[str]:
    """Get ffmpeg path if available."""
    ff = os.getenv("FFMPEG_PATH")
    if ff and Path(ff).exists():
        return ff
    return None


def _safe_outtmpl(
    output_dir: str,
    stem: str = "%(title).200B [%(id)s]",
    ext_placeholder: str = "%(ext)s",
) -> Dict[str, str]:
    """Create safe output template."""
    return {"default": os.path.join(output_dir, f"{stem}.{ext_placeholder}")}


def check_ytdlp_availability() -> bool:
    """Check if yt-dlp is available."""
    try:
        import yt_dlp  # noqa: F401
        return True
    except Exception as e:
        logger.debug("yt-dlp not available: %s", e)
        return False


def _mmss(seconds: float) -> str:
    """Format seconds as MM:SS."""
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def _parse_vtt_to_segments(vtt_text: str) -> List[Dict[str, Any]]:
    """Parse WebVTT to segments."""
    ts_re = re.compile(
        r"(?P<s>\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*"
        r"(?P<e>\d{2}:\d{2}:\d{2}\.\d{3})"
    )

    def _to_seconds(hhmmss_ms: str) -> float:
        hh, mm, ss_ms = hhmmss_ms.split(":")
        ss, ms = ss_ms.split(".")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0

    segs: List[Dict[str, Any]] = []
    buf: List[str] = []
    start = end = None

    for raw in io.StringIO(vtt_text):
        line = raw.strip("\n")
        m = ts_re.search(line)
        if m:
            if start is not None and buf:
                text = " ".join(b for b in buf if b).strip()
                if text:
                    s_val = _to_seconds(start)
                    e_val = _to_seconds(end)
                    segs.append({
                        "start": s_val,
                        "duration": max(0.0, e_val - s_val),
                        "text": text,
                    })
            start, end = m.group("s"), m.group("e")
            buf = []
            continue

        if not line.strip():
            if start is not None and buf:
                text = " ".join(b for b in buf if b).strip()
                if text:
                    s_val = _to_seconds(start)
                    e_val = _to_seconds(end)
                    segs.append({
                        "start": s_val,
                        "duration": max(0.0, e_val - s_val),
                        "text": text,
                    })
            start = end = None
            buf = []
            continue

        if start is not None:
            buf.append(line)

    if start is not None and buf:
        text = " ".join(b for b in buf if b).strip()
        if text:
            s_val = _to_seconds(start)
            e_val = _to_seconds(end)
            segs.append({
                "start": s_val,
                "duration": max(0.0, e_val - s_val),
                "text": text,
            })

    return segs


def _clean_plain_blocks(blocks: List[str]) -> str:
    """Format plain text."""
    out: List[str] = []
    cur: List[str] = []
    chars = 0
    for word in " ".join(blocks).split():
        cur.append(word)
        chars += len(word) + 1
        if chars > 400 and word[-1:] in ".!?":
            out.append(" ".join(cur))
            cur, chars = [], 0
    if cur:
        out.append(" ".join(cur))
    return "\n\n".join(out)


# ======================================================
# Transcript (enhanced with proxy)
# ======================================================

def get_transcript_with_ytdlp(
    video_id_or_url: str,
    clean: bool = True,
    fmt: Optional[str] = None,
) -> Optional[str]:
    """
    Extract subtitles using yt-dlp with proxy support.
    """
    want_fmt = "srt" if fmt == "srt" else "vtt"
    lang_priority = ["en", "en-US", "en-GB", "en-CA", "en-AU"]

    with tempfile.TemporaryDirectory(prefix="ytdlp_subs_") as tmp:
        ydl_opts: Dict[str, Any] = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": want_fmt,
            "subtitleslangs": lang_priority,
            "outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"),
            "quiet": False,
            "no_warnings": False,
            "ignoreerrors": False,
        }
        _apply_cookie_opts(ydl_opts)  # Now includes proxy!

        url = _norm_youtube_url(video_id_or_url)
        vid = video_id_or_url.strip()
        
        logger.info("🎬 Fetching transcript for %s (format=%s, clean=%s)", vid, fmt, clean)
        
        try:
            with YoutubeDL(ydl_opts) as ydl:
                result = ydl.download([url])
                logger.debug("yt-dlp download result: %s", result)
        except Exception as e:
            error_msg = str(e).lower()
            logger.error("yt-dlp transcript fetch failed for %s: %s", vid, e)
            
            if "sign in to confirm" in error_msg or "bot" in error_msg:
                logger.error("❌ YouTube bot detection - proxy may be needed or blocked")
                return None
            elif "no suitable formats" in error_msg or "no subtitles" in error_msg:
                logger.warning("⚠️  No captions available for video %s", vid)
                return None
            else:
                logger.error("❌ Unknown error: %s", e)
                return None

        sub_files = list(Path(tmp).glob(f"*.{want_fmt}"))
        if not sub_files:
            logger.warning("⚠️  No subtitle files produced for %s", vid)
            return None

        content = sub_files[0].read_text(encoding="utf-8", errors="ignore")
        logger.info("✅ Retrieved transcript (%d chars) for %s", len(content), vid)

        if fmt in ("srt", "vtt"):
            return content

        segments = _parse_vtt_to_segments(content)
        if not segments:
            logger.warning("⚠️  No segments parsed from transcript")
            return None

        if clean:
            texts = [
                (s.get("text") or "").replace("\n", " ").strip()
                for s in segments
            ]
            return _clean_plain_blocks(texts)

        lines: List[str] = []
        for s in segments:
            t = int(float(s.get("start", 0)))
            txt = (s.get("text") or "").replace("\n", " ").strip()
            if not txt:
                continue
            lines.append(f"[{t // 60:02d}:{t % 60:02d}] {txt}")
        return "\n".join(lines)


# ======================================================
# Video / Audio (enhanced with proxy)
# ======================================================

def _common_ydl_opts(output_dir: str) -> Dict[str, Any]:
    """Common yt-dlp options with proxy support."""
    ffmpeg_loc = _ensure_ffmpeg_location()
    opts: Dict[str, Any] = {
        "format": (
            "bv*[ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
            "bv*+ba/best"
        ),
        "merge_output_format": "mp4",
        "postprocessors": [
            {"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"},
            {"key": "FFmpegMetadata"},
            {"key": "EmbedThumbnail", "already_have_thumbnail": False},
        ],
        "writethumbnail": True,
        "outtmpl": _safe_outtmpl(output_dir),
        "noprogress": True,
        "quiet": False,
        "concurrent_fragment_downloads": 4,
        "retries": 10,
        "fragment_retries": 10,
        "file_access_retries": 5,
    }
    if ffmpeg_loc:
        opts["ffmpeg_location"] = ffmpeg_loc
    _apply_cookie_opts(opts)  # Now includes proxy!
    return opts


def get_video_info(video_id_or_url: str) -> Dict[str, Any]:
    """Get video metadata with proxy support."""
    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    _apply_cookie_opts(ydl_opts)  # Now includes proxy!
    
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                _norm_youtube_url(video_id_or_url),
                download=False,
            )
            return {
                "id": info.get("id"),
                "title": info.get("title"),
                "uploader": info.get("uploader") or info.get("channel"),
                "duration": info.get("duration"),
            }
    except Exception as e:
        logger.error("Failed to get video info: %s", e)
        return {}


def download_audio_with_ytdlp(
    video_id_or_url: str,
    quality: str,
    output_dir: str,
) -> str:
    """Download audio with proxy support."""
    q = (quality or "").lower()
    kbps = "96" if q in {"low", "l"} else "256" if q in {"high", "h"} else "160"

    opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": kbps,
            },
            {"key": "FFmpegMetadata"},
            {"key": "EmbedThumbnail", "already_have_thumbnail": False},
        ],
        "outtmpl": _safe_outtmpl(output_dir),
        "noprogress": True,
        "quiet": False,
        "writethumbnail": True,
        "retries": 10,
    }
    ffmpeg_loc = _ensure_ffmpeg_location()
    if ffmpeg_loc:
        opts["ffmpeg_location"] = ffmpeg_loc
    _apply_cookie_opts(opts)  # Now includes proxy!

    os.makedirs(output_dir, exist_ok=True)
    url = _norm_youtube_url(video_id_or_url)
    
    logger.info("🎵 Downloading audio for %s (quality=%s)", video_id_or_url, quality)
    
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            base = ydl.prepare_filename(info)
            mp3_path = base.rsplit(".", 1)[0] + ".mp3"
            logger.info("✅ Audio downloaded: %s", mp3_path)
            return mp3_path
    except Exception as e:
        logger.error("❌ Audio download failed: %s", e)
        raise


def download_video_with_ytdlp(
    video_id_or_url: str,
    quality: str,
    output_dir: str,
) -> str:
    """Download video with proxy support."""
    q = re.sub(r"[^0-9]", "", quality or "")
    height = int(q) if q.isdigit() else None
    
    opts = _common_ydl_opts(output_dir)
    if height:
        opts["format"] = (
            f"bv*[height={height}][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
            f"bv*[height<={height}][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
            "bv*+ba/best"
        )

    os.makedirs(output_dir, exist_ok=True)
    url = _norm_youtube_url(video_id_or_url)
    
    logger.info("🎬 Downloading video for %s (quality=%s)", video_id_or_url, quality)
    
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            base, _ = os.path.splitext(ydl.prepare_filename(info))
            mp4 = base + ".mp4"
            result = mp4 if os.path.exists(mp4) else ydl.prepare_filename(info)
            logger.info("✅ Video downloaded: %s", result)
            return result
    except Exception as e:
        logger.error("❌ Video download failed: %s", e)
        raise