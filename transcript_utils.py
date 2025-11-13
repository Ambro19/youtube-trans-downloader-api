# transcript_utils.py — PRODUCTION READY
"""
Utilities for transcripts and media downloads.
Framework-free to avoid circular imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from datetime import timedelta
import logging, os, re, tempfile, io, json, base64

from yt_dlp import YoutubeDL

logger = logging.getLogger("youtube_trans_downloader")

# -----------------------
# Cookies helpers
# -----------------------
_COOKIES_CACHE: Optional[str] = None

def _resolve_cookies_path() -> Optional[str]:
    """
    Returns a filesystem path to a Netscape cookie file if configured,
    handling either YT_COOKIES_FILE or YT_COOKIES_B64.
    """
    global _COOKIES_CACHE
    if _COOKIES_CACHE:
        return _COOKIES_CACHE if os.path.exists(_COOKIES_CACHE) else None

    # 1) direct file
    p = (os.getenv("YT_COOKIES_FILE") or "").strip()
    if p and os.path.exists(p):
        _COOKIES_CACHE = p
        logger.info(f"▶ Using cookies file: {p}")
        return p

    # 2) base64
    b64 = (os.getenv("YT_COOKIES_B64") or "").strip()
    if b64:
        try:
            raw = base64.b64decode(b64)
            # yt-dlp expects Unix line endings; normalize
            raw = raw.replace(b"\r\n", b"\n")
            tmp = tempfile.NamedTemporaryFile(prefix="yt_cookies_", suffix=".txt", delete=False)
            tmp.write(raw)
            tmp.flush(); tmp.close()
            _COOKIES_CACHE = tmp.name
            logger.info(f"▶ Decoded cookies from YT_COOKIES_B64 into {tmp.name}")
            return _COOKIES_CACHE
        except Exception as e:
            logger.warning(f"Could not decode YT_COOKIES_B64: {e}")

    logger.info("▶ No cookies configured (YT_COOKIES_FILE / YT_COOKIES_B64).")
    return None

def _apply_cookie_opts(opts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mutates and returns yt-dlp options to include cookies and a couple of
    pragmatic extractor hints that reduce YouTube consent/bot walls.
    """
    cp = _resolve_cookies_path()
    if cp:
        opts["cookiefile"] = cp

    # Align with your env preference to avoid IPv6 issues
    if os.getenv("YTDLP_BIND_IPV4", "1").strip() == "1":
        # Python API flag for force ipv4
        opts["force_ipv4"] = True

    # These headers + extractor args help on Shorts/consent walls
    opts.setdefault("http_headers", {})
    opts["http_headers"].setdefault(
        "User-Agent",
        "com.google.android.youtube/19.20.34 (Linux; U; Android 11)"
    )
    opts["http_headers"].setdefault("Accept-Language", "en-US,en;q=0.9")

    # Prefer the Android player client path
    ea = opts.setdefault("extractor_args", {})
    youtube_args = ea.setdefault("youtube", {})
    # allow merge or set
    existing = youtube_args.get("player_client", [])
    if not existing:
        youtube_args["player_client"] = ["android", "android_embedded"]

    return opts

# -----------------------
# Common helpers
# -----------------------
def _norm_youtube_url(video_id_or_url: str) -> str:
    """Accept ID or any YT URL (watch/shorts/youtu.be)."""
    s = (video_id_or_url or "").strip()
    m = re.search(r'(?:v=|/shorts/|youtu\.be/)([A-Za-z0-9_-]{6,})', s)
    vid = m.group(1) if m else s
    return f"https://www.youtube.com/watch?v={vid}"

def _ensure_ffmpeg_location() -> Optional[str]:
    ff = os.getenv("FFMPEG_PATH")
    if ff and Path(ff).exists():
        return ff
    return None

def _safe_outtmpl(output_dir: str, stem: str = "%(title).200B [%(id)s]", ext_placeholder: str = "%(ext)s") -> Dict[str, str]:
    return {"default": os.path.join(output_dir, f"{stem}.{ext_placeholder}")}

def check_ytdlp_availability() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except Exception as e:
        logger.debug("yt-dlp not available: %s", e)
        return False

def _mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s//60:02d}:{s%60:02d}"

def _parse_vtt_to_segments(vtt_text: str) -> List[Dict[str, Any]]:
    """
    Parse a WebVTT string into segments: [{'start': float, 'duration': float, 'text': str}, ...]
    Robust enough for yt-dlp auto captions.
    """
    ts_re = re.compile(r"(?P<s>\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<e>\d{2}:\d{2}:\d{2}\.\d{3})")
    def _to_seconds(hhmmss_ms: str) -> float:
        hh, mm, ss_ms = hhmmss_ms.split(":")
        ss, ms = ss_ms.split(".")
        return int(hh)*3600 + int(mm)*60 + int(ss) + int(ms)/1000.0

    segs: List[Dict[str, Any]] = []
    buf: List[str] = []
    start = end = None

    for raw in io.StringIO(vtt_text):
        line = raw.strip("\n")
        m = ts_re.search(line)
        if m:
            # flush previous cue
            if start is not None and buf:
                text = " ".join(b for b in buf if b).strip()
                if text:
                    s = _to_seconds(start); e = _to_seconds(end)
                    segs.append({"start": s, "duration": max(0.0, e - s), "text": text})
            start, end = m.group("s"), m.group("e")
            buf = []
            continue

        if not line.strip():
            if start is not None and buf:
                text = " ".join(b for b in buf if b).strip()
                if text:
                    s = _to_seconds(start); e = _to_seconds(end)
                    segs.append({"start": s, "duration": max(0.0, e - s), "text": text})
            start = end = None
            buf = []
            continue

        if start is not None:
            buf.append(line)

    if start is not None and buf:
        text = " ".join(b for b in buf if b).strip()
        if text:
            s = _to_seconds(start); e = _to_seconds(end)
            segs.append({"start": s, "duration": max(0.0, e - s), "text": text})

    return segs

def _clean_plain_blocks(blocks: List[str]) -> str:
    out, cur, chars = [], [], 0
    for word in " ".join(blocks).split():
        cur.append(word)
        chars += len(word) + 1
        if chars > 400 and word[-1:] in ".!?":
            out.append(" ".join(cur)); cur, chars = [], 0
    if cur: out.append(" ".join(cur))
    return "\n\n".join(out)

# -----------------------
# Transcript (via yt-dlp)
# -----------------------
def get_transcript_with_ytdlp(video_id_or_url: str, clean: bool = True, fmt: Optional[str] = None) -> Optional[str]:
    """
    Extract subtitles using yt-dlp (auto captions allowed) and return:
      - SRT or VTT content if fmt is 'srt' / 'vtt'
      - Otherwise, produce Clean TXT (paragraphs) or Timestamped TXT from VTT.
    Works for IDs or any YT URL; uses auto-captions when authored captions are absent.
    """
    want_fmt = "srt" if fmt == "srt" else "vtt"
    lang_priority = ['en', 'en-US', 'en-GB', 'en-CA']

    with tempfile.TemporaryDirectory() as tmp:
        ydl_opts: Dict[str, Any] = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,            # allow auto-captions
            "subtitlesformat": want_fmt,
            "subtitleslangs": lang_priority,
            "outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }
        _apply_cookie_opts(ydl_opts)

        url = _norm_youtube_url(video_id_or_url)
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            logger.debug(f"yt-dlp subtitles download failed for {video_id_or_url}: {e}")
            return None

        # locate produced subtitle file
        # Try by id first, then any.{fmt}
        sub_files = list(Path(tmp).glob(f"*.{want_fmt}"))
        if not sub_files:
            return None

        content = sub_files[0].read_text(encoding="utf-8", errors="ignore")

        if fmt in ("srt", "vtt"):
            return content

        # Convert VTT to TXT styles
        segments = _parse_vtt_to_segments(content)
        if not segments:
            return None

        if clean:
            texts = [(s.get("text") or "").replace("\n", " ").strip() for s in segments]
            return _clean_plain_blocks(texts)

        lines: List[str] = []
        for s in segments:
            t = int(float(s.get("start", 0)))
            txt = (s.get("text") or "").replace("\n", " ").strip()
            if not txt:
                continue
            lines.append(f"[{t//60:02d}:{t%60:02d}] {txt}")
        return "\n".join(lines)

# -----------------------
# Video / Audio
# -----------------------
def _common_ydl_opts(output_dir: str) -> Dict[str, Any]:
    ffmpeg_loc = _ensure_ffmpeg_location()
    opts: Dict[str, Any] = {
        "format": ("bv*[ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
                   "bv*+ba/best"),
        "merge_output_format": "mp4",
        "postprocessors": [
            {"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"},
            {"key": "FFmpegMetadata"},
            {"key": "EmbedThumbnail", "already_have_thumbnail": False},
        ],
        "writethumbnail": True,
        "outtmpl": _safe_outtmpl(output_dir),
        "noprogress": True,
        "quiet": True,
        "concurrent_fragment_downloads": 4,
        "retries": 5,
        "fragment_retries": 5,
    }
    if ffmpeg_loc:
        opts["ffmpeg_location"] = ffmpeg_loc
    _apply_cookie_opts(opts)
    return opts

def get_video_info(video_id_or_url: str) -> Dict[str, Any]:
    ydl_opts: Dict[str, Any] = {"quiet": True, "no_warnings": True, "skip_download": True}
    _apply_cookie_opts(ydl_opts)
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(_norm_youtube_url(video_id_or_url), download=False)
        return {
            "id": info.get("id"),
            "title": info.get("title"),
            "uploader": info.get("uploader") or info.get("channel"),
            "duration": info.get("duration"),
        }

def download_audio_with_ytdlp(video_id_or_url: str, quality: str, output_dir: str) -> str:
    kbps = "96" if (quality or "").lower() in {"low", "l"} else "256" if (quality or "").lower() in {"high", "h"} else "160"
    opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": kbps},
            {"key": "FFmpegMetadata"},
            {"key": "EmbedThumbnail", "already_have_thumbnail": False},
        ],
        "outtmpl": _safe_outtmpl(output_dir),
        "noprogress": True,
        "quiet": True,
        "writethumbnail": True,
    }
    ffmpeg_loc = _ensure_ffmpeg_location()
    if ffmpeg_loc:
        opts["ffmpeg_location"] = ffmpeg_loc
    _apply_cookie_opts(opts)

    os.makedirs(output_dir, exist_ok=True)
    url = _norm_youtube_url(video_id_or_url)
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        base = ydl.prepare_filename(info)
        return base.rsplit(".", 1)[0] + ".mp3"

def download_video_with_ytdlp(video_id_or_url: str, quality: str, output_dir: str) -> str:
    q = re.sub(r"[^0-9]", "", quality or "")
    height = int(q) if q.isdigit() else None
    opts = _common_ydl_opts(output_dir)
    if height:
        opts["format"] = (f"bv*[height={height}][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
                          f"bv*[height<={height}][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
                          "bv*+ba/best")
    os.makedirs(output_dir, exist_ok=True)
    url = _norm_youtube_url(video_id_or_url)
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        base, _ = os.path.splitext(ydl.prepare_filename(info))
        mp4 = base + ".mp4"
        return mp4 if os.path.exists(mp4) else ydl.prepare_filename(info)


