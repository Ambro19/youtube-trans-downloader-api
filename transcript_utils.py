# transcript_utils.py — PRODUCTION-READY (proxy + cookies + hardened yt-dlp)
"""
YouTube Content Downloader (YCD) — yt-dlp utilities

Key points:
- Reads proxy + cookies env vars at *runtime* (not import time).
- Supports PROXY_ENABLED + PROXY_HOST/PORT/USERNAME/PASSWORD.
- Uses cookie file created by main.py hydration OR decodes YT_COOKIES_B64 into /tmp/yt-dlp/cookies.txt.
- Avoids logging secrets.
- Does not overwrite extractor_args accidentally (merges instead).
- Adds sane retries/timeouts + lightweight “anti-bot hygiene” knobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List
import logging
import os
import re
import tempfile
import io
import base64
import random
import time

from yt_dlp import YoutubeDL  # pyright: ignore[reportMissingModuleSource]

logger = logging.getLogger("youtube_trans_downloader")

# --------------------------------------------------------------------------------------
# Env helpers
# --------------------------------------------------------------------------------------

def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.getenv(name, "") or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}

def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()

def _ytdlp_dir() -> Path:
    # Always writable on Render/containers
    p = Path(_env_str("YT_DLP_DIR", "/tmp/yt-dlp"))
    p.mkdir(parents=True, exist_ok=True)
    return p

# --------------------------------------------------------------------------------------
# Cookies: runtime resolution (works with main.py hydration)
# --------------------------------------------------------------------------------------

_COOKIES_CACHE: Optional[str] = None

def _get_cookies_file() -> Optional[str]:
    """
    Returns a readable cookies file path for yt-dlp, or None.

    Priority:
      1) If main.py already hydrated YT_COOKIES_FILE (points to /tmp/yt-dlp/cookies.txt), use it.
      2) Else, decode YT_COOKIES_B64 -> /tmp/yt-dlp/cookies.txt
      3) Else, use YT_COOKIES_FILE as-is (copy if read-only mount).
    """
    global _COOKIES_CACHE

    # If cached & still valid
    if _COOKIES_CACHE and os.path.exists(_COOKIES_CACHE):
        return _COOKIES_CACHE
    _COOKIES_CACHE = None

    # 1) Prefer already-set YT_COOKIES_FILE (main.py hydration)
    cookies_file_env = _env_str("YT_COOKIES_FILE", "")
    if cookies_file_env and os.path.exists(cookies_file_env) and os.access(cookies_file_env, os.R_OK):
        _COOKIES_CACHE = cookies_file_env
        return _COOKIES_CACHE

    # 2) Decode base64 -> /tmp/yt-dlp/cookies.txt
    cookies_b64 = _env_str("YT_COOKIES_B64", "")
    if cookies_b64:
        try:
            target_dir = _ytdlp_dir()
            target = target_dir / "cookies.txt"

            raw = base64.b64decode(cookies_b64)
            raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

            target.write_bytes(raw)

            if target.exists() and target.stat().st_size > 10:
                _COOKIES_CACHE = str(target)
                # Do NOT log cookie contents; only size.
                logger.info("✅ Cookies hydrated from YT_COOKIES_B64 to %s (%d bytes)", target, target.stat().st_size)
                # Also set env so other modules see it
                os.environ["YT_COOKIES_FILE"] = _COOKIES_CACHE
                return _COOKIES_CACHE
            logger.warning("⚠️ Decoded cookies file is empty (%s)", target)
        except Exception as e:
            logger.warning("⚠️ Failed decoding YT_COOKIES_B64: %s", e)

    # 3) Fallback: copy read-only cookie file into /tmp
    if cookies_file_env and os.path.exists(cookies_file_env) and os.access(cookies_file_env, os.R_OK):
        try:
            if cookies_file_env.startswith(("/etc/", "/run/")):
                import shutil
                target_dir = _ytdlp_dir()
                target = target_dir / "cookies.txt"
                shutil.copyfile(cookies_file_env, target)
                _COOKIES_CACHE = str(target)
                os.environ["YT_COOKIES_FILE"] = _COOKIES_CACHE
                logger.info("✅ Cookies copied into writable path: %s", target)
                return _COOKIES_CACHE
            _COOKIES_CACHE = cookies_file_env
            return _COOKIES_CACHE
        except Exception as e:
            logger.warning("⚠️ Could not copy YT_COOKIES_FILE to /tmp: %s", e)

    return None

# --------------------------------------------------------------------------------------
# Proxy: runtime resolution (NO secrets logged)
# --------------------------------------------------------------------------------------

def _proxy_url() -> Optional[str]:
    """
    Build proxy URL from env:
      PROXY_ENABLED=true
      PROXY_HOST=...
      PROXY_PORT=...
      PROXY_USERNAME=...
      PROXY_PASSWORD=...
    """
    if not _env_bool("PROXY_ENABLED", False):
        return None

    host = _env_str("PROXY_HOST")
    port = _env_str("PROXY_PORT")
    user = _env_str("PROXY_USERNAME")
    pwd  = _env_str("PROXY_PASSWORD")

    if not all([host, port, user, pwd]):
        logger.warning("⚠️ PROXY_ENABLED=true but proxy env vars are incomplete.")
        return None

    # Most residential providers accept http proxy URL even for https sites.
    return f"http://{user}:{pwd}@{host}:{port}"

def _log_proxy_status(proxy: Optional[str]) -> None:
    if not proxy:
        logger.info("ℹ️ Proxy disabled (or not configured).")
        return
    # Mask: show host:port only
    try:
        m = re.search(r"@([^:/]+):(\d+)", proxy)
        if m:
            logger.info("🌐 Proxy enabled -> %s:%s", m.group(1), m.group(2))
        else:
            logger.info("🌐 Proxy enabled.")
    except Exception:
        logger.info("🌐 Proxy enabled.")

# --------------------------------------------------------------------------------------
# yt-dlp option hardening
# --------------------------------------------------------------------------------------

def _merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_dict(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst

def _apply_hardening_opts(opts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies cookies + proxy + hardened headers + retries.
    """
    # Use /tmp for caches (Render-friendly)
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("YTDLP_HOME", "/tmp/yt-dlp")

    # Cookies
    cp = _get_cookies_file()
    if cp:
        opts["cookiefile"] = cp

    # Proxy
    proxy = _proxy_url()
    if proxy:
        opts["proxy"] = proxy
    _log_proxy_status(proxy)

    # IPv4 bind (yt-dlp expects `source_address`, not `force_ipv4`)
    if _env_bool("YTDLP_BIND_IPV4", True):
        opts["source_address"] = "0.0.0.0"

    # Sane timeouts/retries
    opts.setdefault("socket_timeout", int(_env_str("YTDLP_SOCKET_TIMEOUT", "30")))
    opts.setdefault("retries", int(_env_str("YTDLP_RETRIES", "8")))
    opts.setdefault("fragment_retries", int(_env_str("YTDLP_FRAGMENT_RETRIES", "8")))
    opts.setdefault("extractor_retries", int(_env_str("YTDLP_EXTRACTOR_RETRIES", "3")))
    opts.setdefault("file_access_retries", int(_env_str("YTDLP_FILE_ACCESS_RETRIES", "3")))

    # Quiet by default; let your app logs speak
    opts.setdefault("quiet", _env_bool("YTDLP_QUIET", True))
    opts.setdefault("no_warnings", False)

    # Headers (helps baseline fingerprint consistency)
    headers = opts.setdefault("http_headers", {})
    headers.update({
        "User-Agent": _env_str(
            "YTDLP_UA",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    })

    # extractor args (merge, don’t overwrite)
    extractor_args = opts.setdefault("extractor_args", {})
    yt_args = extractor_args.setdefault("youtube", {})

    # Prefer web + android mix (proxy tends to work better with web first)
    if proxy:
        yt_args.setdefault("player_client", ["web", "android"])
    else:
        yt_args.setdefault("player_client", ["android", "web"])

    # Optional PO token / visitor data (if you ever enable them)
    po_token_android = _env_str("YT_PO_TOKEN_ANDROID")
    po_token_web = _env_str("YT_PO_TOKEN_WEB")
    visitor_data = _env_str("YT_VISITOR_DATA")

    # Keep semantics: yt-dlp expects specific shapes; we store without logging secrets.
    if po_token_android:
        yt_args["po_token"] = f"android.{po_token_android}"
    elif po_token_web:
        yt_args["po_token"] = f"web.{po_token_web}"

    if visitor_data:
        yt_args["visitor_data"] = visitor_data

    # Reduce extra surfaces
    yt_args.setdefault("skip", ["dash", "hls"])

    return opts

def _jitter_sleep(min_s: float = 0.25, max_s: float = 0.9) -> None:
    time.sleep(random.uniform(min_s, max_s))

# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------

_YT_ID_RE = re.compile(r"(?<![\w-])([A-Za-z0-9_-]{11})(?![\w-])")

def _extract_video_id(text: str) -> str:
    t = (text or "").strip()
    pats = [
        r"(?:youtube\.com/watch\?[^#\s]*[?&]v=)([^&\n?#]{11})",
        r"(?:youtu\.be/)([^&\n?#/]{11})",
        r"(?:youtube\.com/embed/)([^&\n?#/]{11})",
        r"(?:youtube\.com/shorts/)([^&\n?#/]{11})",
    ]
    for p in pats:
        m = re.search(p, t)
        if m and _YT_ID_RE.fullmatch(m.group(1)):
            return m.group(1)
    m = _YT_ID_RE.search(t)
    return m.group(1) if m else ""

def _norm_youtube_url(video_id_or_url: str) -> str:
    vid = _extract_video_id(video_id_or_url) or (video_id_or_url or "").strip()
    return f"https://www.youtube.com/watch?v={vid}"

def _ensure_ffmpeg_location() -> Optional[str]:
    ff = _env_str("FFMPEG_PATH")
    return ff if ff and Path(ff).exists() else None

def _safe_outtmpl(output_dir: str, stem: str = "%(title).200B [%(id)s]", ext_placeholder: str = "%(ext)s") -> Dict[str, str]:
    return {"default": os.path.join(output_dir, f"{stem}.{ext_placeholder}")}

def check_ytdlp_availability() -> bool:
    try:
        import yt_dlp  # pyright: ignore[reportMissingModuleSource] # noqa: F401
        return True
    except Exception as e:
        logger.debug("yt-dlp not available: %s", e)
        return False

# --------------------------------------------------------------------------------------
# Transcript parsing
# --------------------------------------------------------------------------------------

def _parse_vtt_to_segments(vtt_text: str) -> List[Dict[str, Any]]:
    ts_re = re.compile(
        r"(?P<s>\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<e>\d{2}:\d{2}:\d{2}\.\d{3})"
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
                    segs.append({"start": s_val, "duration": max(0.0, e_val - s_val), "text": text})
            start, end = m.group("s"), m.group("e")
            buf = []
            continue

        if not line.strip():
            if start is not None and buf:
                text = " ".join(b for b in buf if b).strip()
                if text:
                    s_val = _to_seconds(start)
                    e_val = _to_seconds(end)
                    segs.append({"start": s_val, "duration": max(0.0, e_val - s_val), "text": text})
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
            segs.append({"start": s_val, "duration": max(0.0, e_val - s_val), "text": text})

    return segs

def _clean_plain_blocks(blocks: List[str]) -> str:
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

# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def get_transcript_with_ytdlp(video_id_or_url: str, clean: bool = True, fmt: Optional[str] = None) -> Optional[str]:
    """
    Extract subtitles using yt-dlp (with cookies + optional proxy).

    fmt:
      - "srt" returns SRT
      - "vtt" returns VTT
      - None returns either clean text or timestamped text depending on `clean`
    """
    vid = _extract_video_id(video_id_or_url)
    if not vid:
        return None

    want_fmt = "srt" if fmt == "srt" else "vtt"
    lang_priority = ["en", "en-US", "en-GB", "en-CA", "en-AU"]

    # small jitter helps avoid looking like a tight loop
    _jitter_sleep()

    with tempfile.TemporaryDirectory(prefix="ytdlp_subs_") as tmp:
        ydl_opts: Dict[str, Any] = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": want_fmt,
            "subtitleslangs": lang_priority,
            "outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"),
            "ignoreerrors": False,
        }
        _apply_hardening_opts(ydl_opts)

        url = _norm_youtube_url(vid)

        try:
            with YoutubeDL(ydl_opts) as ydl:
                # download() is correct when writing subtitle files
                ydl.download([url])
        except Exception as e:
            msg = str(e).lower()
            logger.warning("yt-dlp transcript fetch failed for %s: %s", vid, str(e)[:250])

            # Bot / auth / block indicators
            if any(x in msg for x in ["sign in to confirm", "not a robot", "captcha", "bot", "429", "too many requests"]):
                return None
            # No subs
            if any(x in msg for x in ["no subtitles", "subtitles are disabled", "no suitable formats"]):
                return None
            return None

        sub_files = list(Path(tmp).glob(f"*.{want_fmt}"))
        if not sub_files:
            return None

        content = sub_files[0].read_text(encoding="utf-8", errors="ignore")

        if fmt in ("srt", "vtt"):
            return content

        # If VTT, parse to segments for clean/timestamped rendering
        segments = _parse_vtt_to_segments(content) if want_fmt == "vtt" else []
        if not segments:
            # If we can’t parse, still return raw content rather than nothing
            return content if not clean else content.strip()

        if clean:
            texts = [(s.get("text") or "").replace("\n", " ").strip() for s in segments]
            return _clean_plain_blocks(texts)

        lines: List[str] = []
        for s in segments:
            t = int(float(s.get("start", 0)))
            txt = (s.get("text") or "").replace("\n", " ").strip()
            if txt:
                lines.append(f"[{t // 60:02d}:{t % 60:02d}] {txt}")
        return "\n".join(lines)

def get_video_info(video_id_or_url: str) -> Dict[str, Any]:
    ydl_opts: Dict[str, Any] = {"quiet": True, "no_warnings": True, "skip_download": True}
    _apply_hardening_opts(ydl_opts)

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(_norm_youtube_url(video_id_or_url), download=False)
            return {
                "id": info.get("id"),
                "title": info.get("title"),
                "uploader": info.get("uploader") or info.get("channel"),
                "duration": info.get("duration"),
            }
    except Exception as e:
        logger.warning("get_video_info failed: %s", str(e)[:250])
        return {}

def download_audio_with_ytdlp(video_id_or_url: str, quality: str, output_dir: str) -> str:
    q = (quality or "").lower()
    kbps = "96" if q in {"low", "l"} else "256" if q in {"high", "h"} else "160"

    opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": kbps},
            {"key": "FFmpegMetadata"},
            {"key": "EmbedThumbnail", "already_have_thumbnail": False},
        ],
        "outtmpl": _safe_outtmpl(output_dir),
        "noprogress": True,
        "writethumbnail": True,
        "ignoreerrors": False,
    }

    ffmpeg_loc = _ensure_ffmpeg_location()
    if ffmpeg_loc:
        opts["ffmpeg_location"] = ffmpeg_loc

    _apply_hardening_opts(opts)
    os.makedirs(output_dir, exist_ok=True)

    _jitter_sleep()

    url = _norm_youtube_url(video_id_or_url)
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        base = ydl.prepare_filename(info)
        mp3_path = base.rsplit(".", 1)[0] + ".mp3"
        return mp3_path

def download_video_with_ytdlp(video_id_or_url: str, quality: str, output_dir: str) -> str:
    q = re.sub(r"[^0-9]", "", quality or "")
    height = int(q) if q.isdigit() else None

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
        "concurrent_fragment_downloads": int(_env_str("YTDLP_CONCURRENT_FRAGMENTS", "3")),
        "ignoreerrors": False,
    }

    if height:
        opts["format"] = (
            f"bv*[height={height}][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
            f"bv*[height<={height}][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
            "bv*+ba/best"
        )

    ffmpeg_loc = _ensure_ffmpeg_location()
    if ffmpeg_loc:
        opts["ffmpeg_location"] = ffmpeg_loc

    _apply_hardening_opts(opts)
    os.makedirs(output_dir, exist_ok=True)

    _jitter_sleep()

    url = _norm_youtube_url(video_id_or_url)
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        base, _ = os.path.splitext(ydl.prepare_filename(info))
        mp4 = base + ".mp4"
        return mp4 if os.path.exists(mp4) else ydl.prepare_filename(info)
