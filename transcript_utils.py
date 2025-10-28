# transcript_utils.py — PRODUCTION-READY with enhanced error handling
# Public API kept stable for main.py:
#   - get_transcript_with_ytdlp(video_id, clean=True, retries=3, wait_sec=1)
#   - download_audio_with_ytdlp(video_id, quality="medium", output_dir=None) -> str
#   - download_video_with_ytdlp(video_id, quality="720p", output_dir=None) -> Optional[str]
#   - get_video_info(video_id) -> Optional[Dict[str, Any]]
#   - check_ytdlp_availability() -> bool

import os
import json
import subprocess
import time
import logging
import tempfile
import re
import shutil
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import random
import sys


logger = logging.getLogger("transcript_utils")

# ==============
# CONFIGURATION
# ==============

# Default download destination (main.py passes an explicit output_dir, but keep a sane default)
DEFAULT_DOWNLOADS_DIR = Path.home() / "Downloads"

# Server files directory for storing decoded cookies
SERVER_FILES_DIR = Path(os.getenv("SERVER_FILES_DIR", "/opt/render/project/src/server_files"))
SERVER_FILES_DIR.mkdir(parents=True, exist_ok=True)

# Cookies configuration
_COOKIES_FILE_CACHE: str | None = None
YTDLP_COOKIES_PATH = os.getenv("YTDLP_COOKIES_PATH", "").strip()
YTDLP_COOKIES_B64 = os.getenv("YTDLP_COOKIES_B64", "").strip()

# Control whether yt-dlp preserves remote mtimes (we want it OFF by default)
YTDLP_NO_MTIME = os.getenv("YTDLP_NO_MTIME", "true").lower() in {"1", "true", "yes", "on"}

# User agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

# build commands routing to either `yt-dlp` or `python -m yt_dlp`
_YTDLP_BASE_CMD: list[str] | None = None


# ========
# Helpers
# ========

def _ua() -> str:
    # Stable, modern UA helps avoid extra challenges
    return (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

_COOKIES_CACHE_PATH = Path("/tmp/yt_cookies.txt")

def _maybe_add_cookies(cmd: list[str]) -> list[str]:
    """
    Prefer YTDLP_COOKIES_FILE. Fallback to YTDLP_COOKIES_B64.
    Returns a new command list with '--cookies <path>' appended if available.
    """
    import os, base64

    file_path = os.getenv("YTDLP_COOKIES_FILE")
    if file_path and Path(file_path).exists():
        return [*cmd, "--cookies", file_path]

    b64 = os.getenv("YTDLP_COOKIES_B64")
    if b64:
        try:
            data = base64.b64decode(b64)
            _COOKIES_CACHE_PATH.write_bytes(data)
            try: _COOKIES_CACHE_PATH.chmod(0o600)
            except Exception: pass
            return [*cmd, "--cookies", str(_COOKIES_CACHE_PATH)]
        except Exception:
            # If bad/too large, just skip using cookies and let caller’s retries handle it.
            pass

    return cmd


def _maybe_no_mtime(cmd: List[str]) -> List[str]:
    """Ensure yt-dlp does NOT stamp remote mtime (prevents Windows 'Yesterday')."""
    if YTDLP_NO_MTIME and "--no-mtime" not in cmd:
        cmd.append("--no-mtime")
    return cmd

def _ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

def _touch_now(p: Path) -> None:
    """Force mtime=now so Windows shows file under 'Today'."""
    try:
        now = time.time()
        os.utime(p, (now, now))
    except Exception as e:
        logger.debug(f"touch-now failed for {p}: {e}")

def _parse_height(quality: str, default: int = 720) -> int:
    if not quality:
        return default
    q = quality.strip().lower()
    if q == "best":
        return 4320
    if q == "worst":
        return 240
    m = re.search(r"(\d+)", q)
    try:
        return int(m.group(1)) if m else default
    except Exception:
        return default

def _resolve_ytdlp_base_cmd() -> list[str] | None:
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    # common paths
    for path in [
        "/usr/local/bin/yt-dlp",
        "/usr/bin/yt-dlp",
        str(Path.home() / ".local" / "bin" / "yt-dlp"),
        str(Path(sys.executable).parent / "yt-dlp"),
    ]:
        if Path(path).exists():
            return [path]
    # fallback to module runner if package is installed but script is not on PATH
    try:
        import yt_dlp  # noqa: F401
        return [sys.executable, "-m", "yt_dlp"]
    except Exception:
        return None

def _ytdlp_cmd(args: list[str]) -> list[str] | None:
    global _YTDLP_BASE_CMD
    if _YTDLP_BASE_CMD is None:
        _YTDLP_BASE_CMD = _resolve_ytdlp_base_cmd()
        if _YTDLP_BASE_CMD:
            logger.info("yt-dlp command resolved to: %s", " ".join(_YTDLP_BASE_CMD))
        else:
            logger.error("yt-dlp not found (neither binary nor module).")
    return (_YTDLP_BASE_CMD + args) if _YTDLP_BASE_CMD else None

# ============
# TRANSCRIPTS
# ============

def get_transcript_with_ytdlp(video_id: str, clean: bool = True, retries: int = 3, wait_sec: int = 1) -> Optional[str]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    base = _ytdlp_cmd([])
    if not base:
        logger.error("yt-dlp is unavailable; cannot fetch subtitles")
        return None
    try:
        with tempfile.TemporaryDirectory(prefix=f"yt_trans_{video_id}_") as tmp:
            tmpdir = Path(tmp)
            strategies = [
                {"sub_langs": "en,en-US,en-GB,en-CA,en-AU", "desc": "All English variants"},
                {"sub_langs": "en", "desc": "Simple English"},
                {"sub_langs": "all", "desc": "Any language"},
            ]
            for i, strat in enumerate(strategies, 1):
                logger.info(f"[transcript] Strategy {i}/{len(strategies)}: {strat['desc']}")
                cmd = _ytdlp_cmd([
                    "--restrict-filenames",
                    "--skip-download",
                    "--write-sub",
                    "--write-auto-sub",
                    "--sub-langs", strat["sub_langs"],
                    "--sub-format", "json3/vtt/srt",
                    "--output", "%(id)s",
                    "--user-agent", _ua(),
                    "--no-warnings",
                    url,
                ])
                if not cmd:  # unlikely after base resolved
                    return None
                cmd = _maybe_add_cookies(cmd)
                cmd = _maybe_no_mtime(cmd)
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=45, check=False, cwd=tmpdir)
                    if result.returncode != 0 and result.stderr:
                        logger.debug(f"yt-dlp(subs) stderr: {result.stderr[:500]}")
                    files = list(tmpdir.iterdir())
                    if not files:
                        logger.warning(f"No subtitle files produced (strategy {i}).")
                        continue
                    out = _process_subtitle_files(tmpdir, video_id, clean)
                    if out:
                        logger.info(f"[transcript] SUCCESS with strategy {i}")
                        return out
                except subprocess.TimeoutExpired:
                    logger.warning(f"yt-dlp timeout on strategy {i}")
                    continue
                except Exception as e:
                    logger.warning(f"Strategy {i} error: {e}")
                    continue
            logger.error(f"All subtitle strategies failed for {video_id}")
            return None
    except Exception as e:
        logger.error(f"yt-dlp transcript failure for {video_id}: {e}")
        return None

def _process_subtitle_files(tmpdir: Path, video_id: str, clean: bool) -> Optional[str]:
    """
    Process downloaded subtitle files in order of preference:
    1. JSON3 (most detailed)
    2. VTT (good fallback)
    3. SRT (last resort)
    """
    json3_files = list(tmpdir.glob(f"{video_id}*.json3")) or list(tmpdir.glob("*.json3"))
    vtt_files = list(tmpdir.glob(f"{video_id}*.vtt")) or list(tmpdir.glob("*.vtt"))
    srt_files = list(tmpdir.glob(f"{video_id}*.srt")) or list(tmpdir.glob("*.srt"))

    for f in json3_files + vtt_files + srt_files:
        logger.debug(f"Processing subtitle file: {f.name}")
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            if f.suffix == ".json3":
                return _parse_json3(content, clean)
            elif f.suffix == ".vtt":
                return _parse_vtt(content, clean)
            elif f.suffix == ".srt":
                return _parse_srt(content, clean)
        except Exception as e:
            logger.warning(f"Error parsing {f.name}: {e}")
            continue
    
    logger.warning("No valid subtitle files could be processed")
    return None

def _parse_json3(content: str, clean: bool) -> Optional[str]:
    """Parse JSON3 format (YouTube's native format with word-level timing)."""
    try:
        data = json.loads(content)
        events = data.get("events", [])
        
        if clean:
            lines = []
            for event in events:
                segs = event.get("segs", [])
                text = "".join(s.get("utf8", "") for s in segs).strip()
                if text:
                    lines.append(text)
            return " ".join(lines)
        else:
            lines = []
            for event in events:
                start_ms = event.get("tStartMs", 0)
                segs = event.get("segs", [])
                text = "".join(s.get("utf8", "") for s in segs).strip()
                if text:
                    timestamp = f"{start_ms // 60000:02d}:{(start_ms // 1000) % 60:02d}"
                    lines.append(f"[{timestamp}] {text}")
            return "\n".join(lines)
    except (json.JSONDecodeError, Exception) as e:
        logger.debug(f"JSON3 parse error: {e}")
        return None

def _parse_vtt(content: str, clean: bool) -> Optional[str]:
    """Parse WebVTT format."""
    lines = content.splitlines()
    output = []
    current_text = []
    current_ts = None
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        if "-->" in line:
            current_ts = line.split("-->")[0].strip()[:5]
            continue
        if line and not line.isdigit():
            text = re.sub(r"<[^>]+>", "", line)
            current_text.append(text)
    
    if clean:
        return " ".join(current_text)
    else:
        # For non-clean, we'd need to track timestamps better
        return " ".join(current_text)

def _parse_srt(content: str, clean: bool) -> Optional[str]:
    """Parse SRT format."""
    lines = content.splitlines()
    output = []
    current_text = []
    
    for line in lines:
        line = line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        text = re.sub(r"<[^>]+>", "", line)
        if text:
            current_text.append(text)
    
    return " ".join(current_text) if current_text else None

# =========
# AUDIO (MP3)
# =========
def download_audio_with_ytdlp(video_id: str, quality: str = "192k", output_dir: str | None = None) -> Optional[str]:
    base = _ytdlp_cmd([])
    if not base:
        raise Exception("yt-dlp not found (binary nor module). Install with: pip install -U yt-dlp")

    out_dir = _ensure_dir(output_dir or DEFAULT_DOWNLOADS_DIR)
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = f"{video_id}_audio.%(ext)s"

    # probe first (same flags)
    try:
        list_cmd = _ytdlp_cmd([
            "--restrict-filenames",
            "--list-formats",
            "--no-warnings",
            "--force-ipv4",
            "--geo-bypass",
            "--extractor-args", "youtube:player_client=android",
            "--user-agent", _ua(),
            url,
        ])
        list_cmd = _maybe_add_cookies(list_cmd)
        list_cmd = _maybe_no_mtime(list_cmd)
        fmts = subprocess.run(list_cmd, capture_output=True, text=True, timeout=60, check=False)
        if fmts.returncode != 0:
            raise Exception(f"Cannot access formats: {fmts.stderr.strip() or 'unknown error'}")
        if not has_actual_video_formats(fmts.stdout):
            raise Exception("Only storyboard formats found (likely restricted).")
    except subprocess.TimeoutExpired:
        raise Exception("Format listing timed out")

    cmd = _ytdlp_cmd([
        "--restrict-filenames",
        "--no-playlist",
        "--output", output_template,
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", quality,   # e.g., "192k"
        "--retries", "3",
        "--fragment-retries", "3",
        "--force-ipv4", "--geo-bypass",
        "--extractor-args", "youtube:player_client=android",
        "--user-agent", _ua(),
        "--no-warnings",
        url,
    ])
    cmd = _maybe_add_cookies(cmd)
    cmd = _maybe_no_mtime(cmd)

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=out_dir, check=False)
    if r.returncode != 0:
        err = r.stderr.strip() or r.stdout.strip() or "unknown error"
        raise Exception(err)

    # locate mp3
    candidates = list(out_dir.glob(f"{video_id}_audio.*"))
    if not candidates:
        candidates = list(out_dir.glob(f"{video_id}*.*"))
    if not candidates:
        raise Exception("Audio file not found after download.")
    found = max([f for f in candidates if f.is_file()], key=lambda f: f.stat().st_size)
    if found.stat().st_size < 30_000:
        try: found.unlink()
        except Exception: pass
        raise Exception("Downloaded audio appears corrupted/too small")

    _touch_now(found)
    return str(found.absolute())

# ========
# VIDEO MP4
# ========
def download_video_with_ytdlp(video_id: str, quality: str = "720p", output_dir: str | None = None) -> Optional[str]:
    base = _ytdlp_cmd([])
    if not base:
        raise Exception("yt-dlp not found (binary nor module). Install with: pip install -U yt-dlp")

    out_dir = _ensure_dir(output_dir or DEFAULT_DOWNLOADS_DIR)
    url = f"https://www.youtube.com/watch?v={video_id}"
    height = _parse_height(quality, 720)
    output_template = f"{video_id}_video_{quality}.%(ext)s"

    # 1) probe formats first (also trips bot checks early)
    try:
        list_cmd = _ytdlp_cmd([
            "--restrict-filenames",
            "--list-formats",
            "--no-warnings",
            "--force-ipv4",
            "--geo-bypass",
            "--extractor-args", "youtube:player_client=android",
            "--user-agent", _ua(),
            url,
        ])
        list_cmd = _maybe_add_cookies(list_cmd)
        list_cmd = _maybe_no_mtime(list_cmd)
        fmts = subprocess.run(list_cmd, capture_output=True, text=True, timeout=60, check=False)
        if fmts.returncode != 0:
            raise Exception(f"Cannot access video formats: {fmts.stderr.strip() or 'unknown error'}")
        if not has_actual_video_formats(fmts.stdout):
            raise Exception("Only storyboard formats found (likely age/region/consent restricted).")
    except subprocess.TimeoutExpired:
        raise Exception("Format listing timed out")

    strategies = [
        f"best[height<={height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={height}]",
        f"(bestvideo[height<={height}]+bestaudio/best[height<={height}])[ext=mp4]/(bestvideo[height<={height}]+bestaudio/best[height<={height}])",
        "bestvideo+bestaudio/best[ext=mp4]/best",
        "best/worst",
    ]

    last_err = None
    found: Optional[Path] = None

    for i, fmt in enumerate(strategies, 1):
        cmd = _ytdlp_cmd([
            "--restrict-filenames",
            "--no-playlist",
            "--output", output_template,
            "--format", fmt,
            "--merge-output-format", "mp4",
            "--embed-metadata",
            "--add-metadata",
            "--retries", "3",
            "--fragment-retries", "3",
            "--force-ipv4", "--geo-bypass",
            "--extractor-args", "youtube:player_client=android",
            "--user-agent", _ua(),
            "--no-warnings",
            url,
        ])
        cmd = _maybe_add_cookies(cmd)   # <— important for bot challenges
        cmd = _maybe_no_mtime(cmd)

        logger.info(f"[video] strategy {i}/{len(strategies)} -> {fmt}")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=out_dir, check=False)
            if r.returncode == 0:
                # locate resulting file
                patterns = [
                    out_dir / f"{video_id}_video_{quality}.mp4",
                    out_dir / f"{video_id}_video_{quality}.*",
                    out_dir / f"{video_id}_video.*",
                    out_dir / f"{video_id}*.*",
                ]
                for pat in patterns:
                    matches = list(out_dir.glob(pat.name))
                    if matches:
                        found = max(matches, key=lambda f: f.stat().st_size)
                        break
                if found:
                    break

            last_err = r.stderr.strip() or r.stdout.strip() or "unknown error"
            logger.debug(f"strategy {i} failed: {last_err[:400]}")
            if i == len(strategies):
                raise Exception(last_err)
        except subprocess.TimeoutExpired:
            last_err = "download timed out"
            if i == len(strategies):
                raise Exception(last_err)

    if not found or not found.exists():
        all_files = [f.name for f in out_dir.iterdir() if f.is_file()]
        raise Exception(f"Video file not found after download. Files: {all_files}")

    if found.stat().st_size < 100_000:
        try: found.unlink()
        except Exception: pass
        raise Exception("Downloaded video appears corrupted/too small")

    _touch_now(found)
    return str(found.absolute())

# =========================
# INFO / CHECKS / ESTIMATES
# =========================

def get_video_info(video_id: str) -> Optional[Dict[str, Any]]:
    base = _ytdlp_cmd([])
    if not base:
        logger.warning("yt-dlp unavailable - cannot get video info")
        return None
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = _ytdlp_cmd(["--dump-json", "--no-warnings", "--no-download", "--user-agent", _ua(), url])
    cmd = _maybe_add_cookies(cmd)
    cmd = _maybe_no_mtime(cmd)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=45, check=False)
        if r.returncode == 0 and r.stdout:
            try:
                vi = json.loads(r.stdout)
            except json.JSONDecodeError:
                last_line = r.stdout.strip().splitlines()[-1]
                vi = json.loads(last_line)
            return {
                "id": vi.get("id"),
                "title": vi.get("title"),
                "duration": vi.get("duration"),
                "upload_date": vi.get("upload_date"),
                "uploader": vi.get("uploader"),
                "uploader_id": vi.get("uploader_id"),
                "view_count": vi.get("view_count"),
                "like_count": vi.get("like_count"),
                "description": (vi.get("description") or "")[:500],
                "thumbnail": vi.get("thumbnail"),
                "has_subtitles": bool(vi.get("subtitles")),
                "has_auto_captions": bool(vi.get("automatic_captions")),
                "format_note": vi.get("format_note"),
                "ext": vi.get("ext"),
                "filesize": vi.get("filesize"),
                "fps": vi.get("fps"),
                "width": vi.get("width"),
                "height": vi.get("height"),
                "age_limit": vi.get("age_limit", 0),
                "availability": vi.get("availability"),
            }
        return None
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.error(f"get_video_info error for {video_id}: {e}")
        return None

def check_ytdlp_availability() -> bool:
    """True if yt-dlp exists (and ffmpeg if possible)."""
    base = _ytdlp_cmd([])
    if not base:
        return False
    
    try:
        r = subprocess.run(base + ["--version"], capture_output=True, text=True, timeout=10, check=False)
        if r.returncode != 0:
            return False
    except Exception:
        return False
    
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10, check=False)
    except Exception:
        pass
    return True

def has_actual_video_formats(fmt_list: str) -> bool:
    """Check if format list contains actual video formats (not just storyboards)."""
    if not fmt_list:
        return False
    lines = fmt_list.lower().splitlines()
    for line in lines:
        if "storyboard" in line:
            continue
        if any(x in line for x in ["mp4", "webm", "mkv", "flv", "avi"]) and any(x in line for x in ["video", "x"]):
            return True
    return False

def estimate_file_size(video_id: str, download_type: str, quality: str) -> Optional[int]:
    try:
        info = get_video_info(video_id)
        if not info or not info.get("duration"):
            return None
        duration = int(info["duration"])
        if download_type == "audio":
            bitrates = {"high": 320, "medium": 192, "low": 96}
            bitrate = bitrates.get(quality, 192)
            return int((duration * bitrate * 1000) // 8)
        if download_type == "video":
            size_per_min = {1080: 100 * 1024 * 1024, 720: 50 * 1024 * 1024, 480: 25 * 1024 * 1024, 360: 15 * 1024 * 1024}
            h = _parse_height(quality, 720)
            key = min(size_per_min.keys(), key=lambda k: abs(k - h))
            return int((duration * size_per_min[key]) // 60)
        return None
    except Exception as e:
        logger.error(f"estimate_file_size error: {e}")
        return None

# =============================================================================
# Misc utilities
# =============================================================================

def validate_video_id(video_id: str) -> bool:
    if not video_id or len(video_id) != 11:
        return False
    import string
    valid = set(string.ascii_letters + string.digits + "-_")
    return all(c in valid for c in video_id)

def format_transcript_clean(text: str) -> str:
    try:
        sentences = re.split(r"[.!?]+\s+", text)
        paras: List[str] = []
        buf: List[str] = []
        count = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            buf.append(s)
            count += len(s) + 1
            if count > 400:
                paras.append(". ".join(buf) + ".")
                buf, count = [], 0
        if buf:
            paras.append(". ".join(buf) + ".")
        return "\n\n".join(paras)
    except Exception:
        return text

def get_downloads_directory() -> Path:
    return DEFAULT_DOWNLOADS_DIR

def set_downloads_directory(path: str) -> bool:
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        t = p / "test_write.tmp"
        t.write_text("ok")
        t.unlink(missing_ok=True)
        global DEFAULT_DOWNLOADS_DIR
        DEFAULT_DOWNLOADS_DIR = p
        logger.info(f"Downloads directory updated to: {p}")
        return True
    except Exception as e:
        logger.error(f"set_downloads_directory error: {e}")
        return False

# ------------------ Optional local test harness ------------------

def _test_transcript(video_id: str = "dQw4w9WgXcQ"):
    clean = get_transcript_with_ytdlp(video_id, clean=True)
    ts = get_transcript_with_ytdlp(video_id, clean=False)
    logger.info(f"Transcript clean={bool(clean)} ts={bool(ts)}")
    return clean, ts

def _test_audio(video_id: str = "dQw4w9WgXcQ", quality: str = "medium"):
    with tempfile.TemporaryDirectory() as td:
        return bool(download_audio_with_ytdlp(video_id, quality, td))

def _test_video(video_id: str = "dQw4w9WgXcQ", quality: str = "720p"):
    with tempfile.TemporaryDirectory() as td:
        p = download_video_with_ytdlp(video_id, quality, td)
        return bool(p and Path(p).exists())

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print("yt-dlp available:", check_ytdlp_availability())
    print("Downloads dir:", DEFAULT_DOWNLOADS_DIR)
    try:
        _test_transcript()
        print("Transcript test: OK")
    except Exception as e:
        print("Transcript test failed:", e)
    try:
        ok = _test_audio()
        print("Audio test:", "OK" if ok else "FAIL")
    except Exception as e:
        print("Audio test failed:", e)
    try:
        ok = _test_video()
        print("Video test:", "OK" if ok else "FAIL")
    except Exception as e:
        print("Video test failed:", e)