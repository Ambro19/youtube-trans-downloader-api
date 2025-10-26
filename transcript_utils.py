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
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import random

logger = logging.getLogger("transcript_utils")

# ==============
# CONFIGURATION
# ==============

# Default download destination (main.py passes an explicit output_dir, but keep a sane default)
DEFAULT_DOWNLOADS_DIR = Path.home() / "Downloads"

# Optional cookies file for yt-dlp (improves success for restricted content)
COOKIES_PATH = os.getenv("YTDLP_COOKIES")  # e.g., "/path/to/cookies.txt"

# Control whether yt-dlp preserves remote mtimes (we want it OFF by default)
YTDLP_NO_MTIME = os.getenv("YTDLP_NO_MTIME", "true").lower() in {"1", "true", "yes", "on"}

# User agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

# ========
# Helpers
# ========

def _ua() -> str:
    return random.choice(USER_AGENTS)

def _maybe_add_cookies(cmd: List[str]) -> List[str]:
    if COOKIES_PATH and Path(COOKIES_PATH).exists():
        return cmd + ["--cookies", COOKIES_PATH]
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

def _get_ytdlp_path() -> Optional[str]:
    """Find yt-dlp executable, checking multiple locations"""
    # Try direct command first
    if shutil.which("yt-dlp"):
        return "yt-dlp"
    
    # Try common installation paths
    possible_paths = [
        "/usr/local/bin/yt-dlp",
        "/usr/bin/yt-dlp",
        str(Path.home() / ".local" / "bin" / "yt-dlp"),
        str(Path(sys.executable).parent / "yt-dlp"),
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            logger.info(f"Found yt-dlp at: {path}")
            return path
    
    logger.error("yt-dlp not found in any common location")
    return None

# ============
# TRANSCRIPTS
# ============

def get_transcript_with_ytdlp(video_id: str, clean: bool = True, retries: int = 3, wait_sec: int = 1) -> Optional[str]:
    """
    PRODUCTION-READY: Fallback transcript extractor using yt-dlp with comprehensive error handling.
    
    Returns clean text or timestamped lines depending on `clean`.
    Prefers JSON3, then VTT, then SRT (all English variants).
    
    ENHANCEMENTS:
    - Downloads both human-created (--write-sub) and auto-generated (--write-auto-sub) subtitles
    - Tries multiple English language codes (en, en-US, en-GB, en-CA, en-AU)
    - Better file pattern matching to catch all subtitle variations
    - Comprehensive error logging for production debugging
    - Fallback to any available subtitle if English not found
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Check if yt-dlp is available
    ytdlp_path = _get_ytdlp_path()
    if not ytdlp_path:
        logger.error("yt-dlp executable not found - cannot download subtitles")
        logger.error("Install with: pip install -U yt-dlp --break-system-packages")
        return None
    
    try:
        with tempfile.TemporaryDirectory(prefix=f"yt_trans_{video_id}_") as tmp:
            tmpdir = Path(tmp)

            # Try multiple subtitle strategies
            strategies = [
                # Strategy 1: Comprehensive English variants
                {
                    "sub_langs": "en,en-US,en-GB,en-CA,en-AU",
                    "desc": "All English variants"
                },
                # Strategy 2: Just English (simpler, faster)
                {
                    "sub_langs": "en",
                    "desc": "Simple English"
                },
                # Strategy 3: Any available subtitle
                {
                    "sub_langs": "all",
                    "desc": "Any language"
                }
            ]

            for strategy_num, strategy in enumerate(strategies, 1):
                logger.info(f"[transcript] Strategy {strategy_num}/{len(strategies)}: {strategy['desc']}")
                
                cmd = [
                    ytdlp_path,
                    "--restrict-filenames",
                    "--skip-download",
                    "--write-sub",                       # Human-created subtitles
                    "--write-auto-sub",                  # Auto-generated subtitles
                    "--sub-langs", strategy["sub_langs"],
                    "--sub-format", "json3/vtt/srt",     # Preferred formats
                    "--output", "%(id)s",
                    "--user-agent", _ua(),
                    "--no-warnings",
                    url,
                ]
                cmd = _maybe_add_cookies(cmd)
                cmd = _maybe_no_mtime(cmd)

                logger.info(f"[transcript] Downloading subtitles for {video_id}")
                logger.debug(f"[transcript] Command: {' '.join(cmd[:10])}...")

                try:
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=45, 
                        check=False, 
                        cwd=tmpdir
                    )
                    
                    if result.returncode != 0:
                        logger.warning(f"yt-dlp returned code {result.returncode}")
                        if result.stderr:
                            logger.debug(f"stderr: {result.stderr[:500]}")
                    
                    # List all files for debugging
                    all_files = list(tmpdir.iterdir())
                    if all_files:
                        logger.debug(f"Files in temp dir: {[f.name for f in all_files]}")
                    else:
                        logger.warning(f"No files downloaded for strategy {strategy_num}")
                        continue

                    # Try to find and process subtitle files
                    result_text = _process_subtitle_files(tmpdir, video_id, clean)
                    if result_text:
                        logger.info(f"[transcript] SUCCESS with strategy {strategy_num}")
                        return result_text
                    else:
                        logger.info(f"[transcript] Strategy {strategy_num} found files but couldn't extract text")
                        
                except subprocess.TimeoutExpired:
                    logger.warning(f"yt-dlp timeout on strategy {strategy_num}")
                    continue
                except Exception as e:
                    logger.warning(f"Strategy {strategy_num} error: {e}")
                    continue

            logger.error(f"All subtitle download strategies failed for {video_id}")
            return None

    except Exception as e:
        logger.error(f"yt-dlp transcript failure for {video_id}: {e}")
        return None

def _process_subtitle_files(tmpdir: Path, video_id: str, clean: bool) -> Optional[str]:
    """Process subtitle files with multiple fallback patterns"""
    
    def _find_subtitle_files(extension: str) -> List[Path]:
        """Find all subtitle files with given extension, trying multiple patterns"""
        patterns = [
            f"{video_id}.en*.{extension}",     # Standard: videoID.en.ext
            f"{video_id}.*.{extension}",       # Any language code
            f"*.en*.{extension}",              # Catch files with en anywhere
            f"*.{extension}",                  # Last resort: any file
        ]
        for pattern in patterns:
            matches = list(tmpdir.glob(pattern))
            if matches:
                logger.debug(f"Found {len(matches)} files matching: {pattern}")
                return sorted(matches, key=lambda f: f.stat().st_size, reverse=True)
        return []

    # Try JSON3 first (best format)
    for json3_file in _find_subtitle_files("json3"):
        try:
            logger.info(f"[transcript] Processing JSON3: {json3_file.name}")
            data = json.loads(json3_file.read_text(encoding="utf-8"))
            result = _process_json3_transcript(data, clean)
            if result:
                return result
        except Exception as e:
            logger.debug(f"JSON3 error ({json3_file.name}): {e}")
            continue

    # Try VTT
    for vtt_file in _find_subtitle_files("vtt"):
        try:
            logger.info(f"[transcript] Processing VTT: {vtt_file.name}")
            vtt_text = vtt_file.read_text(encoding="utf-8", errors="ignore")
            if vtt_text.strip():
                result = extract_vtt_text(vtt_text) if clean else format_transcript_vtt(vtt_text)
                if result:
                    return result
        except Exception as e:
            logger.debug(f"VTT error ({vtt_file.name}): {e}")
            continue

    # Try SRT
    for srt_file in _find_subtitle_files("srt"):
        try:
            logger.info(f"[transcript] Processing SRT: {srt_file.name}")
            srt_text = srt_file.read_text(encoding="utf-8", errors="ignore")
            if not srt_text.strip():
                continue
                
            result = _process_srt_text(srt_text, clean)
            if result:
                return result
        except Exception as e:
            logger.debug(f"SRT error ({srt_file.name}): {e}")
            continue

    return None

def _process_srt_text(srt_text: str, clean: bool) -> Optional[str]:
    """Process SRT subtitle text"""
    if clean:
        # strip indices + timecodes -> plain text
        lines = []
        for line in srt_text.splitlines():
            s = line.strip()
            if not s or s.isdigit() or "-->" in s:
                continue
            s = re.sub(r"<[^>]+>", "", s)      # drop tags
            s = re.sub(r"\{[^}]+\}", "", s)    # drop styling
            if s:
                lines.append(s)
        return " ".join(lines) if lines else None
    else:
        # convert SRT blocks -> "[MM:SS] text"
        out_lines = []
        cur_time = None
        cur_text = []
        for line in srt_text.splitlines():
            s = line.strip()
            if re.search(r"-->", s):
                # new block starting: flush previous
                if cur_time is not None and cur_text:
                    out_lines.append(f"[{cur_time}] " + " ".join(cur_text).strip())
                cur_text = []
                # parse start time
                m = re.search(r"(\d{2}):(\d{2}):(\d{2})[,.]\d{3}\s+-->", s)
                if m:
                    h, mnt, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    total = h * 3600 + mnt * 60 + sec
                    cur_time = f"{total//60:02d}:{total%60:02d}"
                else:
                    cur_time = "00:00"
            elif s and not s.isdigit():
                s = re.sub(r"<[^>]+>", "", s)
                s = re.sub(r"\{[^}]+\}", "", s)
                cur_text.append(s)
        if cur_time is not None and cur_text:
            out_lines.append(f"[{cur_time}] " + " ".join(cur_text).strip())
        return "\n".join(out_lines) if out_lines else None

def _process_json3_transcript(data: Dict[Any, Any], clean: bool) -> Optional[str]:
    try:
        blocks: List[str] = []
        for event in data.get("events", []):
            if "segs" in event and "tStartMs" in event:
                text_segments = [seg.get("utf8", "") for seg in event["segs"] if seg.get("utf8")]
                if not text_segments:
                    continue
                text = "".join(text_segments).strip()
                if not text:
                    continue
                if clean:
                    blocks.append(text)
                else:
                    seconds = int(event["tStartMs"] // 1000)
                    ts = f"[{seconds//60:02d}:{seconds%60:02d}]"
                    blocks.append(f"{ts} {text}")
        if not blocks:
            return None
        return (" ".join(blocks)) if clean else "\n".join(blocks)
    except Exception as e:
        logger.error(f"Error processing JSON3 transcript: {e}")
        return None

def extract_vtt_text(vtt_content: str) -> str:
    try:
        lines = vtt_content.strip().splitlines()
        out: List[str] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
                continue
            if "-->" in s:
                continue
            if s.isdigit():
                continue
            if re.match(r"^\d{2}:\d{2}:\d{2}", s):
                continue
            s = re.sub(r"<[^>]+>", "", s)      # strip tags
            s = re.sub(r"\{[^}]+\}", "", s)    # strip styling
            if s:
                out.append(s)
        return " ".join(out)
    except Exception as e:
        logger.error(f"VTT clean error: {e}")
        return ""

def format_transcript_vtt(raw_vtt: str) -> str:
    try:
        lines = raw_vtt.strip().splitlines()
        formatted: List[str] = ["WEBVTT", "Kind: captions", "Language: en", ""]
        i = 0
        while i < len(lines):
            s = lines[i].strip()
            if s.startswith(("WEBVTT", "Kind:", "Language:")):
                i += 1
                continue
            if "-->" in s:
                s = re.sub(r"(\d{2}:\d{2}:\d{2})[.,](\d{3})", r"\1.\2", s)
                formatted.append(s)
                i += 1
                text_bits = []
                while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                    t = lines[i].strip()
                    if not t.isdigit():
                        t = re.sub(r"<[^>]+>", "", t)
                        t = re.sub(r"\{[^}]+\}", "", t)
                        if t:
                            text_bits.append(t)
                    i += 1
                if text_bits:
                    formatted.append(" ".join(text_bits))
                formatted.append("")
            else:
                i += 1
        return "\n".join(formatted)
    except Exception as e:
        logger.error(f"VTT format error: {e}")
        return raw_vtt

# =================
# FORMAT DISCOVERY
# =================

def has_actual_video_formats(formats_output: str) -> bool:
    try:
        for line in formats_output.splitlines():
            s = line.strip().lower()
            if not s or ("id" in s and "ext" in s):
                continue
            if any(ext in s for ext in ["mp4", "webm", "mkv", "avi", "flv"]):
                if "storyboard" not in s and not s.startswith("sb"):
                    return True
            parts = line.split()
            if parts:
                fid = parts[0].lower()
                if (fid.isdigit() or any(p in fid for p in ["dash", "hls", "http"])) and not fid.startswith("sb"):
                    return True
        return False
    except Exception as e:
        logger.error(f"Format parse error: {e}")
        return False

# ======
# AUDIO
# ======

def download_audio_with_ytdlp(video_id: str, quality: str = "medium", output_dir: str | None = None) -> str:
    """
    Download audio as MP3. Returns absolute file path.
    """
    ytdlp_path = _get_ytdlp_path()
    if not ytdlp_path:
        raise Exception("yt-dlp not found - install with: pip install -U yt-dlp --break-system-packages")
    
    out_dir = _ensure_dir(output_dir or DEFAULT_DOWNLOADS_DIR)
    url = f"https://www.youtube.com/watch?v={video_id}"
    qmap = {
        "high":   ("0",   "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio"),
        "medium": ("2",   "bestaudio[abr<=192]/bestaudio[ext=m4a]/bestaudio"),
        "low":    ("5",   "bestaudio[abr<=96]/bestaudio[ext=m4a]/bestaudio"),
    }
    aq, fmt = qmap.get(quality, qmap["medium"])
    output_template = f"{video_id}_audio_{quality}.%(ext)s"

    cmd = [
        ytdlp_path,
        "--restrict-filenames",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", aq,
        "--format", fmt,
        "--output", output_template,
        "--no-playlist",
        "--prefer-ffmpeg",
        "--embed-metadata",
        "--add-metadata",
        "--user-agent", _ua(),
        "--no-warnings",
        url,
    ]
    cmd = _maybe_add_cookies(cmd)
    cmd = _maybe_no_mtime(cmd)

    logger.info(f"[audio] {video_id} -> {out_dir} ({quality})")
    logger.debug(" ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=out_dir, check=False)
        if result.stderr:
            logger.debug(f"yt-dlp(audio) stderr: {result.stderr.strip()}")

        # Resolve resulting file
        candidate = out_dir / f"{video_id}_audio_{quality}.mp3"
        if not candidate.exists():
            matches = (
                list(out_dir.glob(f"{video_id}_audio_{quality}.mp3"))
                or list(out_dir.glob(f"{video_id}_audio_{quality}.*"))
                or list(out_dir.glob(f"{video_id}*.mp3"))
                or list(out_dir.glob("*.mp3"))
            )
            if matches:
                candidate = max(matches, key=lambda f: (f.stat().st_mtime, f.stat().st_size))
        if not candidate.exists():
            all_files = [f.name for f in out_dir.iterdir() if f.is_file()]
            raise Exception(f"No audio file found. Files in dir: {all_files}")

        if candidate.stat().st_size < 1_000:
            try:
                candidate.unlink()
            except Exception:
                pass
            raise Exception("Downloaded audio appears corrupted/empty")

        _touch_now(candidate)
        return str(candidate.absolute())

    except subprocess.TimeoutExpired:
        raise Exception("Audio download timed out")
    except Exception as e:
        raise Exception(f"Audio download error: {e}")


# ======
# VIDEO
# ======

def download_video_with_ytdlp(video_id: str, quality: str = "720p", output_dir: str | None = None) -> Optional[str]:
    """
    Download a video file (merged MP4 when possible). Returns absolute path or raises Exception.
    """
    ytdlp_path = _get_ytdlp_path()
    if not ytdlp_path:
        raise Exception("yt-dlp not found - install with: pip install -U yt-dlp --break-system-packages")
    
    out_dir = _ensure_dir(output_dir or DEFAULT_DOWNLOADS_DIR)
    url = f"https://www.youtube.com/watch?v={video_id}"
    height = _parse_height(quality, 720)
    output_template = f"{video_id}_video_{quality}.%(ext)s"

    # Step 1: list formats
    try:
        list_cmd = [ytdlp_path, "--restrict-filenames", "--list-formats", "--no-warnings", "--user-agent", _ua(), url]
        list_cmd = _maybe_add_cookies(list_cmd)
        list_cmd = _maybe_no_mtime(list_cmd)
        fmts = subprocess.run(list_cmd, capture_output=True, text=True, timeout=60, check=False)
        if fmts.returncode != 0:
            raise Exception(f"Cannot access video formats: {fmts.stderr.strip() or 'unknown error'}")
        if not has_actual_video_formats(fmts.stdout):
            raise Exception("Only storyboard formats found (likely age/region/restricted content).")
    except subprocess.TimeoutExpired:
        raise Exception("Format listing timed out")

    # Step 2: strategies
    strategies = [
        f"best[height<={height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={height}]",
        f"(bestvideo[height<={height}]+bestaudio/best[height<={height}])[ext=mp4]/(bestvideo[height<={height}]+bestaudio/best[height<={height}])",
        "bestvideo+bestaudio/best[ext=mp4]/best",
        "best/worst",
    ]

    last_err = None
    for i, fmt in enumerate(strategies, 1):
        cmd = [
            ytdlp_path,
            "--restrict-filenames",
            "--no-playlist",
            "--output", output_template,
            "--format", fmt,
            "--merge-output-format", "mp4",
            "--embed-metadata",
            "--add-metadata",
            "--retries", "3",
            "--fragment-retries", "3",
            "--user-agent", _ua(),
            "--no-warnings",
            url,
        ]
        cmd = _maybe_add_cookies(cmd)
        cmd = _maybe_no_mtime(cmd)
        logger.info(f"[video] strategy {i}/{len(strategies)} -> {fmt}")
        logger.debug(" ".join(cmd))
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=out_dir, check=False)
            if r.returncode == 0:
                break
            last_err = r.stderr.strip() or r.stdout.strip() or "unknown error"
            logger.debug(f"strategy {i} failed: {last_err[:400]}")
            if i == len(strategies):
                raise Exception(last_err)
        except subprocess.TimeoutExpired:
            last_err = "download timed out"
            if i == len(strategies):
                raise Exception(last_err)

    # Step 3: resolve file
    patterns = [
        out_dir / f"{video_id}_video_{quality}.mp4",
        out_dir / f"{video_id}_video_{quality}.*",
        out_dir / f"{video_id}_video.*",
        out_dir / f"{video_id}*.*",
    ]
    found: Optional[Path] = None
    for pat in patterns:
        matches = list(out_dir.glob(pat.name if pat.is_file() else pat.name))
        if matches:
            found = max(matches, key=lambda f: f.stat().st_size)
            break
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
    """Return compact metadata for a video."""
    ytdlp_path = _get_ytdlp_path()
    if not ytdlp_path:
        logger.warning("yt-dlp not found - cannot get video info")
        return None
    
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [ytdlp_path, "--dump-json", "--no-warnings", "--no-download", "--user-agent", _ua(), url]
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
    ytdlp_path = _get_ytdlp_path()
    if not ytdlp_path:
        return False
    
    try:
        r = subprocess.run([ytdlp_path, "--version"], capture_output=True, text=True, timeout=10, check=False)
        if r.returncode != 0:
            return False
    except Exception:
        return False
    
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10, check=False)
    except Exception:
        pass
    return True

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
# #---------------------------------------------------------------------------------
# # transcript_utils.py — Enhanced helpers for transcripts, audio, and video
# # Public API kept stable for main.py:
# #   - get_transcript_with_ytdlp(video_id, clean=True, retries=3, wait_sec=1)
# #   - download_audio_with_ytdlp(video_id, quality="medium", output_dir=None) -> str
# #   - download_video_with_ytdlp(video_id, quality="720p", output_dir=None) -> Optional[str]
# #   - get_video_info(video_id) -> Optional[Dict[str, Any]]
# #   - check_ytdlp_availability() -> bool

# import os
# import json
# import subprocess
# import time
# import logging
# import tempfile
# import re
# from pathlib import Path
# from typing import Optional, Dict, Any, List
# from datetime import datetime
# import random

# logger = logging.getLogger("transcript_utils")

# # ==============
# # CONFIGURATION
# # ==============

# # Default download destination (main.py passes an explicit output_dir, but keep a sane default)
# DEFAULT_DOWNLOADS_DIR = Path.home() / "Downloads"

# # Optional cookies file for yt-dlp (improves success for restricted content)
# COOKIES_PATH = os.getenv("YTDLP_COOKIES")  # e.g., "/path/to/cookies.txt"

# # Control whether yt-dlp preserves remote mtimes (we want it OFF by default)
# YTDLP_NO_MTIME = os.getenv("YTDLP_NO_MTIME", "true").lower() in {"1", "true", "yes", "on"}

# # User agents to rotate
# USER_AGENTS = [
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
#     "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
#     "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
# ]

# # ========
# # Helpers
# # ========

# def _ua() -> str:
#     return random.choice(USER_AGENTS)

# def _maybe_add_cookies(cmd: List[str]) -> List[str]:
#     if COOKIES_PATH and Path(COOKIES_PATH).exists():
#         return cmd + ["--cookies", COOKIES_PATH]
#     return cmd

# def _maybe_no_mtime(cmd: List[str]) -> List[str]:
#     """Ensure yt-dlp does NOT stamp remote mtime (prevents Windows 'Yesterday')."""
#     if YTDLP_NO_MTIME and "--no-mtime" not in cmd:
#         cmd.append("--no-mtime")
#     return cmd

# def _ensure_dir(p: str | Path) -> Path:
#     path = Path(p)
#     path.mkdir(parents=True, exist_ok=True)
#     return path

# def _touch_now(p: Path) -> None:
#     """Force mtime=now so Windows shows file under 'Today'."""
#     try:
#         now = time.time()
#         os.utime(p, (now, now))
#     except Exception as e:
#         logger.debug(f"touch-now failed for {p}: {e}")

# def _parse_height(quality: str, default: int = 720) -> int:
#     if not quality:
#         return default
#     q = quality.strip().lower()
#     if q == "best":
#         return 4320
#     if q == "worst":
#         return 240
#     m = re.search(r"(\d+)", q)
#     try:
#         return int(m.group(1)) if m else default
#     except Exception:
#         return default

# # ============
# # TRANSCRIPTS
# # ============

# def get_transcript_with_ytdlp(video_id: str, clean: bool = True, retries: int = 3, wait_sec: int = 1) -> Optional[str]:
#     """
#     Fallback transcript extractor using yt-dlp. Works in a temp dir and cleans up.
#     Returns clean text or timestamped lines depending on `clean`.
#     Prefers JSON3, then VTT, then SRT (all English variants like en or en-US).
#     """
#     url = f"https://www.youtube.com/watch?v={video_id}"
#     try:
#         with tempfile.TemporaryDirectory(prefix=f"yt_trans_{video_id}_") as tmp:
#             tmpdir = Path(tmp)

#             cmd = [
#                 "yt-dlp",
#                 "--restrict-filenames",
#                 "--skip-download",
#                 "--write-auto-sub",          # auto (if present)
#                 "--write-sub",               # human (if present)
#                 "--sub-langs", "en.*,en",    # all English variants + plain 'en'
#                 "--sub-format", "json3/vtt/srt",
#                 "--output", "%(id)s",
#                 "--user-agent", _ua(),
#                 "--no-warnings",
#                 "--quiet",
#                 url,
#             ]
#             cmd = _maybe_add_cookies(cmd)
#             cmd = _maybe_no_mtime(cmd)

#             result = subprocess.run(cmd, capture_output=True, text=True, timeout=45, check=False, cwd=tmpdir)
#             if result.returncode != 0 and result.stderr:
#                 logger.debug(f"yt-dlp (subs) stderr: {result.stderr.strip()}")

#             def _find_first(pattern: str) -> Optional[Path]:
#                 matches = sorted(tmpdir.glob(pattern))
#                 return matches[0] if matches else None

#             # Try a few times for slow disk/network
#             for _ in range(max(1, retries)):
#                 # 1) JSON3 (best for robust text extraction)
#                 json3 = _find_first(f"{video_id}.en*.json3")
#                 if json3 and json3.exists():
#                     try:
#                         data = json.loads(json3.read_text(encoding="utf-8"))
#                         return _process_json3_transcript(data, clean)
#                     except Exception as e:
#                         logger.debug(f"JSON3 parse error: {e}")

#                 # 2) VTT
#                 vtt = _find_first(f"{video_id}.en*.vtt")
#                 if vtt and vtt.exists():
#                     vtt_text = vtt.read_text(encoding="utf-8", errors="ignore")
#                     return extract_vtt_text(vtt_text) if clean else format_transcript_vtt(vtt_text)

#                 # 3) SRT (fallback)
#                 srt = _find_first(f"{video_id}.en*.srt")
#                 if srt and srt.exists():
#                     srt_text = srt.read_text(encoding="utf-8", errors="ignore")
#                     if clean:
#                         # strip indices + timecodes -> plain text
#                         lines = []
#                         for line in srt_text.splitlines():
#                             s = line.strip()
#                             if not s or s.isdigit() or "-->" in s:
#                                 continue
#                             s = re.sub(r"<[^>]+>", "", s)      # drop tags
#                             s = re.sub(r"\{[^}]+\}", "", s)    # drop styling
#                             if s:
#                                 lines.append(s)
#                         return " ".join(lines)
#                     else:
#                         # convert SRT blocks -> "[MM:SS] text"
#                         out_lines = []
#                         cur_time = None
#                         cur_text = []
#                         for line in srt_text.splitlines():
#                             s = line.strip()
#                             if re.search(r"-->", s):
#                                 # new block starting: flush previous
#                                 if cur_time is not None and cur_text:
#                                     out_lines.append(f"[{cur_time}] " + " ".join(cur_text).strip())
#                                 cur_text = []
#                                 # parse start time
#                                 m = re.search(r"(\d{2}):(\d{2}):(\d{2})[,.]\d{3}\s+-->", s)
#                                 if m:
#                                     h, mnt, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
#                                     total = h * 3600 + mnt * 60 + sec
#                                     cur_time = f"{total//60:02d}:{total%60:02d}"
#                                 else:
#                                     cur_time = "00:00"
#                             elif s and not s.isdigit():
#                                 s = re.sub(r"<[^>]+>", "", s)
#                                 s = re.sub(r"\{[^}]+\}", "", s)
#                                 cur_text.append(s)
#                         if cur_time is not None and cur_text:
#                             out_lines.append(f"[{cur_time}] " + " ".join(cur_text).strip())
#                         return "\n".join(out_lines)

#                 time.sleep(max(0, wait_sec))

#             logger.error(f"No transcript files found for video: {video_id}")
#             return None

#     except subprocess.TimeoutExpired:
#         logger.error(f"yt-dlp transcript timeout for {video_id}")
#         return None
#     except Exception as e:
#         logger.error(f"yt-dlp transcript failure for {video_id}: {e}")
#         return None

# def _process_json3_transcript(data: Dict[Any, Any], clean: bool) -> Optional[str]:
#     try:
#         blocks: List[str] = []
#         for event in data.get("events", []):
#             if "segs" in event and "tStartMs" in event:
#                 text_segments = [seg.get("utf8", "") for seg in event["segs"] if seg.get("utf8")]
#                 if not text_segments:
#                     continue
#                 text = "".join(text_segments).strip()
#                 if not text:
#                     continue
#                 if clean:
#                     blocks.append(text)
#                 else:
#                     seconds = int(event["tStartMs"] // 1000)
#                     ts = f"[{seconds//60:02d}:{seconds%60:02d}]"
#                     blocks.append(f"{ts} {text}")
#         if not blocks:
#             return None
#         return (" ".join(blocks)) if clean else "\n".join(blocks)
#     except Exception as e:
#         logger.error(f"Error processing JSON3 transcript: {e}")
#         return None

# def extract_vtt_text(vtt_content: str) -> str:
#     try:
#         lines = vtt_content.strip().splitlines()
#         out: List[str] = []
#         for line in lines:
#             s = line.strip()
#             if not s:
#                 continue
#             if s.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
#                 continue
#             if "-->" in s:
#                 continue
#             if s.isdigit():
#                 continue
#             if re.match(r"^\d{2}:\d{2}:\d{2}", s):
#                 continue
#             s = re.sub(r"<[^>]+>", "", s)      # strip tags
#             s = re.sub(r"\{[^}]+\}", "", s)    # strip styling
#             if s:
#                 out.append(s)
#         return " ".join(out)
#     except Exception as e:
#         logger.error(f"VTT clean error: {e}")
#         return ""

# def format_transcript_vtt(raw_vtt: str) -> str:
#     try:
#         lines = raw_vtt.strip().splitlines()
#         formatted: List[str] = ["WEBVTT", "Kind: captions", "Language: en", ""]
#         i = 0
#         while i < len(lines):
#             s = lines[i].strip()
#             if s.startswith(("WEBVTT", "Kind:", "Language:")):
#                 i += 1
#                 continue
#             if "-->" in s:
#                 s = re.sub(r"(\d{2}:\d{2}:\d{2})[.,](\d{3})", r"\1.\2", s)
#                 formatted.append(s)
#                 i += 1
#                 text_bits = []
#                 while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
#                     t = lines[i].strip()
#                     if not t.isdigit():
#                         t = re.sub(r"<[^>]+>", "", t)
#                         t = re.sub(r"\{[^}]+\}", "", t)
#                         if t:
#                             text_bits.append(t)
#                     i += 1
#                 if text_bits:
#                     formatted.append(" ".join(text_bits))
#                 formatted.append("")
#             else:
#                 i += 1
#         return "\n".join(formatted)
#     except Exception as e:
#         logger.error(f"VTT format error: {e}")
#         return raw_vtt

# # =================
# # FORMAT DISCOVERY
# # =================

# def has_actual_video_formats(formats_output: str) -> bool:
#     try:
#         for line in formats_output.splitlines():
#             s = line.strip().lower()
#             if not s or ("id" in s and "ext" in s):
#                 continue
#             if any(ext in s for ext in ["mp4", "webm", "mkv", "avi", "flv"]):
#                 if "storyboard" not in s and not s.startswith("sb"):
#                     return True
#             parts = line.split()
#             if parts:
#                 fid = parts[0].lower()
#                 if (fid.isdigit() or any(p in fid for p in ["dash", "hls", "http"])) and not fid.startswith("sb"):
#                     return True
#         return False
#     except Exception as e:
#         logger.error(f"Format parse error: {e}")
#         return False

# # ======
# # AUDIO
# # ======

# def download_audio_with_ytdlp(video_id: str, quality: str = "medium", output_dir: str | None = None) -> str:
#     """
#     Download audio as MP3. Returns absolute file path.
#     """
#     out_dir = _ensure_dir(output_dir or DEFAULT_DOWNLOADS_DIR)
#     url = f"https://www.youtube.com/watch?v={video_id}"
#     qmap = {
#         "high":   ("0",   "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio"),
#         "medium": ("2",   "bestaudio[abr<=192]/bestaudio[ext=m4a]/bestaudio"),
#         "low":    ("5",   "bestaudio[abr<=96]/bestaudio[ext=m4a]/bestaudio"),
#     }
#     aq, fmt = qmap.get(quality, qmap["medium"])
#     output_template = f"{video_id}_audio_{quality}.%(ext)s"

#     cmd = [
#         "yt-dlp",
#         "--restrict-filenames",
#         "--extract-audio",
#         "--audio-format", "mp3",
#         "--audio-quality", aq,
#         "--format", fmt,
#         "--output", output_template,
#         "--no-playlist",
#         "--prefer-ffmpeg",
#         "--embed-metadata",
#         "--add-metadata",
#         "--user-agent", _ua(),
#         "--no-warnings",
#         url,
#     ]
#     cmd = _maybe_add_cookies(cmd)
#     cmd = _maybe_no_mtime(cmd)

#     logger.info(f"[audio] {video_id} -> {out_dir} ({quality})")
#     logger.debug(" ".join(cmd))

#     try:
#         result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=out_dir, check=False)
#         if result.stderr:
#             logger.debug(f"yt-dlp(audio) stderr: {result.stderr.strip()}")

#         # Resolve resulting file (prefer the exact template, but recover if needed)
#         candidate = out_dir / f"{video_id}_audio_{quality}.mp3"
#         if not candidate.exists():
#             matches = (
#                 list(out_dir.glob(f"{video_id}_audio_{quality}.mp3"))
#                 or list(out_dir.glob(f"{video_id}_audio_{quality}.*"))
#                 or list(out_dir.glob(f"{video_id}*.mp3"))
#                 or list(out_dir.glob("*.mp3"))
#             )
#             if matches:
#                 # pick latest & largest to be safe
#                 candidate = max(matches, key=lambda f: (f.stat().st_mtime, f.stat().st_size))
#         if not candidate.exists():
#             all_files = [f.name for f in out_dir.iterdir() if f.is_file()]
#             raise Exception(f"No audio file found. Files in dir: {all_files}")

#         if candidate.stat().st_size < 1_000:
#             try:
#                 candidate.unlink()
#             except Exception:
#                 pass
#             raise Exception("Downloaded audio appears corrupted/empty")

#         # Force mtime->now so it shows under 'Today' on Windows
#         _touch_now(candidate)

#         return str(candidate.absolute())

#     except subprocess.TimeoutExpired:
#         raise Exception("Audio download timed out")
#     except Exception as e:
#         raise Exception(f"Audio download error: {e}")


# # ======
# # VIDEO
# # ======

# def download_video_with_ytdlp(video_id: str, quality: str = "720p", output_dir: str | None = None) -> Optional[str]:
#     """
#     Download a video file (merged MP4 when possible). Returns absolute path or raises Exception.
#     """
#     out_dir = _ensure_dir(output_dir or DEFAULT_DOWNLOADS_DIR)
#     url = f"https://www.youtube.com/watch?v={video_id}"
#     height = _parse_height(quality, 720)
#     output_template = f"{video_id}_video_{quality}.%(ext)s"

#     # Step 1: list formats to detect restrictions  ✅ add --restrict-filenames here
#     try:
#         list_cmd = ["yt-dlp", "--restrict-filenames", "--list-formats", "--no-warnings", "--user-agent", _ua(), url]
#         list_cmd = _maybe_add_cookies(list_cmd)
#         list_cmd = _maybe_no_mtime(list_cmd)
#         fmts = subprocess.run(list_cmd, capture_output=True, text=True, timeout=60, check=False)
#         if fmts.returncode != 0:
#             raise Exception(f"Cannot access video formats: {fmts.stderr.strip() or 'unknown error'}")
#         if not has_actual_video_formats(fmts.stdout):
#             raise Exception("Only storyboard formats found (likely age/region/restricted content). Try another video.")
#     except subprocess.TimeoutExpired:
#         raise Exception("Format listing timed out")

#     # Step 2: strategies
#     strategies = [
#         f"best[height<={height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={height}]",
#         f"(bestvideo[height<={height}]+bestaudio/best[height<={height}])[ext=mp4]/(bestvideo[height<={height}]+bestaudio/best[height<={height}])",
#         "bestvideo+bestaudio/best[ext=mp4]/best",
#         "best/worst",
#     ]

#     last_err = None
#     for i, fmt in enumerate(strategies, 1):
#         cmd = [
#             "yt-dlp",
#             "--restrict-filenames",           # ✅ keep here too
#             "--no-playlist",
#             "--output", output_template,
#             "--format", fmt,
#             "--merge-output-format", "mp4",
#             "--embed-metadata",
#             "--add-metadata",
#             "--retries", "3",
#             "--fragment-retries", "3",
#             "--user-agent", _ua(),
#             "--no-warnings",
#             url,
#         ]
#         cmd = _maybe_add_cookies(cmd)
#         cmd = _maybe_no_mtime(cmd)
#         logger.info(f"[video] strategy {i}/{len(strategies)} -> {fmt}")
#         logger.debug(" ".join(cmd))
#         try:
#             r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=out_dir, check=False)
#             if r.returncode == 0:
#                 break
#             last_err = r.stderr.strip() or r.stdout.strip() or "unknown error"
#             logger.debug(f"strategy {i} failed: {last_err[:400]}")
#             if i == len(strategies):
#                 raise Exception(last_err)
#         except subprocess.TimeoutExpired:
#             last_err = "download timed out"
#             if i == len(strategies):
#                 raise Exception(last_err)

#     # Step 3: resolve file
#     patterns = [
#         out_dir / f"{video_id}_video_{quality}.mp4",
#         out_dir / f"{video_id}_video_{quality}.*",
#         out_dir / f"{video_id}_video.*",
#         out_dir / f"{video_id}*.*",
#     ]
#     found: Optional[Path] = None
#     for pat in patterns:
#         matches = list(out_dir.glob(pat.name if pat.is_file() else pat.name))
#         if matches:
#             found = max(matches, key=lambda f: f.stat().st_size)
#             break
#     if not found or not found.exists():
#         all_files = [f.name for f in out_dir.iterdir() if f.is_file()]
#         raise Exception(f"Video file not found after download. Files: {all_files}")

#     if found.stat().st_size < 100_000:
#         try: found.unlink()
#         except Exception: pass
#         raise Exception("Downloaded video appears corrupted/too small")

#     _touch_now(found)
#     return str(found.absolute())

# # =========================
# # INFO / CHECKS / ESTIMATES
# # =========================

# def get_video_info(video_id: str) -> Optional[Dict[str, Any]]:
#     """Return compact metadata for a video."""
#     url = f"https://www.youtube.com/watch?v={video_id}"
#     cmd = ["yt-dlp", "--dump-json", "--no-warnings", "--no-download", "--user-agent", _ua(), url]
#     cmd = _maybe_add_cookies(cmd)
#     cmd = _maybe_no_mtime(cmd)
#     try:
#         r = subprocess.run(cmd, capture_output=True, text=True, timeout=45, check=False)
#         if r.returncode == 0 and r.stdout:
#             try:
#                 vi = json.loads(r.stdout)
#             except json.JSONDecodeError:
#                 last_line = r.stdout.strip().splitlines()[-1]
#                 vi = json.loads(last_line)
#             return {
#                 "id": vi.get("id"),
#                 "title": vi.get("title"),
#                 "duration": vi.get("duration"),
#                 "upload_date": vi.get("upload_date"),
#                 "uploader": vi.get("uploader"),
#                 "uploader_id": vi.get("uploader_id"),
#                 "view_count": vi.get("view_count"),
#                 "like_count": vi.get("like_count"),
#                 "description": (vi.get("description") or "")[:500],
#                 "thumbnail": vi.get("thumbnail"),
#                 "has_subtitles": bool(vi.get("subtitles")),
#                 "has_auto_captions": bool(vi.get("automatic_captions")),
#                 "format_note": vi.get("format_note"),
#                 "ext": vi.get("ext"),
#                 "filesize": vi.get("filesize"),
#                 "fps": vi.get("fps"),
#                 "width": vi.get("width"),
#                 "height": vi.get("height"),
#                 "age_limit": vi.get("age_limit", 0),
#                 "availability": vi.get("availability"),
#             }
#         return None
#     except (subprocess.TimeoutExpired, Exception) as e:
#         logger.error(f"get_video_info error for {video_id}: {e}")
#         return None

# def check_ytdlp_availability() -> bool:
#     """True if yt-dlp exists (and ffmpeg if possible)."""
#     try:
#         r = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, timeout=10, check=False)
#         if r.returncode != 0:
#             return False
#     except Exception:
#         return False
#     try:
#         subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10, check=False)
#     except Exception:
#         pass
#     return True

# def estimate_file_size(video_id: str, download_type: str, quality: str) -> Optional[int]:
#     try:
#         info = get_video_info(video_id)
#         if not info or not info.get("duration"):
#             return None
#         duration = int(info["duration"])  # seconds
#         if download_type == "audio":
#             bitrates = {"high": 320, "medium": 192, "low": 96}
#             bitrate = bitrates.get(quality, 192)
#             return int((duration * bitrate * 1000) // 8)
#         if download_type == "video":
#             size_per_min = {1080: 100 * 1024 * 1024, 720: 50 * 1024 * 1024, 480: 25 * 1024 * 1024, 360: 15 * 1024 * 1024}
#             # nearest bucket
#             h = _parse_height(quality, 720)
#             key = min(size_per_min.keys(), key=lambda k: abs(k - h))
#             return int((duration * size_per_min[key]) // 60)
#         return None
#     except Exception as e:
#         logger.error(f"estimate_file_size error: {e}")
#         return None

# # =============================================================================
# # Misc utilities used elsewhere / tests
# # =============================================================================

# def validate_video_id(video_id: str) -> bool:
#     if not video_id or len(video_id) != 11:
#         return False
#     import string
#     valid = set(string.ascii_letters + string.digits + "-_")
#     return all(c in valid for c in video_id)

# def format_transcript_clean(text: str) -> str:
#     try:
#         sentences = re.split(r"[.!?]+\s+", text)
#         paras: List[str] = []
#         buf: List[str] = []
#         count = 0
#         for s in sentences:
#             s = s.strip()
#             if not s:
#                 continue
#             buf.append(s)
#             count += len(s) + 1
#             if count > 400:
#                 paras.append(". ".join(buf) + ".")
#                 buf, count = [], 0
#         if buf:
#             paras.append(". ".join(buf) + ".")
#         return "\n\n".join(paras)
#     except Exception:
#         return text

# def get_downloads_directory() -> Path:
#     return DEFAULT_DOWNLOADS_DIR

# def set_downloads_directory(path: str) -> bool:
#     try:
#         p = Path(path)
#         p.mkdir(parents=True, exist_ok=True)
#         t = p / "test_write.tmp"
#         t.write_text("ok")
#         t.unlink(missing_ok=True)
#         global DEFAULT_DOWNLOADS_DIR
#         DEFAULT_DOWNLOADS_DIR = p
#         logger.info(f"Downloads directory updated to: {p}")
#         return True
#     except Exception as e:
#         logger.error(f"set_downloads_directory error: {e}")
#         return False

# # ------------------ Optional local test harness ------------------

# def _test_transcript(video_id: str = "dQw4w9WgXcQ"):
#     clean = get_transcript_with_ytdlp(video_id, clean=True)
#     ts = get_transcript_with_ytdlp(video_id, clean=False)
#     logger.info(f"Transcript clean={bool(clean)} ts={bool(ts)}")
#     return clean, ts

# def _test_audio(video_id: str = "dQw4w9WgXcQ", quality: str = "medium"):
#     with tempfile.TemporaryDirectory() as td:
#         return bool(download_audio_with_ytdlp(video_id, quality, td))

# def _test_video(video_id: str = "dQw4w9WgXcQ", quality: str = "720p"):
#     with tempfile.TemporaryDirectory() as td:
#         p = download_video_with_ytdlp(video_id, quality, td)
#         return bool(p and Path(p).exists())

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     print("yt-dlp available:", check_ytdlp_availability())
#     print("Downloads dir:", DEFAULT_DOWNLOADS_DIR)
#     try:
#         _test_transcript()
#         print("Transcript test: OK")
#     except Exception as e:
#         print("Transcript test failed:", e)
#     try:
#         ok = _test_audio()
#         print("Audio test:", "OK" if ok else "FAIL")
#     except Exception as e:
#         print("Audio test failed:", e)
#     try:
#         ok = _test_video()
#         print("Video test:", "OK" if ok else "FAIL")
#     except Exception as e:
#         print("Video test failed:", e)


# # =============================================================================
# # -----My Old transcript_utiles.py that was worked good with actual main.py ---

# # ===============================================================================
# # transcript_utils.py — Enhanced helpers for transcripts, audio, and video
# # Public API kept stable for main.py:
# #   - get_transcript_with_ytdlp(video_id, clean=True, retries=3, wait_sec=1)
# #   - download_audio_with_ytdlp(video_id, quality="medium", output_dir=None) -> str
# #   - download_video_with_ytdlp(video_id, quality="720p", output_dir=None) -> Optional[str]
# #   - get_video_info(video_id) -> Optional[Dict[str, Any]]
# #   - check_ytdlp_availability() -> bool

# import os
# import json
# import subprocess
# import time
# import logging
# import tempfile
# import re
# from pathlib import Path
# from typing import Optional, Dict, Any, List
# from datetime import datetime
# import random

# logger = logging.getLogger("transcript_utils")

# # ==============
# # CONFIGURATION
# # ==============

# # Default download destination (main.py passes an explicit output_dir, but keep a sane default)
# DEFAULT_DOWNLOADS_DIR = Path.home() / "Downloads"

# # Optional cookies file for yt-dlp (improves success for restricted content)
# COOKIES_PATH = os.getenv("YTDLP_COOKIES")  # e.g., "/path/to/cookies.txt"

# # Control whether yt-dlp preserves remote mtimes (we want it OFF by default)
# YTDLP_NO_MTIME = os.getenv("YTDLP_NO_MTIME", "true").lower() in {"1", "true", "yes", "on"}

# # User agents to rotate
# USER_AGENTS = [
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
#     "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
#     "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
# ]

# # ========
# # Helpers
# # ========

# def _ua() -> str:
#     return random.choice(USER_AGENTS)

# def _maybe_add_cookies(cmd: List[str]) -> List[str]:
#     if COOKIES_PATH and Path(COOKIES_PATH).exists():
#         return cmd + ["--cookies", COOKIES_PATH]
#     return cmd

# def _maybe_no_mtime(cmd: List[str]) -> List[str]:
#     """Ensure yt-dlp does NOT stamp remote mtime (prevents Windows 'Yesterday')."""
#     if YTDLP_NO_MTIME and "--no-mtime" not in cmd:
#         cmd.append("--no-mtime")
#     return cmd

# def _ensure_dir(p: str | Path) -> Path:
#     path = Path(p)
#     path.mkdir(parents=True, exist_ok=True)
#     return path

# def _touch_now(p: Path) -> None:
#     """Force mtime=now so Windows shows file under 'Today'."""
#     try:
#         now = time.time()
#         os.utime(p, (now, now))
#     except Exception as e:
#         logger.debug(f"touch-now failed for {p}: {e}")

# def _parse_height(quality: str, default: int = 720) -> int:
#     if not quality:
#         return default
#     q = quality.strip().lower()
#     if q == "best":
#         return 4320
#     if q == "worst":
#         return 240
#     m = re.search(r"(\d+)", q)
#     try:
#         return int(m.group(1)) if m else default
#     except Exception:
#         return default

# # ============
# # TRANSCRIPTS
# # ============

# def get_transcript_with_ytdlp(video_id: str, clean: bool = True, retries: int = 3, wait_sec: int = 1) -> Optional[str]:
#     """
#     Fallback transcript extractor using yt-dlp. Works in a temp dir and cleans up.
#     Returns clean text or timestamped lines depending on `clean`.
#     Prefers JSON3, then VTT, then SRT (all English variants like en or en-US).
#     """
#     url = f"https://www.youtube.com/watch?v={video_id}"
#     try:
#         with tempfile.TemporaryDirectory(prefix=f"yt_trans_{video_id}_") as tmp:
#             tmpdir = Path(tmp)

#             cmd = [
#                 "yt-dlp",
#                 "--restrict-filenames",
#                 "--skip-download",
#                 "--write-auto-sub",                  # prefer auto if human not available
#                 "--sub-langs", "en",                 # download English (may save as en or en-XX)
#                 "--sub-format", "json3/vtt/srt",     # first available of these
#                 "--output", "%(id)s",
#                 "--user-agent", _ua(),
#                 "--no-warnings",
#                 "--quiet",
#                 url,
#             ]
#             cmd = _maybe_add_cookies(cmd)
#             cmd = _maybe_no_mtime(cmd)

#             result = subprocess.run(cmd, capture_output=True, text=True, timeout=45, check=False, cwd=tmpdir)
#             if result.returncode != 0 and result.stderr:
#                 logger.debug(f"yt-dlp (subs) stderr: {result.stderr.strip()}")

#             def _find_first(pattern: str) -> Optional[Path]:
#                 matches = sorted(tmpdir.glob(pattern))
#                 return matches[0] if matches else None

#             # Try a few times for slow disk/network
#             for _ in range(max(1, retries)):
#                 # 1) JSON3 (best for robust text extraction)
#                 json3 = _find_first(f"{video_id}.en*.json3")
#                 if json3 and json3.exists():
#                     try:
#                         data = json.loads(json3.read_text(encoding="utf-8"))
#                         return _process_json3_transcript(data, clean)
#                     except Exception as e:
#                         logger.debug(f"JSON3 parse error: {e}")

#                 # 2) VTT
#                 vtt = _find_first(f"{video_id}.en*.vtt")
#                 if vtt and vtt.exists():
#                     vtt_text = vtt.read_text(encoding="utf-8", errors="ignore")
#                     return extract_vtt_text(vtt_text) if clean else format_transcript_vtt(vtt_text)

#                 # 3) SRT (fallback)
#                 srt = _find_first(f"{video_id}.en*.srt")
#                 if srt and srt.exists():
#                     srt_text = srt.read_text(encoding="utf-8", errors="ignore")
#                     if clean:
#                         # strip indices + timecodes -> plain text
#                         lines = []
#                         for line in srt_text.splitlines():
#                             s = line.strip()
#                             if not s or s.isdigit() or "-->" in s:
#                                 continue
#                             s = re.sub(r"<[^>]+>", "", s)      # drop tags
#                             s = re.sub(r"\{[^}]+\}", "", s)    # drop styling
#                             if s:
#                                 lines.append(s)
#                         return " ".join(lines)
#                     else:
#                         # convert SRT blocks -> "[MM:SS] text"
#                         out_lines = []
#                         cur_time = None
#                         cur_text = []
#                         for line in srt_text.splitlines():
#                             s = line.strip()
#                             if re.search(r"-->", s):
#                                 # new block starting: flush previous
#                                 if cur_time is not None and cur_text:
#                                     out_lines.append(f"[{cur_time}] " + " ".join(cur_text).strip())
#                                 cur_text = []
#                                 # parse start time
#                                 m = re.search(r"(\d{2}):(\d{2}):(\d{2})[,.]\d{3}\s+-->", s)
#                                 if m:
#                                     h, mnt, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
#                                     total = h * 3600 + mnt * 60 + sec
#                                     cur_time = f"{total//60:02d}:{total%60:02d}"
#                                 else:
#                                     cur_time = "00:00"
#                             elif s and not s.isdigit():
#                                 s = re.sub(r"<[^>]+>", "", s)
#                                 s = re.sub(r"\{[^}]+\}", "", s)
#                                 cur_text.append(s)
#                         if cur_time is not None and cur_text:
#                             out_lines.append(f"[{cur_time}] " + " ".join(cur_text).strip())
#                         return "\n".join(out_lines)

#                 time.sleep(max(0, wait_sec))

#             logger.error(f"No transcript files found for video: {video_id}")
#             return None

#     except subprocess.TimeoutExpired:
#         logger.error(f"yt-dlp transcript timeout for {video_id}")
#         return None
#     except Exception as e:
#         logger.error(f"yt-dlp transcript failure for {video_id}: {e}")
#         return None

# def _process_json3_transcript(data: Dict[Any, Any], clean: bool) -> Optional[str]:
#     try:
#         blocks: List[str] = []
#         for event in data.get("events", []):
#             if "segs" in event and "tStartMs" in event:
#                 text_segments = [seg.get("utf8", "") for seg in event["segs"] if seg.get("utf8")]
#                 if not text_segments:
#                     continue
#                 text = "".join(text_segments).strip()
#                 if not text:
#                     continue
#                 if clean:
#                     blocks.append(text)
#                 else:
#                     seconds = int(event["tStartMs"] // 1000)
#                     ts = f"[{seconds//60:02d}:{seconds%60:02d}]"
#                     blocks.append(f"{ts} {text}")
#         if not blocks:
#             return None
#         return (" ".join(blocks)) if clean else "\n".join(blocks)
#     except Exception as e:
#         logger.error(f"Error processing JSON3 transcript: {e}")
#         return None

# def extract_vtt_text(vtt_content: str) -> str:
#     try:
#         lines = vtt_content.strip().splitlines()
#         out: List[str] = []
#         for line in lines:
#             s = line.strip()
#             if not s:
#                 continue
#             if s.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
#                 continue
#             if "-->" in s:
#                 continue
#             if s.isdigit():
#                 continue
#             if re.match(r"^\d{2}:\d{2}:\d{2}", s):
#                 continue
#             s = re.sub(r"<[^>]+>", "", s)      # strip tags
#             s = re.sub(r"\{[^}]+\}", "", s)    # strip styling
#             if s:
#                 out.append(s)
#         return " ".join(out)
#     except Exception as e:
#         logger.error(f"VTT clean error: {e}")
#         return ""

# def format_transcript_vtt(raw_vtt: str) -> str:
#     try:
#         lines = raw_vtt.strip().splitlines()
#         formatted: List[str] = ["WEBVTT", "Kind: captions", "Language: en", ""]
#         i = 0
#         while i < len(lines):
#             s = lines[i].strip()
#             if s.startswith(("WEBVTT", "Kind:", "Language:")):
#                 i += 1
#                 continue
#             if "-->" in s:
#                 s = re.sub(r"(\d{2}:\d{2}:\d{2})[.,](\d{3})", r"\1.\2", s)
#                 formatted.append(s)
#                 i += 1
#                 text_bits = []
#                 while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
#                     t = lines[i].strip()
#                     if not t.isdigit():
#                         t = re.sub(r"<[^>]+>", "", t)
#                         t = re.sub(r"\{[^}]+\}", "", t)
#                         if t:
#                             text_bits.append(t)
#                     i += 1
#                 if text_bits:
#                     formatted.append(" ".join(text_bits))
#                 formatted.append("")
#             else:
#                 i += 1
#         return "\n".join(formatted)
#     except Exception as e:
#         logger.error(f"VTT format error: {e}")
#         return raw_vtt

# # =================
# # FORMAT DISCOVERY
# # =================

# def has_actual_video_formats(formats_output: str) -> bool:
#     try:
#         for line in formats_output.splitlines():
#             s = line.strip().lower()
#             if not s or ("id" in s and "ext" in s):
#                 continue
#             if any(ext in s for ext in ["mp4", "webm", "mkv", "avi", "flv"]):
#                 if "storyboard" not in s and not s.startswith("sb"):
#                     return True
#             parts = line.split()
#             if parts:
#                 fid = parts[0].lower()
#                 if (fid.isdigit() or any(p in fid for p in ["dash", "hls", "http"])) and not fid.startswith("sb"):
#                     return True
#         return False
#     except Exception as e:
#         logger.error(f"Format parse error: {e}")
#         return False

# # ======
# # AUDIO
# # ======

# def download_audio_with_ytdlp(video_id: str, quality: str = "medium", output_dir: str | None = None) -> str:
#     """
#     Download audio as MP3. Returns absolute file path.
#     """
#     out_dir = _ensure_dir(output_dir or DEFAULT_DOWNLOADS_DIR)
#     url = f"https://www.youtube.com/watch?v={video_id}"
#     qmap = {
#         "high":   ("0",   "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio"),
#         "medium": ("2",   "bestaudio[abr<=192]/bestaudio[ext=m4a]/bestaudio"),
#         "low":    ("5",   "bestaudio[abr<=96]/bestaudio[ext=m4a]/bestaudio"),
#     }
#     aq, fmt = qmap.get(quality, qmap["medium"])
#     output_template = f"{video_id}_audio_{quality}.%(ext)s"

#     cmd = [
#         "yt-dlp",
#         "--restrict-filenames",
#         "--extract-audio",
#         "--audio-format", "mp3",
#         "--audio-quality", aq,
#         "--format", fmt,
#         "--output", output_template,
#         "--no-playlist",
#         "--prefer-ffmpeg",
#         "--embed-metadata",
#         "--add-metadata",
#         "--user-agent", _ua(),
#         "--no-warnings",
#         url,
#     ]
#     cmd = _maybe_add_cookies(cmd)
#     cmd = _maybe_no_mtime(cmd)

#     logger.info(f"[audio] {video_id} -> {out_dir} ({quality})")
#     logger.debug(" ".join(cmd))

#     try:
#         result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=out_dir, check=False)
#         if result.stderr:
#             logger.debug(f"yt-dlp(audio) stderr: {result.stderr.strip()}")

#         # Resolve resulting file (prefer the exact template, but recover if needed)
#         candidate = out_dir / f"{video_id}_audio_{quality}.mp3"
#         if not candidate.exists():
#             matches = (
#                 list(out_dir.glob(f"{video_id}_audio_{quality}.mp3"))
#                 or list(out_dir.glob(f"{video_id}_audio_{quality}.*"))
#                 or list(out_dir.glob(f"{video_id}*.mp3"))
#                 or list(out_dir.glob("*.mp3"))
#             )
#             if matches:
#                 # pick latest & largest to be safe
#                 candidate = max(matches, key=lambda f: (f.stat().st_mtime, f.stat().st_size))
#         if not candidate.exists():
#             all_files = [f.name for f in out_dir.iterdir() if f.is_file()]
#             raise Exception(f"No audio file found. Files in dir: {all_files}")

#         if candidate.stat().st_size < 1_000:
#             try:
#                 candidate.unlink()
#             except Exception:
#                 pass
#             raise Exception("Downloaded audio appears corrupted/empty")

#         # Force mtime->now so it shows under 'Today' on Windows
#         _touch_now(candidate)

#         return str(candidate.absolute())

#     except subprocess.TimeoutExpired:
#         raise Exception("Audio download timed out")
#     except Exception as e:
#         raise Exception(f"Audio download error: {e}")


# # ======
# # VIDEO
# # ======

# def download_video_with_ytdlp(video_id: str, quality: str = "720p", output_dir: str | None = None) -> Optional[str]:
#     """
#     Download a video file (merged MP4 when possible). Returns absolute path or raises Exception.
#     """
#     out_dir = _ensure_dir(output_dir or DEFAULT_DOWNLOADS_DIR)
#     url = f"https://www.youtube.com/watch?v={video_id}"
#     height = _parse_height(quality, 720)
#     output_template = f"{video_id}_video_{quality}.%(ext)s"

#     # Step 1: list formats to detect restrictions  ✅ add --restrict-filenames here
#     try:
#         list_cmd = ["yt-dlp", "--restrict-filenames", "--list-formats", "--no-warnings", "--user-agent", _ua(), url]
#         list_cmd = _maybe_add_cookies(list_cmd)
#         list_cmd = _maybe_no_mtime(list_cmd)
#         fmts = subprocess.run(list_cmd, capture_output=True, text=True, timeout=60, check=False)
#         if fmts.returncode != 0:
#             raise Exception(f"Cannot access video formats: {fmts.stderr.strip() or 'unknown error'}")
#         if not has_actual_video_formats(fmts.stdout):
#             raise Exception("Only storyboard formats found (likely age/region/restricted content). Try another video.")
#     except subprocess.TimeoutExpired:
#         raise Exception("Format listing timed out")

#     # Step 2: strategies
#     strategies = [
#         f"best[height<={height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={height}]",
#         f"(bestvideo[height<={height}]+bestaudio/best[height<={height}])[ext=mp4]/(bestvideo[height<={height}]+bestaudio/best[height<={height}])",
#         "bestvideo+bestaudio/best[ext=mp4]/best",
#         "best/worst",
#     ]

#     last_err = None
#     for i, fmt in enumerate(strategies, 1):
#         cmd = [
#             "yt-dlp",
#             "--restrict-filenames",           # ✅ keep here too
#             "--no-playlist",
#             "--output", output_template,
#             "--format", fmt,
#             "--merge-output-format", "mp4",
#             "--embed-metadata",
#             "--add-metadata",
#             "--retries", "3",
#             "--fragment-retries", "3",
#             "--user-agent", _ua(),
#             "--no-warnings",
#             url,
#         ]
#         cmd = _maybe_add_cookies(cmd)
#         cmd = _maybe_no_mtime(cmd)
#         logger.info(f"[video] strategy {i}/{len(strategies)} -> {fmt}")
#         logger.debug(" ".join(cmd))
#         try:
#             r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=out_dir, check=False)
#             if r.returncode == 0:
#                 break
#             last_err = r.stderr.strip() or r.stdout.strip() or "unknown error"
#             logger.debug(f"strategy {i} failed: {last_err[:400]}")
#             if i == len(strategies):
#                 raise Exception(last_err)
#         except subprocess.TimeoutExpired:
#             last_err = "download timed out"
#             if i == len(strategies):
#                 raise Exception(last_err)

#     # Step 3: resolve file
#     patterns = [
#         out_dir / f"{video_id}_video_{quality}.mp4",
#         out_dir / f"{video_id}_video_{quality}.*",
#         out_dir / f"{video_id}_video.*",
#         out_dir / f"{video_id}*.*",
#     ]
#     found: Optional[Path] = None
#     for pat in patterns:
#         matches = list(out_dir.glob(pat.name if pat.is_file() else pat.name))
#         if matches:
#             found = max(matches, key=lambda f: f.stat().st_size)
#             break
#     if not found or not found.exists():
#         all_files = [f.name for f in out_dir.iterdir() if f.is_file()]
#         raise Exception(f"Video file not found after download. Files: {all_files}")

#     if found.stat().st_size < 100_000:
#         try: found.unlink()
#         except Exception: pass
#         raise Exception("Downloaded video appears corrupted/too small")

#     _touch_now(found)
#     return str(found.absolute())

# # =========================
# # INFO / CHECKS / ESTIMATES
# # =========================

# def get_video_info(video_id: str) -> Optional[Dict[str, Any]]:
#     """Return compact metadata for a video."""
#     url = f"https://www.youtube.com/watch?v={video_id}"
#     cmd = ["yt-dlp", "--dump-json", "--no-warnings", "--no-download", "--user-agent", _ua(), url]
#     cmd = _maybe_add_cookies(cmd)
#     cmd = _maybe_no_mtime(cmd)
#     try:
#         r = subprocess.run(cmd, capture_output=True, text=True, timeout=45, check=False)
#         if r.returncode == 0 and r.stdout:
#             try:
#                 vi = json.loads(r.stdout)
#             except json.JSONDecodeError:
#                 last_line = r.stdout.strip().splitlines()[-1]
#                 vi = json.loads(last_line)
#             return {
#                 "id": vi.get("id"),
#                 "title": vi.get("title"),
#                 "duration": vi.get("duration"),
#                 "upload_date": vi.get("upload_date"),
#                 "uploader": vi.get("uploader"),
#                 "uploader_id": vi.get("uploader_id"),
#                 "view_count": vi.get("view_count"),
#                 "like_count": vi.get("like_count"),
#                 "description": (vi.get("description") or "")[:500],
#                 "thumbnail": vi.get("thumbnail"),
#                 "has_subtitles": bool(vi.get("subtitles")),
#                 "has_auto_captions": bool(vi.get("automatic_captions")),
#                 "format_note": vi.get("format_note"),
#                 "ext": vi.get("ext"),
#                 "filesize": vi.get("filesize"),
#                 "fps": vi.get("fps"),
#                 "width": vi.get("width"),
#                 "height": vi.get("height"),
#                 "age_limit": vi.get("age_limit", 0),
#                 "availability": vi.get("availability"),
#             }
#         return None
#     except (subprocess.TimeoutExpired, Exception) as e:
#         logger.error(f"get_video_info error for {video_id}: {e}")
#         return None

# def check_ytdlp_availability() -> bool:
#     """True if yt-dlp exists (and ffmpeg if possible)."""
#     try:
#         r = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, timeout=10, check=False)
#         if r.returncode != 0:
#             return False
#     except Exception:
#         return False
#     try:
#         subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10, check=False)
#     except Exception:
#         pass
#     return True

# def estimate_file_size(video_id: str, download_type: str, quality: str) -> Optional[int]:
#     try:
#         info = get_video_info(video_id)
#         if not info or not info.get("duration"):
#             return None
#         duration = int(info["duration"])  # seconds
#         if download_type == "audio":
#             bitrates = {"high": 320, "medium": 192, "low": 96}
#             bitrate = bitrates.get(quality, 192)
#             return int((duration * bitrate * 1000) // 8)
#         if download_type == "video":
#             size_per_min = {1080: 100 * 1024 * 1024, 720: 50 * 1024 * 1024, 480: 25 * 1024 * 1024, 360: 15 * 1024 * 1024}
#             # nearest bucket
#             h = _parse_height(quality, 720)
#             key = min(size_per_min.keys(), key=lambda k: abs(k - h))
#             return int((duration * size_per_min[key]) // 60)
#         return None
#     except Exception as e:
#         logger.error(f"estimate_file_size error: {e}")
#         return None

# # =============================================================================
# # Misc utilities used elsewhere / tests
# # =============================================================================

# def validate_video_id(video_id: str) -> bool:
#     if not video_id or len(video_id) != 11:
#         return False
#     import string
#     valid = set(string.ascii_letters + string.digits + "-_")
#     return all(c in valid for c in video_id)

# def format_transcript_clean(text: str) -> str:
#     try:
#         sentences = re.split(r"[.!?]+\s+", text)
#         paras: List[str] = []
#         buf: List[str] = []
#         count = 0
#         for s in sentences:
#             s = s.strip()
#             if not s:
#                 continue
#             buf.append(s)
#             count += len(s) + 1
#             if count > 400:
#                 paras.append(". ".join(buf) + ".")
#                 buf, count = [], 0
#         if buf:
#             paras.append(". ".join(buf) + ".")
#         return "\n\n".join(paras)
#     except Exception:
#         return text

# def get_downloads_directory() -> Path:
#     return DEFAULT_DOWNLOADS_DIR

# def set_downloads_directory(path: str) -> bool:
#     try:
#         p = Path(path)
#         p.mkdir(parents=True, exist_ok=True)
#         t = p / "test_write.tmp"
#         t.write_text("ok")
#         t.unlink(missing_ok=True)
#         global DEFAULT_DOWNLOADS_DIR
#         DEFAULT_DOWNLOADS_DIR = p
#         logger.info(f"Downloads directory updated to: {p}")
#         return True
#     except Exception as e:
#         logger.error(f"set_downloads_directory error: {e}")
#         return False

# # ------------------ Optional local test harness ------------------

# def _test_transcript(video_id: str = "dQw4w9WgXcQ"):
#     clean = get_transcript_with_ytdlp(video_id, clean=True)
#     ts = get_transcript_with_ytdlp(video_id, clean=False)
#     logger.info(f"Transcript clean={bool(clean)} ts={bool(ts)}")
#     return clean, ts

# def _test_audio(video_id: str = "dQw4w9WgXcQ", quality: str = "medium"):
#     with tempfile.TemporaryDirectory() as td:
#         return bool(download_audio_with_ytdlp(video_id, quality, td))

# def _test_video(video_id: str = "dQw4w9WgXcQ", quality: str = "720p"):
#     with tempfile.TemporaryDirectory() as td:
#         p = download_video_with_ytdlp(video_id, quality, td)
#         return bool(p and Path(p).exists())

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     print("yt-dlp available:", check_ytdlp_availability())
#     print("Downloads dir:", DEFAULT_DOWNLOADS_DIR)
#     try:
#         _test_transcript()
#         print("Transcript test: OK")
#     except Exception as e:
#         print("Transcript test failed:", e)
#     try:
#         ok = _test_audio()
#         print("Audio test:", "OK" if ok else "FAIL")
#     except Exception as e:
#         print("Audio test failed:", e)
#     try:
#         ok = _test_video()
#         print("Video test:", "OK" if ok else "FAIL")
#     except Exception as e:
#         print("Video test failed:", e)


