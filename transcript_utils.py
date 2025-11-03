# backend/transcript_utils.py
"""
Utilities for transcripts and media downloads.
Framework-free to avoid circular imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List
import logging
import os
import re
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

logger = logging.getLogger("youtube_trans_downloader")

#------------------- Newly Added functions ------------

def _norm_youtube_url(video_id_or_url: str) -> str:
    # Accept ID or any YT URL (watch/shorts/youtu.be)
    s = video_id_or_url.strip()
    m = re.search(r'(?:v=|/shorts/|youtu\.be/)([A-Za-z0-9_-]{6,})', s)
    vid = m.group(1) if m else s
    return f'https://www.youtube.com/watch?v={vid}'

def _ensure_ffmpeg_location() -> Optional[str]:
    # If you ship FFmpeg with the app, return its folder here and set in opts
    # Otherwise return None and make sure FFmpeg is on PATH
    return None

def _outtmpl(output_dir: str) -> Dict[str, str]:
    # Title + ID + height; Windows-safe; keeps original ext after remux
    # .200B caps long titles safely
    return {
        'default': os.path.join(
            output_dir, '%(title).200B [%(id)s]_%(height)sp.%(ext)s'
        )
    }

def _common_ydl_opts(output_dir: str) -> Dict:
    return {
        # Avoid the problematic web client; prefer Android to bypass SABR
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'android_embedded']
            }
        },
        # Pick AVC video first (widely compatible), then best available; always include audio
        'format': (
            # mp4/avc first, merged with best audio
            'bv*[ext=mp4][vcodec^=avc1]+ba[ext=m4a]/'
            # any bestvideo + bestaudio fallback
            'bv*+ba/best'
        ),
        # Merge/remux to MP4 even if the best stream is webm
        'merge_output_format': 'mp4',
        'postprocessors': [
            {'key': 'FFmpegVideoRemuxer', 'preferedformat': 'mp4'},
            {'key': 'FFmpegMetadata'},         # write title/artist where possible
            {'key': 'EmbedThumbnail', 'already_have_thumbnail': False},  # harmless if none
        ],
        'writethumbnail': True,
        'outtmpl': _outtmpl(output_dir),
        'noprogress': True,
        'quiet': True,
        'concurrent_fragment_downloads': 4,
        'retries': 5,
        'fragment_retries': 5,
        # Slightly “Android-ish” headers help in some edge cases
        'http_headers': {
            'User-Agent': 'com.google.android.youtube/19.20.34 (Linux; U; Android 11)',
            'Accept-Language': 'en-US,en;q=0.9',
        },
        # Make sure ffmpeg is found
        **({'ffmpeg_location': _ensure_ffmpeg_location()} if _ensure_ffmpeg_location() else {})
    }
#-------------------- End of - Newly Added functions -------------

# def _common_ydl_opts(tmp_dir: str | None = None) -> dict:
#     """Base yt-dlp options shared by audio/video flows."""
#     prefer_ipv4 = str(os.getenv("YTDLP_BIND_IPV4", "1")).lower() in ("1", "true", "yes")
#     headers = {
#         # Keep a modern UA; some CDNs 403 on unknown agents
#         "User-Agent": (
#             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#             "AppleWebKit/537.36 (KHTML, like Gecko) "
#             "Chrome/119.0 Safari/537.36"
#         ),
#         "Accept-Language": "en-US,en;q=0.9",
#     }
#     opts = {
#         "quiet": True,
#         "noprogress": True,
#         "concurrent_fragment_downloads": 4,
#         "retries": 3,
#         "fragment_retries": 10,
#         "retry_sleep": "1,2,4,8",
#         "http_headers": headers,
#         "nocheckcertificate": True,
#         # IPv4 preference to dodge some ISP/IPv6 routing/CDN blocks
#         "prefer_ipv6": False if prefer_ipv4 else None,
#         "source_address": "0.0.0.0" if prefer_ipv4 else None,
#         # Temp/work dir
#         "paths": {"home": tmp_dir} if tmp_dir else None,
#     }
#     # Remove None values to keep yt-dlp happy
#     return {k: v for k, v in opts.items() if v is not None}



# -----------------------
# ID / validation helpers
# -----------------------
_YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def validate_video_id(video_id: str) -> bool:
    return bool(_YT_ID_RE.fullmatch((video_id or "").strip()))


# -----------------------
# yt-dlp / ffmpeg helpers
# -----------------------
def check_ytdlp_availability() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except Exception as e:
        logger.debug("yt-dlp not available: %s", e)
        return False


def _yt_dlp():
    import yt_dlp  # imported lazily
    return yt_dlp


def _youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _ffmpeg_location_opts() -> Dict[str, Any]:
    """
    Honor FFMPEG_PATH env var on Windows if user installed a local binary.
    """
    ff = os.getenv("FFMPEG_PATH")
    if ff and Path(ff).exists():
        return {"ffmpeg_location": ff}
    return {}


# -----------------------
# Video info via yt-dlp
# -----------------------
def get_video_info(video_id: str) -> Dict[str, Any]:
    if not validate_video_id(video_id):
        raise ValueError("Invalid YouTube video ID")

    yt_dlp = _yt_dlp()
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        **_ffmpeg_location_opts(),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(_youtube_url(video_id), download=False)
        return {
            "id": info.get("id"),
            "title": info.get("title"),
            "uploader": info.get("uploader") or info.get("channel"),
            "duration": info.get("duration"),
        }


# -----------------------
# Transcript helpers
# -----------------------
_EN_PRIORITY = ["en", "en-US", "en-GB", "en-CA", "en-AU", "en-IE", "en-NZ"]


def _clean_plain_blocks(blocks: List[str]) -> str:
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


def _get_segment_value(seg: Any, key: str, default: Any = None) -> Any:
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def get_transcript_with_ytdlp(video_id: str, clean: bool = True) -> Optional[str]:
    """
    Best-effort transcript retrieval returning **plain text**.
    Uses youtube_transcript_api; kept here so main can call a single place.
    """
    try:
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            NoTranscriptFound,
            TranscriptsDisabled,
        )
    except Exception as e:
        logger.warning("youtube_transcript_api not installed: %s", e)
        return None

    try:
        listing = YouTubeTranscriptApi.list_transcripts(video_id)

        # 1) Authored English variants
        for code in _EN_PRIORITY:
            try:
                t = listing.find_transcript([code])
                segs = t.fetch()
                texts = [
                    _get_segment_value(s, "text", "").replace("\n", " ")
                    for s in segs
                    if _get_segment_value(s, "text", "")
                ]
                return _clean_plain_blocks(texts) if clean else "\n".join(texts)
            except NoTranscriptFound:
                continue
            except Exception:
                continue

        # 2) Generated English
        try:
            t = listing.find_generated_transcript(_EN_PRIORITY)
            segs = t.fetch()
            texts = [
                _get_segment_value(s, "text", "").replace("\n", " ")
                for s in segs
                if _get_segment_value(s, "text", "")
            ]
            return _clean_plain_blocks(texts) if clean else "\n".join(texts)
        except NoTranscriptFound:
            pass
        except Exception:
            pass

        # 3) Translate to en
        for t in listing:
            try:
                segs = t.translate("en").fetch()
                texts = [
                    _get_segment_value(s, "text", "").replace("\n", " ")
                    for s in segs
                    if _get_segment_value(s, "text", "")
                ]
                return _clean_plain_blocks(texts) if clean else "\n".join(texts)
            except Exception:
                continue

        # 4) Direct fallback
        try:
            segs = YouTubeTranscriptApi.get_transcript(video_id, languages=_EN_PRIORITY)
            texts = [
                _get_segment_value(s, "text", "").replace("\n", " ")
                for s in segs
                if _get_segment_value(s, "text", "")
            ]
            return _clean_plain_blocks(texts) if clean else "\n".join(texts)
        except Exception:
            return None

    except TranscriptsDisabled:
        logger.info("Transcripts disabled for %s", video_id)
        return None
    except Exception as e:
        logger.debug("Transcript retrieval failed for %s: %s", video_id, e)
        return None


# -----------------------
# Media download helpers
# -----------------------
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _audio_quality_to_kbps(quality: str) -> str:
    q = (quality or "").lower()
    if q in {"low", "l"}:
        return "96"
    if q in {"high", "h"}:
        return "256"
    return "160"  # default medium


def _video_quality_to_height(quality: str) -> int:
    q = (quality or "").lower()
    if q in {"144p", "144"}:
        return 144
    if q in {"240p", "240"}:
        return 240
    if q in {"360p", "360"}:
        return 360
    if q in {"480p", "480"}:
        return 480
    if q in {"720p", "720"}:
        return 720
    if q in {"1080p", "1080", "fullhd", "fhd"}:
        return 1080
    if q in {"1440p", "1440", "2k"}:
        return 1440
    if q in {"2160p", "2160", "4k", "uhd"}:
        return 2160
    return 720


def _finalize_path(output_dir: str, filename: str) -> Path:
    base = Path(output_dir).resolve()
    _ensure_dir(base)
    return base / filename

def download_video_with_ytdlp(video_id_or_url: str, quality: str, output_dir: str) -> str:
    """
    quality: '1080p' | '720p' | '480p' | '360p'
    Returns final absolute file path.
    """
    url = _norm_youtube_url(video_id_or_url)
    q = re.sub(r'[^0-9]', '', quality or '')
    height = int(q) if q.isdigit() else None

    fmt = _common_ydl_opts(output_dir)
    if height:
        # prefer requested height, then nearest lower, still merging with audio
        fmt['format'] = (
            f"bv*[height={height}][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
            f"bv*[height<={height}][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
            "bv*+ba/best"
        )

    os.makedirs(output_dir, exist_ok=True)

    with YoutubeDL(fmt) as ydl:
        info = ydl.extract_info(url, download=True)
        # yt-dlp returns final filename here (after remux/merge)
        fname = ydl.prepare_filename(info)
        # after remux it might have changed extension to mp4
        base, _ = os.path.splitext(fname)
        mp4 = base + '.mp4'
        return mp4 if os.path.exists(mp4) else fname

def download_audio_with_ytdlp(video_id_or_url: str, output_dir: str) -> str:
    """
    Extracts audio to MP3 (with correct title in filename).
    """
    url = _norm_youtube_url(video_id_or_url)
    opts = _common_ydl_opts(output_dir)
    opts.update({
        'format': 'bestaudio/best',
        'postprocessors': [
            {'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'},
            {'key': 'FFmpegMetadata'},
            {'key': 'EmbedThumbnail', 'already_have_thumbnail': False},
        ],
        'outtmpl': {
            'default': os.path.join(output_dir, '%(title).200B [%(id)s].%(ext)s')
        }
    })

    os.makedirs(output_dir, exist_ok=True)
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # yt-dlp tells us the actual filename after audio extraction
        return ydl.prepare_filename(info).rsplit('.', 1)[0] + '.mp3'
#----------------------

# def download_audio_with_ytdlp(video_id: str, quality: str, output_dir: str) -> str:
#     """
#     Extracts audio as MP3 using FFmpegExtractAudio (the robust, standard PP).
#     """
#     abr_map = {"low": "64", "medium": "128", "high": "192"}
#     abr = abr_map.get(quality.lower(), "128")
#     base_opts = _common_ydl_opts(tmp_dir=output_dir)

#     ydl_opts = {
#         **base_opts,
#         "format": "bestaudio/best",
#         "outtmpl": os.path.join(output_dir, f"{video_id}_audio_%(abr)sk.%(ext)s"),
#         "postprocessors": [
#             {
#                 "key": "FFmpegExtractAudio",
#                 "preferredcodec": "mp3",
#                 "preferredquality": abr,
#             }
#         ],
#         # Better container probing
#         "postprocessor_args": ["-vn"],
#     }

#     url = f"https://www.youtube.com/watch?v={video_id}"
#     with YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(url, download=True)
#         # yt-dlp returns output paths in info when postprocessors run
#         # We normalize to our deterministic file name:
#         path = os.path.join(output_dir, f"{video_id}_audio_{abr}k.mp3")
#         if not os.path.exists(path):
#             # fallback: find the first created file
#             candidates = [f for f in os.listdir(output_dir) if f.startswith(f"{video_id}_audio_") and f.endswith(".mp3")]
#             if candidates:
#                 path = os.path.join(output_dir, candidates[0])
#         return path


# def download_video_with_ytdlp(video_id: str, quality: str, output_dir: str) -> str:
#     """
#     Downloads MP4 video at the requested quality (1080p/720p/480p/360p).
#     """
#     fmt_map = {
#         "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
#         "720p":  "bestvideo[height<=720]+bestaudio/best[height<=720]",
#         "480p":  "bestvideo[height<=480]+bestaudio/best[height<=480]",
#         "360p":  "bestvideo[height<=360]+bestaudio/best[height<=360]",
#     }
#     fmt = fmt_map.get(quality.lower(), fmt_map["720p"])

#     base_opts = _common_ydl_opts(tmp_dir=output_dir)
#     ydl_opts = {
#         **base_opts,
#         "format": fmt,
#         "merge_output_format": "mp4",
#         "outtmpl": os.path.join(output_dir, f"{video_id}_video_{quality.lower()}.%(ext)s"),
#         "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
#     }

#     url = f"https://www.youtube.com/watch?v={video_id}"
#     with YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(url, download=True)
#         path = os.path.join(output_dir, f"{video_id}_video_{quality.lower()}.mp4")
#         return path

