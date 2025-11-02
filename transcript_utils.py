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

logger = logging.getLogger("youtube_trans_downloader")

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


def download_audio_with_ytdlp(video_id: str, quality: str = "medium", output_dir: str = ".") -> str:
    """
    Download audio and transcode to **MP3** using a single, stable post-processor.
    Returns the final .mp3 path.
    """
    if not validate_video_id(video_id):
        raise ValueError("Invalid YouTube video ID")
    if not check_ytdlp_availability():
        raise RuntimeError("yt-dlp is not available")

    kbps = _audio_quality_to_kbps(quality)
    # Deterministic template; ExtractAudio will replace ext to .mp3
    out_tmpl = _finalize_path(output_dir, f"{video_id}_audio_{quality}.%(ext)s")

    yt_dlp = _yt_dlp()
    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "concurrent_fragment_downloads": 1,  # reduces flakiness on Windows
        "outtmpl": str(out_tmpl),
        "format": "bestaudio/best",
        "prefer_ffmpeg": True,
        "postprocessors": [
            {
                # ✅ stable, supported PP – handles extraction & mp3 encode
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": kbps,  # kbps as string, e.g. "160"
            },
            {"key": "FFmpegMetadata"},
        ],
        **_ffmpeg_location_opts(),
    }

    # Execute
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(_youtube_url(video_id), download=True)

    # Resolve expected final path
    final_path = _finalize_path(output_dir, f"{video_id}_audio_{quality}.mp3")
    if final_path.exists() and final_path.stat().st_size > 0:
        return str(final_path)

    # Fallback: search produced files
    for p in Path(output_dir).glob(f"{video_id}_audio_{quality}*.mp3"):
        if p.is_file() and p.stat().st_size > 0:
            return str(p.resolve())

    # Last resort: infer from yt-dlp filename hint
    fname = info.get("_filename")
    if fname:
        p = Path(fname).with_suffix(".mp3")
        if p.exists() and p.stat().st_size > 0:
            return str(p.resolve())

    raise RuntimeError("Audio file was not produced")

def download_video_with_ytdlp(video_id: str, quality: str = "720p", output_dir: str = ".") -> str:
    """
    Download video merged with audio to MP4 with height <= requested.
    Strongly prefers non-HLS to avoid empty-file + 403 fragment issues.
    Falls back through safer formats if the first choice is blocked.
    """
    if not validate_video_id(video_id):
        raise ValueError("Invalid YouTube video ID")
    if not check_ytdlp_availability():
        raise RuntimeError("yt-dlp is not available")

    yt_dlp = _yt_dlp()
    height = _video_quality_to_height(quality)

    # Prefer AVC MP4 video + M4A audio, explicitly avoid HLS when possible.
    # Fallbacks: best MP4, then non-HLS best, then anything that works.
    fmt = (
        f"(bestvideo[ext=mp4][vcodec^=avc1][protocol!=m3u8][height<={height}]"
        f"/bestvideo[ext=mp4][protocol!=m3u8][height<={height}]"
        f"/bestvideo[protocol!=m3u8][height<={height}])"
        f"+(bestaudio[ext=m4a]/bestaudio)"
        f"/best[ext=mp4][protocol!=m3u8][height<={height}]"
        f"/best[protocol!=m3u8][height<={height}]"
        f"/best[height<={height}]"
    )

    out_tmpl = _finalize_path(output_dir, f"{video_id}_video_{quality}.%(ext)s")

    # Optional cookies from env (one of these; both are optional)
    cookies_from_browser = os.getenv("COOKIES_FROM_BROWSER")  # e.g., "chrome"
    cookies_file = os.getenv("COOKIES_FILE")  # e.g., "cookies.txt"

    # Desktop UA + robust retries/backoff
    http_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        )
    }

    base_opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "outtmpl": str(out_tmpl),
        "noplaylist": True,
        "format": fmt,
        "merge_output_format": "mp4",
        "prefer_ffmpeg": True,
        "http_headers": http_headers,
        "retries": 10,
        "fragment_retries": 10,
        "retry_sleep_functions": {
            "http": "exponential_backoff",
            "fragment": "exponential_backoff",
        },
        "concurrent_fragment_downloads": 1,  # prevents Windows file handle issues
        "skip_unavailable_fragments": True,
        "geo_bypass": True,
        "geo_bypass_country": "US",
        # Force a stable YouTube client to reduce 403s due to odd signatures:
        "extractor_args": {
            "youtube": {
                "player_client": ["web"],
                # Keep DASH manifest; we already exclude HLS by format filter.
                "include_hls_manifest": ["no"],
            }
        },
        **_ffmpeg_location_opts(),
    }

    if cookies_from_browser:
        base_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if cookies_file and Path(cookies_file).exists():
        base_opts["cookiefile"] = cookies_file

    # Try primary attempt; if it fails with 403, fallback to a very safe format.
    try:
        with yt_dlp.YoutubeDL(base_opts) as ydl:
            info = ydl.extract_info(_youtube_url(video_id), download=True)
    except Exception as e:
        # Fallback to progressive MP4 only (no mux), excluding HLS entirely.
        fallback_fmt = (
            f"best[ext=mp4][protocol!=m3u8][protocol!=dash][height<={height}]"
            f"/best[ext=mp4][protocol!=m3u8][height<={height}]"
            f"/best[ext=mp4][height<={height}]"
        )
        fb_opts = dict(base_opts)
        fb_opts["format"] = fallback_fmt
        fb_opts["postprocessors"] = [
            {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
            {"key": "FFmpegMetadata"},
        ]
        with yt_dlp.YoutubeDL(fb_opts) as ydl:
            info = ydl.extract_info(_youtube_url(video_id), download=True)

    final_path = _finalize_path(output_dir, f"{video_id}_video_{quality}.mp4")
    if final_path.exists() and final_path.stat().st_size > 0:
        return str(final_path)

    # Resolve alternative names produced by yt-dlp
    for p in Path(output_dir).glob(f"{video_id}_video_{quality}*.mp4"):
        if p.is_file() and p.stat().st_size > 0:
            return str(p.resolve())

    fname = info.get("_filename")
    if fname:
        p = Path(fname)
        if p.suffix.lower() != ".mp4":
            p = p.with_suffix(".mp4")
        if p.exists() and p.stat().st_size > 0:
            return str(p.resolve())

    raise RuntimeError("Video file was not produced")


# def download_video_with_ytdlp(video_id: str, quality: str = "720p", output_dir: str = ".") -> str:
#     """
#     Download video merged with audio to MP4 with height <= requested.
#     """
#     if not validate_video_id(video_id):
#         raise ValueError("Invalid YouTube video ID")
#     if not check_ytdlp_availability():
#         raise RuntimeError("yt-dlp is not available")

#     height = _video_quality_to_height(quality)
#     out_tmpl = _finalize_path(output_dir, f"{video_id}_video_{quality}.%(ext)s")

#     fmt = (
#         f"bestvideo[ext=mp4][height<={height}]+bestaudio[ext=m4a]/"
#         f"best[ext=mp4][height<={height}]/"
#         f"bestvideo[height<={height}]+bestaudio/best[height<={height}]"
#     )

#     yt_dlp = _yt_dlp()
#     ydl_opts = {
#         "quiet": True,
#         "no_warnings": True,
#         "noprogress": True,
#         "outtmpl": str(out_tmpl),
#         "format": fmt,
#         "merge_output_format": "mp4",
#         "prefer_ffmpeg": True,
#         "postprocessors": [
#             {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},  # yt-dlp internal spelling
#             {"key": "FFmpegMetadata"},
#         ],
#         **_ffmpeg_location_opts(),
#     }

#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(_youtube_url(video_id), download=True)

#     final_path = _finalize_path(output_dir, f"{video_id}_video_{quality}.mp4")
#     if final_path.exists() and final_path.stat().st_size > 0:
#         return str(final_path)

#     for p in Path(output_dir).glob(f"{video_id}_video_{quality}*.mp4"):
#         if p.is_file() and p.stat().st_size > 0:
#             return str(p.resolve())

#     fname = info.get("_filename")
#     if fname:
#         p = Path(fname).with_suffix(".mp4")
#         if p.exists() and p.stat().st_size > 0:
#             return str(p.resolve())

#     raise RuntimeError("Video file was not produced")

# #=========================
# # backend/transcript_utils.py
# """
# Utilities for transcripts and media downloads.

# This module is intentionally **framework-free** (no FastAPI routes) to avoid
# circular imports. `main.py` can safely import from here.

# Requires:
# - yt-dlp (and ffmpeg in PATH for muxing/transcoding)
# - youtube-transcript-api (optional but recommended)
# """
# from __future__ import annotations

# from pathlib import Path
# from typing import Any, Dict, Optional, Tuple, List
# import logging
# import re
# import os

# logger = logging.getLogger("youtube_trans_downloader")

# # -----------------------
# # ID / validation helpers
# # -----------------------
# _YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# def validate_video_id(video_id: str) -> bool:
#     """Strictly validate a YouTube 11-char ID."""
#     return bool(_YT_ID_RE.fullmatch((video_id or "").strip()))


# # -----------------------
# # yt-dlp availability
# # -----------------------
# def check_ytdlp_availability() -> bool:
#     try:
#         import yt_dlp  # noqa: F401
#         return True
#     except Exception as e:
#         logger.debug("yt-dlp not available: %s", e)
#         return False


# # -----------------------
# # Video info via yt-dlp
# # -----------------------
# def _yt_dlp() -> Any:
#     import yt_dlp  # imported late to avoid mandatory dep for non-download paths
#     return yt_dlp

# def _youtube_url(video_id: str) -> str:
#     return f"https://www.youtube.com/watch?v={video_id}"

# def get_video_info(video_id: str) -> Dict[str, Any]:
#     """
#     Extract basic info for a YouTube video without downloading media.
#     """
#     if not validate_video_id(video_id):
#         raise ValueError("Invalid YouTube video ID")

#     yt_dlp = _yt_dlp()
#     ydl_opts = {
#         "quiet": True,
#         "no_warnings": True,
#         "skip_download": True,
#     }
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(_youtube_url(video_id), download=False)
#         return {
#             "id": info.get("id"),
#             "title": info.get("title"),
#             "uploader": info.get("uploader") or info.get("channel"),
#             "duration": info.get("duration"),
#         }


# # -----------------------
# # Transcript helpers
# # -----------------------
# _EN_PRIORITY = ["en", "en-US", "en-GB", "en-CA", "en-AU", "en-IE", "en-NZ"]

# def _clean_plain_blocks(blocks: List[str]) -> str:
#     """
#     Turn raw segment texts into readable paragraphs (~400+ chars per paragraph).
#     """
#     out, cur, chars = [], [], 0
#     for word in " ".join(blocks).split():
#         cur.append(word)
#         chars += len(word) + 1
#         if chars > 400 and word.endswith((".", "!", "?")):
#             out.append(" ".join(cur))
#             cur, chars = [], 0
#     if cur:
#         out.append(" ".join(cur))
#     return "\n\n".join(out)

# def _get_segment_value(seg: Any, key: str, default: Any = None) -> Any:
#     if isinstance(seg, dict):
#         return seg.get(key, default)
#     return getattr(seg, key, default)


# def get_transcript_with_ytdlp(video_id: str, clean: bool = True) -> Optional[str]:
#     """
#     Best-effort transcript retrieval returning **plain text**.
#     Despite the name, this uses `youtube_transcript_api` because that's
#     the most reliable way to fetch text segments. It's kept here so
#     `main.py` can call it as a fallback without importing that package directly.
#     """
#     try:
#         from youtube_transcript_api import (
#             YouTubeTranscriptApi,
#             NoTranscriptFound,
#             TranscriptsDisabled,
#         )
#     except Exception as e:
#         logger.warning("youtube_transcript_api not installed: %s", e)
#         return None

#     try:
#         # 1) Try authored English variants (priority order)
#         listing = YouTubeTranscriptApi.list_transcripts(video_id)
#         for code in _EN_PRIORITY:
#             try:
#                 t = listing.find_transcript([code])
#                 segments = t.fetch()
#                 texts = [
#                     _get_segment_value(s, "text", "").replace("\n", " ")
#                     for s in segments
#                     if _get_segment_value(s, "text", "")
#                 ]
#                 return _clean_plain_blocks(texts) if clean else "\n".join(texts)
#             except NoTranscriptFound:
#                 continue
#             except Exception:
#                 continue

#         # 2) Generated/auto English
#         try:
#             t = listing.find_generated_transcript(_EN_PRIORITY)
#             segments = t.fetch()
#             texts = [
#                 _get_segment_value(s, "text", "").replace("\n", " ")
#                 for s in segments
#                 if _get_segment_value(s, "text", "")
#             ]
#             return _clean_plain_blocks(texts) if clean else "\n".join(texts)
#         except NoTranscriptFound:
#             pass
#         except Exception:
#             pass

#         # 3) Translate some track to English
#         for t in listing:
#             try:
#                 translated = t.translate("en")
#                 segments = translated.fetch()
#                 texts = [
#                     _get_segment_value(s, "text", "").replace("\n", " ")
#                     for s in segments
#                     if _get_segment_value(s, "text", "")
#                 ]
#                 return _clean_plain_blocks(texts) if clean else "\n".join(texts)
#             except Exception:
#                 continue

#         # 4) Fallback: direct get_transcript
#         try:
#             segments = YouTubeTranscriptApi.get_transcript(video_id, languages=_EN_PRIORITY)
#             texts = [
#                 _get_segment_value(s, "text", "").replace("\n", " ")
#                 for s in segments
#                 if _get_segment_value(s, "text", "")
#             ]
#             return _clean_plain_blocks(texts) if clean else "\n".join(texts)
#         except Exception:
#             return None

#     except TranscriptsDisabled:
#         logger.info("Transcripts disabled for %s", video_id)
#         return None
#     except Exception as e:
#         logger.debug("Transcript retrieval failed for %s: %s", video_id, e)
#         return None


# # -----------------------
# # Media download helpers
# # -----------------------
# def _ensure_dir(path: Path) -> None:
#     path.mkdir(parents=True, exist_ok=True)

# def _audio_quality_to_bitrate(quality: str) -> str:
#     """
#     Map abstract quality to bitrate. Returns ffmpeg '-b:a' value.
#     """
#     q = (quality or "").lower()
#     if q in {"low", "l"}:
#         return "96k"
#     if q in {"high", "h"}:
#         return "256k"
#     return "160k"  # default 'medium'

# def _video_quality_to_height(quality: str) -> int:
#     q = (quality or "").lower()
#     if q in {"144p", "144"}: return 144
#     if q in {"240p", "240"}: return 240
#     if q in {"360p", "360"}: return 360
#     if q in {"480p", "480"}: return 480
#     if q in {"720p", "720"}: return 720
#     if q in {"1080p", "1080", "fullhd", "fhd"}: return 1080
#     if q in {"1440p", "1440", "2k"}: return 1440
#     if q in {"2160p", "2160", "4k", "uhd"}: return 2160
#     return 720

# def _finalize_path(output_dir: str, filename: str) -> Path:
#     base = Path(output_dir).resolve()
#     _ensure_dir(base)
#     return base / filename

# def download_audio_with_ytdlp(video_id: str, quality: str = "medium", output_dir: str = ".") -> str:
#     """
#     Download audio-only and transcode to MP3 using ffmpeg.
#     Returns the final file path.
#     """
#     if not validate_video_id(video_id):
#         raise ValueError("Invalid YouTube video ID")
#     if not check_ytdlp_availability():
#         raise RuntimeError("yt-dlp is not available")

#     bitrate = _audio_quality_to_bitrate(quality)
#     out_tmpl = _finalize_path(output_dir, f"{video_id}_audio_{quality}.%(ext)s")

#     yt_dlp = _yt_dlp()
#     ydl_opts = {
#         "quiet": True,
#         "no_warnings": True,
#         "outtmpl": str(out_tmpl),
#         "noprogress": True,
#         "format": "bestaudio/best",
#         "postprocessors": [
#             {  # extract audio
#                 "key": "FFmpegExtractAudio",
#                 "preferredcodec": "mp3",
#                 "preferredquality": "0",  # ignored for mp3; we'll set bitrate in next step
#             },
#             {  # enforce bitrate
#                 "key": "FFmpegAudioConvertor",
#                 "preferedformat": "mp3",  # yt-dlp typo kept for compat; ignored if ffmpeg present
#             },
#             {
#                 "key": "FFmpegMetadata",
#             },
#         ],
#         "postprocessor_args": [
#             "-b:a", bitrate,
#         ],
#     }

#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(_youtube_url(video_id), download=True)

#     # Resolve final .mp3 path
#     final_path = _finalize_path(output_dir, f"{video_id}_audio_{quality}.mp3")
#     if final_path.exists():
#         return str(final_path)

#     # Fallback: try to find produced mp3
#     produced = Path(output_dir).glob(f"{video_id}_audio_{quality}.*")
#     for p in produced:
#         if p.suffix.lower() == ".mp3":
#             return str(p.resolve())

#     # Last resort: infer from info
#     fname = info.get("_filename")
#     if fname:
#         p = Path(fname).with_suffix(".mp3")
#         if p.exists():
#             return str(p.resolve())

#     raise RuntimeError("Audio file was not produced")


# def download_video_with_ytdlp(video_id: str, quality: str = "720p", output_dir: str = ".") -> str:
#     """
#     Download video merged with audio to MP4 with height <= requested.
#     Returns the final file path.
#     """
#     if not validate_video_id(video_id):
#         raise ValueError("Invalid YouTube video ID")
#     if not check_ytdlp_availability():
#         raise RuntimeError("yt-dlp is not available")

#     height = _video_quality_to_height(quality)
#     out_tmpl = _finalize_path(output_dir, f"{video_id}_video_{quality}.%(ext)s")

#     # Prefer mp4-compatible formats
#     fmt = (
#         f"bestvideo[ext=mp4][height<={height}]+bestaudio[ext=m4a]/"
#         f"best[ext=mp4][height<={height}]/"
#         f"bestvideo[height<={height}]+bestaudio/best[height<={height}]"
#     )

#     yt_dlp = _yt_dlp()
#     ydl_opts = {
#         "quiet": True,
#         "no_warnings": True,
#         "outtmpl": str(out_tmpl),
#         "noprogress": True,
#         "format": fmt,
#         "merge_output_format": "mp4",
#         "postprocessors": [
#             {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
#             {"key": "FFmpegMetadata"},
#         ],
#     }

#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(_youtube_url(video_id), download=True)

#     final_path = _finalize_path(output_dir, f"{video_id}_video_{quality}.mp4")
#     if final_path.exists():
#         return str(final_path)

#     # Fallback: search for produced mp4
#     produced = Path(output_dir).glob(f"{video_id}_video_{quality}.*")
#     for p in produced:
#         if p.suffix.lower() == ".mp4":
#             return str(p.resolve())

#     # Last resort: infer from info
#     fname = info.get("_filename")
#     if fname:
#         p = Path(fname).with_suffix(".mp4")
#         if p.exists():
#             return str(p.resolve())

#     raise RuntimeError("Video file was not produced")

##########################################################
# # backend/transcript_utils.py
# """
# YouTube transcript and media download utilities using yt-dlp.
# This module provides utility functions only - no route definitions.
# """

# import os
# import re
# import logging
# import subprocess
# from pathlib import Path
# from typing import Optional, Dict, Any

# logger = logging.getLogger("youtube_trans_downloader")

# # ============================================================================
# # YT-DLP AVAILABILITY CHECK
# # ============================================================================

# def check_ytdlp_availability() -> bool:
#     """Check if yt-dlp is available and working."""
#     try:
#         result = subprocess.run(
#             ["yt-dlp", "--version"],
#             capture_output=True,
#             text=True,
#             timeout=5
#         )
#         return result.returncode == 0
#     except Exception as e:
#         logger.warning(f"yt-dlp not available: {e}")
#         return False


# # ============================================================================
# # VIDEO INFO
# # ============================================================================

# def get_video_info(video_id: str) -> Optional[Dict[str, Any]]:
#     """
#     Get video metadata using yt-dlp.
    
#     Args:
#         video_id: YouTube video ID
        
#     Returns:
#         Dictionary with title, uploader, duration, etc. or None if failed
#     """
#     try:
#         url = f"https://www.youtube.com/watch?v={video_id}"
#         result = subprocess.run(
#             [
#                 "yt-dlp",
#                 "--dump-json",
#                 "--no-playlist",
#                 url
#             ],
#             capture_output=True,
#             text=True,
#             timeout=30
#         )
        
#         if result.returncode == 0:
#             import json
#             data = json.loads(result.stdout)
#             return {
#                 "title": data.get("title", "Unknown"),
#                 "uploader": data.get("uploader", "Unknown"),
#                 "duration": data.get("duration", 0),
#                 "description": data.get("description", ""),
#                 "view_count": data.get("view_count", 0),
#                 "upload_date": data.get("upload_date", ""),
#             }
#     except Exception as e:
#         logger.warning(f"Failed to get video info for {video_id}: {e}")
    
#     return None


# # ============================================================================
# # TRANSCRIPT EXTRACTION
# # ============================================================================

# def get_transcript_with_ytdlp(
#     video_id: str, 
#     clean: bool = True
# ) -> Optional[str]:
#     """
#     Extract transcript using yt-dlp.
    
#     Args:
#         video_id: YouTube video ID
#         clean: If True, return plain text without timestamps
        
#     Returns:
#         Transcript text or None if not available
#     """
#     try:
#         url = f"https://www.youtube.com/watch?v={video_id}"
        
#         # Try to get auto-generated English subtitles
#         result = subprocess.run(
#             [
#                 "yt-dlp",
#                 "--write-auto-sub",
#                 "--sub-lang", "en",
#                 "--skip-download",
#                 "--sub-format", "vtt",
#                 "--output", f"/tmp/{video_id}.%(ext)s",
#                 url
#             ],
#             capture_output=True,
#             text=True,
#             timeout=60
#         )
        
#         # Look for the subtitle file
#         subtitle_file = Path(f"/tmp/{video_id}.en.vtt")
        
#         if not subtitle_file.exists():
#             # Try without auto-sub flag (for manual subtitles)
#             result = subprocess.run(
#                 [
#                     "yt-dlp",
#                     "--write-sub",
#                     "--sub-lang", "en",
#                     "--skip-download",
#                     "--sub-format", "vtt",
#                     "--output", f"/tmp/{video_id}.%(ext)s",
#                     url
#                 ],
#                 capture_output=True,
#                 text=True,
#                 timeout=60
#             )
        
#         if subtitle_file.exists():
#             content = subtitle_file.read_text(encoding="utf-8")
            
#             # Clean up the file
#             try:
#                 subtitle_file.unlink()
#             except Exception:
#                 pass
            
#             if clean:
#                 return _clean_vtt_content(content)
#             else:
#                 return _format_vtt_with_timestamps(content)
        
#         logger.warning(f"No subtitles found for {video_id}")
#         return None
        
#     except subprocess.TimeoutExpired:
#         logger.error(f"yt-dlp timed out for {video_id}")
#         return None
#     except Exception as e:
#         logger.error(f"yt-dlp transcript extraction failed for {video_id}: {e}")
#         return None


# def _clean_vtt_content(vtt_content: str) -> str:
#     """Extract plain text from VTT content, removing timestamps and formatting."""
#     lines = []
    
#     for line in vtt_content.split('\n'):
#         line = line.strip()
        
#         # Skip VTT headers, timestamps, and empty lines
#         if not line or line.startswith('WEBVTT') or '-->' in line or line.startswith('Kind:') or line.startswith('Language:'):
#             continue
        
#         # Skip cue identifiers (numbers)
#         if line.isdigit():
#             continue
        
#         # Remove VTT formatting tags like <c> </c>
#         line = re.sub(r'<[^>]+>', '', line)
        
#         if line:
#             lines.append(line)
    
#     # Join with spaces and clean up
#     text = ' '.join(lines)
    
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     return text


# def _format_vtt_with_timestamps(vtt_content: str) -> str:
#     """Format VTT content with readable timestamps [MM:SS]."""
#     lines = []
#     current_time = None
    
#     for line in vtt_content.split('\n'):
#         line = line.strip()
        
#         # Extract timestamp
#         if '-->' in line:
#             # Parse start time (format: 00:00:12.000)
#             start = line.split('-->')[0].strip()
#             # Convert to [MM:SS] format
#             parts = start.split(':')
#             if len(parts) >= 3:
#                 minutes = int(parts[1])
#                 seconds = int(float(parts[2].split('.')[0]))
#                 current_time = f"[{minutes:02d}:{seconds:02d}]"
        
#         # Skip headers and empty lines
#         elif not line or line.startswith('WEBVTT') or line.isdigit() or line.startswith('Kind:') or line.startswith('Language:'):
#             continue
        
#         # Text line
#         else:
#             clean_line = re.sub(r'<[^>]+>', '', line)
#             if clean_line and current_time:
#                 lines.append(f"{current_time} {clean_line}")
    
#     return '\n'.join(lines)


# # ============================================================================
# # AUDIO DOWNLOAD
# # ============================================================================

# def download_audio_with_ytdlp(
#     video_id: str,
#     quality: str = "medium",
#     output_dir: str = "./downloads"
# ) -> Optional[str]:
#     """
#     Download audio from YouTube video.
    
#     Args:
#         video_id: YouTube video ID
#         quality: Audio quality (low, medium, high, best)
#         output_dir: Directory to save the file
        
#     Returns:
#         Path to downloaded file or None if failed
#     """
#     try:
#         os.makedirs(output_dir, exist_ok=True)
        
#         url = f"https://www.youtube.com/watch?v={video_id}"
#         output_template = os.path.join(output_dir, f"{video_id}_audio_%(quality)s.%(ext)s")
        
#         # Map quality to yt-dlp format
#         quality_map = {
#             "low": "worst",
#             "medium": "192",
#             "high": "256", 
#             "best": "best"
#         }
#         audio_quality = quality_map.get(quality, "192")
        
#         cmd = [
#             "yt-dlp",
#             "-f", "bestaudio",
#             "--extract-audio",
#             "--audio-format", "mp3",
#             "--audio-quality", audio_quality,
#             "--output", output_template,
#             "--no-playlist",
#             url
#         ]
        
#         logger.info(f"Downloading audio for {video_id} with quality={quality}")
        
#         result = subprocess.run(
#             cmd,
#             capture_output=True,
#             text=True,
#             timeout=300  # 5 minutes
#         )
        
#         if result.returncode != 0:
#             logger.error(f"yt-dlp audio download failed: {result.stderr}")
#             raise Exception(f"Download failed: {result.stderr[:200]}")
        
#         # Find the downloaded file
#         for file in Path(output_dir).glob(f"{video_id}_audio_*"):
#             if file.is_file():
#                 logger.info(f"✅ Audio downloaded: {file}")
#                 return str(file)
        
#         raise Exception("Audio file not found after download")
        
#     except subprocess.TimeoutExpired:
#         logger.error(f"Audio download timed out for {video_id}")
#         raise Exception("Download timed out")
#     except Exception as e:
#         logger.error(f"Audio download failed for {video_id}: {e}")
#         raise


# # ============================================================================
# # VIDEO DOWNLOAD
# # ============================================================================

# def download_video_with_ytdlp(
#     video_id: str,
#     quality: str = "720p",
#     output_dir: str = "./downloads"
# ) -> Optional[str]:
#     """
#     Download video from YouTube.
    
#     Args:
#         video_id: YouTube video ID
#         quality: Video quality (360p, 480p, 720p, 1080p, best)
#         output_dir: Directory to save the file
        
#     Returns:
#         Path to downloaded file or None if failed
#     """
#     try:
#         os.makedirs(output_dir, exist_ok=True)
        
#         url = f"https://www.youtube.com/watch?v={video_id}"
#         output_template = os.path.join(output_dir, f"{video_id}_video_%(quality)s.%(ext)s")
        
#         # Map quality to yt-dlp format
#         quality_map = {
#             "360p": "bestvideo[height<=360]+bestaudio/best[height<=360]",
#             "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
#             "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
#             "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
#             "best": "bestvideo+bestaudio/best"
#         }
#         format_string = quality_map.get(quality, quality_map["720p"])
        
#         cmd = [
#             "yt-dlp",
#             "-f", format_string,
#             "--merge-output-format", "mp4",
#             "--output", output_template,
#             "--no-playlist",
#             url
#         ]
        
#         logger.info(f"Downloading video for {video_id} with quality={quality}")
        
#         result = subprocess.run(
#             cmd,
#             capture_output=True,
#             text=True,
#             timeout=600  # 10 minutes
#         )
        
#         if result.returncode != 0:
#             logger.error(f"yt-dlp video download failed: {result.stderr}")
#             raise Exception(f"Download failed: {result.stderr[:200]}")
        
#         # Find the downloaded file
#         for file in Path(output_dir).glob(f"{video_id}_video_*"):
#             if file.is_file():
#                 logger.info(f"✅ Video downloaded: {file}")
#                 return str(file)
        
#         raise Exception("Video file not found after download")
        
#     except subprocess.TimeoutExpired:
#         logger.error(f"Video download timed out for {video_id}")
#         raise Exception("Download timed out")
#     except Exception as e:
#         logger.error(f"Video download failed for {video_id}: {e}")
#         raise


# # ============================================================================
# # VALIDATION
# # ============================================================================

# def validate_video_id(video_id: str) -> bool:
#     """Validate YouTube video ID format."""
#     if not video_id or len(video_id) != 11:
#         return False
#     return bool(re.match(r'^[A-Za-z0-9_-]{11}$', video_id))

# #===========================================================================
# # ============================================================================
# # FIXED TRANSCRIPT ENDPOINT FOR main.py
# # ============================================================================
# # Replace your current /download_transcript/ endpoint with this code
# # 
# # Key fixes:
# # 1. Properly handles FetchedTranscriptSnippet objects (no .get() method)
# # 2. Uses yt-dlp as primary method (from transcript_utils.py)
# # 3. Falls back to youtube-transcript-api if yt-dlp fails
# # 4. Handles all formats: clean text, SRT, VTT
# # ============================================================================

# from fastapi import APIRouter, HTTPException, Depends
# from fastapi.responses import Response
# from pydantic import BaseModel
# from typing import Optional
# import logging

# # Your existing imports
# #from transcript_utils import get_transcript_with_ytdlp, validate_video_id


# logger = logging.getLogger("youtube_trans_downloader")

# # Request model
# class TranscriptRequest(BaseModel):
#     video_id: str
#     clean: bool = True
#     format: Optional[str] = None  # 'srt', 'vtt', or None for plain text

# # ============================================================================
# # FIXED ENDPOINT
# # ============================================================================

# @router.post("/download_transcript/")
# async def download_transcript(
#     request: TranscriptRequest,
#     # Add your auth dependencies here
#     # current_user: User = Depends(get_current_user)
# ):
#     """
#     Download transcript with proper error handling.
    
#     Formats:
#     - clean=True, format=None -> Plain text (no timestamps)
#     - clean=False, format='srt' -> SRT format with timestamps
#     - clean=False, format='vtt' -> VTT format with timestamps
#     """
#     video_id = request.video_id.strip()
    
#     # Validate video ID
#     if not validate_video_id(video_id):
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID")
    
#     logger.info(f"Transcript for {video_id} (clean={request.clean}, fmt={request.format})")
    
#     try:
#         # ====================================================================
#         # PRIMARY METHOD: Use yt-dlp (from transcript_utils.py)
#         # ====================================================================
#         transcript_text = get_transcript_with_ytdlp(
#             video_id=video_id, 
#             clean=request.clean
#         )
        
#         if transcript_text:
#             # Format based on user request
#             if request.format == 'srt' and not request.clean:
#                 content = _convert_to_srt(transcript_text, video_id)
#                 media_type = "text/plain"
#                 filename = f"{video_id}_transcript.srt"
#             elif request.format == 'vtt' and not request.clean:
#                 content = _convert_to_vtt(transcript_text, video_id)
#                 media_type = "text/vtt"
#                 filename = f"{video_id}_transcript.vtt"
#             else:
#                 # Plain text (clean format)
#                 content = transcript_text
#                 media_type = "text/plain"
#                 filename = f"{video_id}_transcript.txt"
            
#             logger.info(f"✅ Transcript ready: {filename}")
            
#             return Response(
#                 content=content,
#                 media_type=media_type,
#                 headers={
#                     "Content-Disposition": f'attachment; filename="{filename}"'
#                 }
#             )
        
#         # ====================================================================
#         # FALLBACK METHOD: Use youtube-transcript-api
#         # ====================================================================
#         logger.warning(f"yt-dlp failed, trying fallback method for {video_id}")
#         transcript_text = _get_transcript_fallback(video_id, request.clean, request.format)
        
#         if transcript_text:
#             if request.format == 'srt':
#                 media_type = "text/plain"
#                 filename = f"{video_id}_transcript.srt"
#             elif request.format == 'vtt':
#                 media_type = "text/vtt"
#                 filename = f"{video_id}_transcript.vtt"
#             else:
#                 media_type = "text/plain"
#                 filename = f"{video_id}_transcript.txt"
            
#             return Response(
#                 content=transcript_text,
#                 media_type=media_type,
#                 headers={
#                     "Content-Disposition": f'attachment; filename="{filename}"'
#                 }
#             )
        
#         # No transcript found by any method
#         raise HTTPException(
#             status_code=404, 
#             detail="No transcript/captions found for this video"
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Transcript pipeline error for {video_id}: {e}")
#         raise HTTPException(
#             status_code=404,
#             detail=f"Could not retrieve transcript: {str(e)}"
#         )


# # ============================================================================
# # HELPER FUNCTIONS
# # ============================================================================

# def _convert_to_srt(text: str, video_id: str) -> str:
#     """
#     Convert plain text to SRT format with estimated timestamps.
#     If text already has timestamps like [00:12], preserve them.
#     """
#     lines = text.strip().split('\n')
#     srt_content = []
#     index = 1
    
#     for i, line in enumerate(lines):
#         line = line.strip()
#         if not line:
#             continue
        
#         # Estimate timing (5 seconds per line)
#         start_sec = i * 5
#         end_sec = start_sec + 5
        
#         start_time = _format_srt_timestamp(start_sec)
#         end_time = _format_srt_timestamp(end_sec)
        
#         srt_content.append(f"{index}")
#         srt_content.append(f"{start_time} --> {end_time}")
#         srt_content.append(line)
#         srt_content.append("")  # Blank line
#         index += 1
    
#     return "\n".join(srt_content)


# def _convert_to_vtt(text: str, video_id: str) -> str:
#     """Convert plain text to WebVTT format."""
#     vtt_content = ["WEBVTT", ""]
    
#     lines = text.strip().split('\n')
#     for i, line in enumerate(lines):
#         line = line.strip()
#         if not line:
#             continue
        
#         start_sec = i * 5
#         end_sec = start_sec + 5
        
#         start_time = _format_vtt_timestamp(start_sec)
#         end_time = _format_vtt_timestamp(end_sec)
        
#         vtt_content.append(f"{start_time} --> {end_time}")
#         vtt_content.append(line)
#         vtt_content.append("")
    
#     return "\n".join(vtt_content)


# def _format_srt_timestamp(seconds: int) -> str:
#     """Format seconds as SRT timestamp: 00:00:12,000"""
#     hours = seconds // 3600
#     minutes = (seconds % 3600) // 60
#     secs = seconds % 60
#     return f"{hours:02d}:{minutes:02d}:{secs:02d},000"


# def _format_vtt_timestamp(seconds: int) -> str:
#     """Format seconds as WebVTT timestamp: 00:00:12.000"""
#     hours = seconds // 3600
#     minutes = (seconds % 3600) // 60
#     secs = seconds % 60
#     return f"{hours:02d}:{minutes:02d}:{secs:02d}.000"


# def _get_transcript_fallback(video_id: str, clean: bool, fmt: Optional[str]) -> Optional[str]:
#     """
#     Fallback method using youtube-transcript-api.
    
#     FIXED: Properly handles FetchedTranscriptSnippet objects.
#     """
#     try:
#         from youtube_transcript_api import YouTubeTranscriptApi
        
#         # Get transcript
#         transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
#         if not transcript_list:
#             return None
        
#         # ================================================================
#         # FIX: Access snippet attributes correctly
#         # ================================================================
#         if clean:
#             # Clean format: just text, no timestamps
#             texts = []
#             for snippet in transcript_list:
#                 # ✅ FIXED: Use dictionary access or attribute access
#                 # Option 1: Dictionary-style
#                 if isinstance(snippet, dict):
#                     text = snippet.get('text', '')
#                 else:
#                     # Option 2: Attribute-style (for FetchedTranscriptSnippet)
#                     text = getattr(snippet, 'text', '')
                
#                 if text:
#                     texts.append(text.strip())
            
#             return " ".join(texts)
        
#         elif fmt == 'srt':
#             # SRT format with timestamps
#             srt_lines = []
#             for i, snippet in enumerate(transcript_list, 1):
#                 # ✅ FIXED: Proper attribute access
#                 if isinstance(snippet, dict):
#                     text = snippet.get('text', '')
#                     start = snippet.get('start', 0)
#                     duration = snippet.get('duration', 3)
#                 else:
#                     text = getattr(snippet, 'text', '')
#                     start = getattr(snippet, 'start', 0)
#                     duration = getattr(snippet, 'duration', 3)
                
#                 if not text:
#                     continue
                
#                 end = start + duration
#                 start_ts = _format_srt_timestamp(int(start))
#                 end_ts = _format_srt_timestamp(int(end))
                
#                 srt_lines.append(f"{i}")
#                 srt_lines.append(f"{start_ts} --> {end_ts}")
#                 srt_lines.append(text.strip())
#                 srt_lines.append("")
            
#             return "\n".join(srt_lines)
        
#         elif fmt == 'vtt':
#             # WebVTT format
#             vtt_lines = ["WEBVTT", ""]
            
#             for snippet in transcript_list:
#                 # ✅ FIXED: Proper attribute access
#                 if isinstance(snippet, dict):
#                     text = snippet.get('text', '')
#                     start = snippet.get('start', 0)
#                     duration = snippet.get('duration', 3)
#                 else:
#                     text = getattr(snippet, 'text', '')
#                     start = getattr(snippet, 'start', 0)
#                     duration = getattr(snippet, 'duration', 3)
                
#                 if not text:
#                     continue
                
#                 end = start + duration
#                 start_ts = _format_vtt_timestamp(int(start))
#                 end_ts = _format_vtt_timestamp(int(end))
                
#                 vtt_lines.append(f"{start_ts} --> {end_ts}")
#                 vtt_lines.append(text.strip())
#                 vtt_lines.append("")
            
#             return "\n".join(vtt_lines)
        
#         return None
        
#     except Exception as e:
#         logger.warning(f"Fallback transcript method failed: {e}")
#         return None

