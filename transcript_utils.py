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

logger = logging.getLogger("youtube_trans_downloader")

def _common_ydl_opts(tmp_dir: str | None = None) -> dict:
    """Base yt-dlp options shared by audio/video flows."""
    prefer_ipv4 = str(os.getenv("YTDLP_BIND_IPV4", "1")).lower() in ("1", "true", "yes")
    headers = {
        # Keep a modern UA; some CDNs 403 on unknown agents
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    opts = {
        "quiet": True,
        "noprogress": True,
        "concurrent_fragment_downloads": 4,
        "retries": 3,
        "fragment_retries": 10,
        "retry_sleep": "1,2,4,8",
        "http_headers": headers,
        "nocheckcertificate": True,
        # IPv4 preference to dodge some ISP/IPv6 routing/CDN blocks
        "prefer_ipv6": False if prefer_ipv4 else None,
        "source_address": "0.0.0.0" if prefer_ipv4 else None,
        # Temp/work dir
        "paths": {"home": tmp_dir} if tmp_dir else None,
    }
    # Remove None values to keep yt-dlp happy
    return {k: v for k, v in opts.items() if v is not None}



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


# def download_audio_with_ytdlp(video_id: str, quality: str = "medium", output_dir: str = ".") -> str:
#     """
#     Download audio and transcode to **MP3** using a single, stable post-processor.
#     Returns the final .mp3 path.
#     """
#     if not validate_video_id(video_id):
#         raise ValueError("Invalid YouTube video ID")
#     if not check_ytdlp_availability():
#         raise RuntimeError("yt-dlp is not available")

#     kbps = _audio_quality_to_kbps(quality)
#     # Deterministic template; ExtractAudio will replace ext to .mp3
#     out_tmpl = _finalize_path(output_dir, f"{video_id}_audio_{quality}.%(ext)s")

#     yt_dlp = _yt_dlp()
#     ydl_opts: Dict[str, Any] = {
#         "quiet": True,
#         "no_warnings": True,
#         "noprogress": True,
#         "concurrent_fragment_downloads": 1,  # reduces flakiness on Windows
#         "outtmpl": str(out_tmpl),
#         "format": "bestaudio/best",
#         "prefer_ffmpeg": True,
#         "postprocessors": [
#             {
#                 # ✅ stable, supported PP – handles extraction & mp3 encode
#                 "key": "FFmpegExtractAudio",
#                 "preferredcodec": "mp3",
#                 "preferredquality": kbps,  # kbps as string, e.g. "160"
#             },
#             {"key": "FFmpegMetadata"},
#         ],
#         **_ffmpeg_location_opts(),
#     }

#     # Execute
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(_youtube_url(video_id), download=True)

#     # Resolve expected final path
#     final_path = _finalize_path(output_dir, f"{video_id}_audio_{quality}.mp3")
#     if final_path.exists() and final_path.stat().st_size > 0:
#         return str(final_path)

#     # Fallback: search produced files
#     for p in Path(output_dir).glob(f"{video_id}_audio_{quality}*.mp3"):
#         if p.is_file() and p.stat().st_size > 0:
#             return str(p.resolve())

#     # Last resort: infer from yt-dlp filename hint
#     fname = info.get("_filename")
#     if fname:
#         p = Path(fname).with_suffix(".mp3")
#         if p.exists() and p.stat().st_size > 0:
#             return str(p.resolve())

#     raise RuntimeError("Audio file was not produced")

def download_audio_with_ytdlp(video_id: str, quality: str, output_dir: str) -> str:
    """
    Extracts audio as MP3 using FFmpegExtractAudio (the robust, standard PP).
    """
    abr_map = {"low": "64", "medium": "128", "high": "192"}
    abr = abr_map.get(quality.lower(), "128")
    base_opts = _common_ydl_opts(tmp_dir=output_dir)

    ydl_opts = {
        **base_opts,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, f"{video_id}_audio_%(abr)sk.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": abr,
            }
        ],
        # Better container probing
        "postprocessor_args": ["-vn"],
    }

    url = f"https://www.youtube.com/watch?v={video_id}"
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # yt-dlp returns output paths in info when postprocessors run
        # We normalize to our deterministic file name:
        path = os.path.join(output_dir, f"{video_id}_audio_{abr}k.mp3")
        if not os.path.exists(path):
            # fallback: find the first created file
            candidates = [f for f in os.listdir(output_dir) if f.startswith(f"{video_id}_audio_") and f.endswith(".mp3")]
            if candidates:
                path = os.path.join(output_dir, candidates[0])
        return path


# def download_video_with_ytdlp(video_id: str, quality: str = "720p", output_dir: str = ".") -> str:
#     """
#     Download video merged with audio to MP4 with height <= requested.
#     Strongly prefers non-HLS to avoid empty-file + 403 fragment issues.
#     Falls back through safer formats if the first choice is blocked.
#     """
#     if not validate_video_id(video_id):
#         raise ValueError("Invalid YouTube video ID")
#     if not check_ytdlp_availability():
#         raise RuntimeError("yt-dlp is not available")

#     yt_dlp = _yt_dlp()
#     height = _video_quality_to_height(quality)

#     # Prefer AVC MP4 video + M4A audio, explicitly avoid HLS when possible.
#     # Fallbacks: best MP4, then non-HLS best, then anything that works.
#     fmt = (
#         f"(bestvideo[ext=mp4][vcodec^=avc1][protocol!=m3u8][height<={height}]"
#         f"/bestvideo[ext=mp4][protocol!=m3u8][height<={height}]"
#         f"/bestvideo[protocol!=m3u8][height<={height}])"
#         f"+(bestaudio[ext=m4a]/bestaudio)"
#         f"/best[ext=mp4][protocol!=m3u8][height<={height}]"
#         f"/best[protocol!=m3u8][height<={height}]"
#         f"/best[height<={height}]"
#     )

#     out_tmpl = _finalize_path(output_dir, f"{video_id}_video_{quality}.%(ext)s")

#     # Optional cookies from env (one of these; both are optional)
#     cookies_from_browser = os.getenv("COOKIES_FROM_BROWSER")  # e.g., "chrome"
#     cookies_file = os.getenv("COOKIES_FILE")  # e.g., "cookies.txt"

#     # Desktop UA + robust retries/backoff
#     http_headers = {
#         "User-Agent": (
#             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
#             "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
#         )
#     }

#     base_opts = {
#         "quiet": True,
#         "no_warnings": True,
#         "noprogress": True,
#         "outtmpl": str(out_tmpl),
#         "noplaylist": True,
#         "format": fmt,
#         "merge_output_format": "mp4",
#         "prefer_ffmpeg": True,
#         "http_headers": http_headers,
#         "retries": 10,
#         "fragment_retries": 10,
#         "retry_sleep_functions": {
#             "http": "exponential_backoff",
#             "fragment": "exponential_backoff",
#         },
#         "concurrent_fragment_downloads": 1,  # prevents Windows file handle issues
#         "skip_unavailable_fragments": True,
#         "geo_bypass": True,
#         "geo_bypass_country": "US",
#         # Force a stable YouTube client to reduce 403s due to odd signatures:
#         "extractor_args": {
#             "youtube": {
#                 "player_client": ["web"],
#                 # Keep DASH manifest; we already exclude HLS by format filter.
#                 "include_hls_manifest": ["no"],
#             }
#         },
#         **_ffmpeg_location_opts(),
#     }

#     if cookies_from_browser:
#         base_opts["cookiesfrombrowser"] = (cookies_from_browser,)
#     if cookies_file and Path(cookies_file).exists():
#         base_opts["cookiefile"] = cookies_file

#     # Try primary attempt; if it fails with 403, fallback to a very safe format.
#     try:
#         with yt_dlp.YoutubeDL(base_opts) as ydl:
#             info = ydl.extract_info(_youtube_url(video_id), download=True)
#     except Exception as e:
#         # Fallback to progressive MP4 only (no mux), excluding HLS entirely.
#         fallback_fmt = (
#             f"best[ext=mp4][protocol!=m3u8][protocol!=dash][height<={height}]"
#             f"/best[ext=mp4][protocol!=m3u8][height<={height}]"
#             f"/best[ext=mp4][height<={height}]"
#         )
#         fb_opts = dict(base_opts)
#         fb_opts["format"] = fallback_fmt
#         fb_opts["postprocessors"] = [
#             {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
#             {"key": "FFmpegMetadata"},
#         ]
#         with yt_dlp.YoutubeDL(fb_opts) as ydl:
#             info = ydl.extract_info(_youtube_url(video_id), download=True)

#     final_path = _finalize_path(output_dir, f"{video_id}_video_{quality}.mp4")
#     if final_path.exists() and final_path.stat().st_size > 0:
#         return str(final_path)

#     # Resolve alternative names produced by yt-dlp
#     for p in Path(output_dir).glob(f"{video_id}_video_{quality}*.mp4"):
#         if p.is_file() and p.stat().st_size > 0:
#             return str(p.resolve())

#     fname = info.get("_filename")
#     if fname:
#         p = Path(fname)
#         if p.suffix.lower() != ".mp4":
#             p = p.with_suffix(".mp4")
#         if p.exists() and p.stat().st_size > 0:
#             return str(p.resolve())

#     raise RuntimeError("Video file was not produced")


def download_video_with_ytdlp(video_id: str, quality: str, output_dir: str) -> str:
    """
    Downloads MP4 video at the requested quality (1080p/720p/480p/360p).
    """
    fmt_map = {
        "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "720p":  "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "480p":  "bestvideo[height<=480]+bestaudio/best[height<=480]",
        "360p":  "bestvideo[height<=360]+bestaudio/best[height<=360]",
    }
    fmt = fmt_map.get(quality.lower(), fmt_map["720p"])

    base_opts = _common_ydl_opts(tmp_dir=output_dir)
    ydl_opts = {
        **base_opts,
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(output_dir, f"{video_id}_video_{quality.lower()}.%(ext)s"),
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
    }

    url = f"https://www.youtube.com/watch?v={video_id}"
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = os.path.join(output_dir, f"{video_id}_video_{quality.lower()}.mp4")
        return path

