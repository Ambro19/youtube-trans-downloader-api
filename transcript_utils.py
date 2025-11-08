# transcript_utils.py — PRODUCTION READY
"""
Utilities for transcripts and media downloads.
Framework-free to avoid circular imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import logging, os, re, tempfile, json

from yt_dlp import YoutubeDL

logger = logging.getLogger("youtube_trans_downloader")

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

def _parse_vtt_to_segments(vtt_text: str) -> List[Tuple[float, float, str]]:
    """
    Very small VTT parser: returns list of (start, end, text).
    Expects '00:00:01.000 --> 00:00:02.000' lines; joins following text lines.
    """
    lines = [ln.strip("\ufeff").strip() for ln in (vtt_text or "").splitlines()]
    segments: List[Tuple[float, float, str]] = []
    i = 0
    time_re = re.compile(r"^(\d\d:\d\d:\d\d\.\d{3})\s*-->\s*(\d\d:\d\d:\d\d\.\d{3})")
    def _to_sec(ts: str) -> float:
        h, m, rest = ts.split(":")
        s, ms = rest.split(".")
        return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0

    while i < len(lines):
        m = time_re.match(lines[i])
        if not m:
            i += 1
            continue
        start, end = _to_sec(m.group(1)), _to_sec(m.group(2))
        i += 1
        text_lines = []
        while i < len(lines) and lines[i] and not time_re.match(lines[i]):
            # Skip cue settings like "align:start position:0%" etc
            if "-->" in lines[i]:
                break
            text_lines.append(lines[i])
            i += 1
        text = " ".join(t for t in text_lines).strip()
        if text:
            segments.append((start, end, text))
        # skip blank separation if present
        while i < len(lines) and not lines[i]:
            i += 1
    return segments

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
def get_transcript_with_ytdlp(video_id: str, clean: bool = True, fmt: Optional[str] = None) -> Optional[str]:
    """
    Fetch subtitles with yt-dlp.
    - fmt in {"srt","vtt"} => return file content in that format.
    - fmt None and clean==True => return plain paragraphs (no timestamps).
    - fmt None and clean==False => return [mm:ss] timestamped lines.
    """
    url = _norm_youtube_url(video_id)
    if not check_ytdlp_availability():
        return None

    ffmpeg_loc = _ensure_ffmpeg_location()

    # We’ll always try to *write* subs (auto or authored). Prefer English variants.
    sub_langs = ["en", "en-US", "en-GB", "en-CA", "en-AU", "en-IE", "en-NZ"]

    with tempfile.TemporaryDirectory() as td:
        # If a concrete subtitle format was requested, ask yt-dlp for that directly.
        want_fmt = (fmt or "").lower()
        sub_fmt = want_fmt if want_fmt in {"srt", "vtt"} else "vtt"

        ydl_opts: Dict[str, Any] = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": sub_fmt,
            "subtitleslangs": sub_langs,
            "outtmpl": os.path.join(td, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }
        if ffmpeg_loc:
            ydl_opts["ffmpeg_location"] = ffmpeg_loc

        # Download captions
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            logger.debug("yt-dlp subtitle download failed: %s", e)
            return None

        # Find resulting file(s). yt-dlp may emit e.g. <id>.<lang>.<ext> or just <id>.<ext>
        p = Path(td)
        candidates = sorted(list(p.glob(f"{video_id}*.{sub_fmt}")))
        if not candidates:
            # Fallback: any VTT/SRT for this temp dir (some extractors name with title-id)
            candidates = sorted(list(p.glob(f"*.{sub_fmt}")))

        if not candidates:
            logger.debug("yt-dlp: no subtitle files found for %s", video_id)
            return None

        text = candidates[0].read_text(encoding="utf-8", errors="replace")

        # If user explicitly requested SRT/VTT, return verbatim content
        if want_fmt in {"srt", "vtt"}:
            return text

        # Otherwise we return TXT (clean or timestamped) by parsing VTT
        segs = _parse_vtt_to_segments(text)
        if not segs:
            return None

        if clean:
            # Plain paragraphs
            return _clean_plain_blocks([t for _, _, t in segs])

        # Timestamped [mm:ss] text
        lines = [f"[{_mmss(start)}] {t.replace(chr(10), ' ')}" for start, _, t in segs if t]
        return "\n".join(lines)

# -----------------------
# Video / Audio (unchanged)
# -----------------------
def _common_ydl_opts(output_dir: str) -> Dict:
    ffmpeg_loc = _ensure_ffmpeg_location()
    opts = {
        "extractor_args": {"youtube": {"player_client": ["android", "android_embedded"]}},
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
        "http_headers": {"User-Agent": "com.google.android.youtube/19.20.34 (Linux; U; Android 11)",
                         "Accept-Language": "en-US,en;q=0.9"},
    }
    if ffmpeg_loc:
        opts["ffmpeg_location"] = ffmpeg_loc
    return opts

def get_video_info(video_id: str) -> Dict[str, Any]:
    import yt_dlp
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(_norm_youtube_url(video_id), download=False)
        return {"id": info.get("id"),
                "title": info.get("title"),
                "uploader": info.get("uploader") or info.get("channel"),
                "duration": info.get("duration")}

def download_audio_with_ytdlp(video_id_or_url: str, quality: str, output_dir: str) -> str:
    from yt_dlp.utils import DownloadError
    kbps = "96" if (quality or "").lower() in {"low", "l"} else "256" if (quality or "").lower() in {"high", "h"} else "160"
    opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": kbps},
                           {"key": "FFmpegMetadata"},
                           {"key": "EmbedThumbnail", "already_have_thumbnail": False}],
        "outtmpl": _safe_outtmpl(output_dir),
        "noprogress": True, "quiet": True, "writethumbnail": True,
        "extractor_args": {"youtube": {"player_client": ["android", "android_embedded"]}},
        "http_headers": {"User-Agent": "com.google.android.youtube/19.20.34 (Linux; U; Android 11)",
                         "Accept-Language": "en-US,en;q=0.9"},
    }
    ffmpeg_loc = _ensure_ffmpeg_location()
    if ffmpeg_loc: opts["ffmpeg_location"] = ffmpeg_loc
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


########################################################################
# # backend/transcript_utils.py
# """
# Utilities for transcripts and media downloads.
# Framework-free to avoid circular imports.
# """

# from __future__ import annotations

# from pathlib import Path
# from typing import Any, Dict, Optional, List
# import logging
# import os
# import re
# from yt_dlp import YoutubeDL
# from yt_dlp.utils import DownloadError

# logger = logging.getLogger("youtube_trans_downloader")


# # -----------------------
# # Helper functions
# # -----------------------
# def _mp4_merge_fmt() -> str:
#     """Prefer mp4 video + m4a audio; fall back to best single file."""
#     return "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"


# def _safe_outtmpl(output_dir: str, stem: str = "%(title).200B [%(id)s]", ext_placeholder: str = "%(ext)s") -> Dict[str, str]:
#     """Windows-safe filenames + keep proper title."""
#     return {
#         "default": os.path.join(
#             output_dir,
#             f"{stem}.{ext_placeholder}"
#         )
#     }


# def _norm_youtube_url(video_id_or_url: str) -> str:
#     """Accept ID or any YT URL (watch/shorts/youtu.be)."""
#     s = video_id_or_url.strip()
#     m = re.search(r'(?:v=|/shorts/|youtu\.be/)([A-Za-z0-9_-]{6,})', s)
#     vid = m.group(1) if m else s
#     return f'https://www.youtube.com/watch?v={vid}'


# def _ensure_ffmpeg_location() -> Optional[str]:
#     """
#     If you ship FFmpeg with the app, return its folder here and set in opts.
#     Otherwise return None and make sure FFmpeg is on PATH.
#     """
#     ff = os.getenv("FFMPEG_PATH")
#     if ff and Path(ff).exists():
#         return ff
#     return None


# def _common_ydl_opts(output_dir: str) -> Dict:
#     """Common yt-dlp options for downloads."""
#     ffmpeg_loc = _ensure_ffmpeg_location()
#     opts = {
#         # Avoid the problematic web client; prefer Android to bypass SABR
#         'extractor_args': {
#             'youtube': {
#                 'player_client': ['android', 'android_embedded']
#             }
#         },
#         # Pick AVC video first (widely compatible), then best available; always include audio
#         'format': (
#             # mp4/avc first, merged with best audio
#             'bv*[ext=mp4][vcodec^=avc1]+ba[ext=m4a]/'
#             # any bestvideo + bestaudio fallback
#             'bv*+ba/best'
#         ),
#         # Merge/remux to MP4 even if the best stream is webm
#         'merge_output_format': 'mp4',
#         'postprocessors': [
#             {'key': 'FFmpegVideoRemuxer', 'preferedformat': 'mp4'},
#             {'key': 'FFmpegMetadata'},
#             {'key': 'EmbedThumbnail', 'already_have_thumbnail': False},
#         ],
#         'writethumbnail': True,
#         'outtmpl': _safe_outtmpl(output_dir),
#         'noprogress': True,
#         'quiet': True,
#         'concurrent_fragment_downloads': 4,
#         'retries': 5,
#         'fragment_retries': 5,
#         # Slightly "Android-ish" headers help in some edge cases
#         'http_headers': {
#             'User-Agent': 'com.google.android.youtube/19.20.34 (Linux; U; Android 11)',
#             'Accept-Language': 'en-US,en;q=0.9',
#         },
#     }
    
#     if ffmpeg_loc:
#         opts['ffmpeg_location'] = ffmpeg_loc
    
#     return opts


# # -----------------------
# # ID / validation helpers
# # -----------------------
# _YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


# def validate_video_id(video_id: str) -> bool:
#     return bool(_YT_ID_RE.fullmatch((video_id or "").strip()))


# # -----------------------
# # yt-dlp / ffmpeg helpers
# # -----------------------
# def check_ytdlp_availability() -> bool:
#     try:
#         import yt_dlp  # noqa: F401
#         return True
#     except Exception as e:
#         logger.debug("yt-dlp not available: %s", e)
#         return False


# def _yt_dlp():
#     import yt_dlp  # imported lazily
#     return yt_dlp


# def _youtube_url(video_id: str) -> str:
#     return f"https://www.youtube.com/watch?v={video_id}"


# def _ffmpeg_location_opts() -> Dict[str, Any]:
#     """
#     Honor FFMPEG_PATH env var on Windows if user installed a local binary.
#     """
#     ff = os.getenv("FFMPEG_PATH")
#     if ff and Path(ff).exists():
#         return {"ffmpeg_location": ff}
#     return {}


# # -----------------------
# # Video info via yt-dlp
# # -----------------------
# def get_video_info(video_id: str) -> Dict[str, Any]:
#     if not validate_video_id(video_id):
#         raise ValueError("Invalid YouTube video ID")

#     yt_dlp = _yt_dlp()
#     ydl_opts = {
#         "quiet": True,
#         "no_warnings": True,
#         "skip_download": True,
#         **_ffmpeg_location_opts(),
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
#     """Format plain text into readable paragraphs."""
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

# def get_transcript_with_ytdlp(video_id: str, clean: bool = True, fmt: str = None) -> str:
#     """
#     Get transcript using yt-dlp as fallback.
#     Now supports SRT and VTT format extraction!
#     """
#     try:
#         import yt_dlp
#         import tempfile
#         import os
#         from pathlib import Path
#         import logging
        
#         logger = logging.getLogger("youtube_trans_downloader")
        
#         # If SRT or VTT format requested, extract subtitles directly
#         if fmt in ('srt', 'vtt'):
#             with tempfile.TemporaryDirectory() as temp_dir:
#                 output_template = os.path.join(temp_dir, '%(id)s.%(ext)s')
                
#                 ydl_opts = {
#                     'skip_download': True,
#                     'writesubtitles': True,
#                     'writeautomaticsub': True,
#                     'subtitlesformat': fmt,  # Request SRT or VTT
#                     'subtitleslangs': ['en', 'en-US', 'en-GB', 'en-CA'],
#                     'outtmpl': output_template,
#                     'quiet': True,
#                     'no_warnings': True,
#                 }
                
#                 try:
#                     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                         ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
                    
#                     # Find the subtitle file (try multiple patterns)
#                     subtitle_files = list(Path(temp_dir).glob(f'{video_id}*.{fmt}'))
                    
#                     if subtitle_files:
#                         with open(subtitle_files[0], 'r', encoding='utf-8') as f:
#                             content = f.read()
#                         logger.info(f"✅ yt-dlp extracted {fmt.upper()} subtitles for {video_id}")
#                         return content
#                     else:
#                         logger.warning(f"yt-dlp: No {fmt} subtitles found for {video_id}")
#                         return None
                        
#                 except Exception as e:
#                     logger.debug(f"yt-dlp subtitle extraction failed: {e}")
#                     return None
        
#         # ... keep the rest of your existing plain text extraction code ...
        
#     except Exception as e:
#         logger.error(f"yt-dlp fallback failed for {video_id}: {e}")
#         return None
        
# # def get_transcript_with_ytdlp(video_id: str, clean: bool = True) -> Optional[str]:
# #     """
# #     Best-effort transcript retrieval returning **plain text**.
# #     Uses youtube_transcript_api; kept here so main can call a single place.
# #     """
# #     try:
# #         from youtube_transcript_api import (
# #             YouTubeTranscriptApi,
# #             NoTranscriptFound,
# #             TranscriptsDisabled,
# #         )
# #     except Exception as e:
# #         logger.warning("youtube_transcript_api not installed: %s", e)
# #         return None

# #     try:
# #         listing = YouTubeTranscriptApi.list_transcripts(video_id)

# #         # 1) Authored English variants
# #         for code in _EN_PRIORITY:
# #             try:
# #                 t = listing.find_transcript([code])
# #                 segs = t.fetch()
# #                 texts = [
# #                     _get_segment_value(s, "text", "").replace("\n", " ")
# #                     for s in segs
# #                     if _get_segment_value(s, "text", "")
# #                 ]
# #                 return _clean_plain_blocks(texts) if clean else "\n".join(texts)
# #             except NoTranscriptFound:
# #                 continue
# #             except Exception:
# #                 continue

# #         # 2) Generated English
# #         try:
# #             t = listing.find_generated_transcript(_EN_PRIORITY)
# #             segs = t.fetch()
# #             texts = [
# #                 _get_segment_value(s, "text", "").replace("\n", " ")
# #                 for s in segs
# #                 if _get_segment_value(s, "text", "")
# #             ]
# #             return _clean_plain_blocks(texts) if clean else "\n".join(texts)
# #         except NoTranscriptFound:
# #             pass
# #         except Exception:
# #             pass

# #         # 3) Translate to en
# #         for t in listing:
# #             try:
# #                 segs = t.translate("en").fetch()
# #                 texts = [
# #                     _get_segment_value(s, "text", "").replace("\n", " ")
# #                     for s in segs
# #                     if _get_segment_value(s, "text", "")
# #                 ]
# #                 return _clean_plain_blocks(texts) if clean else "\n".join(texts)
# #             except Exception:
# #                 continue

# #         # 4) Direct fallback
# #         try:
# #             segs = YouTubeTranscriptApi.get_transcript(video_id, languages=_EN_PRIORITY)
# #             texts = [
# #                 _get_segment_value(s, "text", "").replace("\n", " ")
# #                 for s in segs
# #                 if _get_segment_value(s, "text", "")
# #             ]
# #             return _clean_plain_blocks(texts) if clean else "\n".join(texts)
# #         except Exception:
# #             return None

# #     except TranscriptsDisabled:
# #         logger.info("Transcripts disabled for %s", video_id)
# #         return None
# #     except Exception as e:
# #         logger.debug("Transcript retrieval failed for %s: %s", video_id, e)
# #         return None


# # -----------------------
# # Media download helpers
# # -----------------------
# def _ensure_dir(path: Path) -> None:
#     path.mkdir(parents=True, exist_ok=True)


# def _audio_quality_to_kbps(quality: str) -> str:
#     """Convert quality string to kbps."""
#     q = (quality or "").lower()
#     if q in {"low", "l"}:
#         return "96"
#     if q in {"high", "h"}:
#         return "256"
#     return "160"  # default medium


# def _video_quality_to_height(quality: str) -> int:
#     q = (quality or "").lower()
#     if q in {"144p", "144"}:
#         return 144
#     if q in {"240p", "240"}:
#         return 240
#     if q in {"360p", "360"}:
#         return 360
#     if q in {"480p", "480"}:
#         return 480
#     if q in {"720p", "720"}:
#         return 720
#     if q in {"1080p", "1080", "fullhd", "fhd"}:
#         return 1080
#     if q in {"1440p", "1440", "2k"}:
#         return 1440
#     if q in {"2160p", "2160", "4k", "uhd"}:
#         return 2160
#     return 720


# def _finalize_path(output_dir: str, filename: str) -> Path:
#     base = Path(output_dir).resolve()
#     _ensure_dir(base)
#     return base / filename


# # -----------------
# # Audio download 
# # -----------------
# def download_audio_with_ytdlp(video_id_or_url: str, quality: str, output_dir: str) -> str:
#     """
#     Extracts audio to MP3 (with correct title in filename).
    
#     Args:
#         video_id_or_url: YouTube video ID or URL
#         quality: Audio quality (low/medium/high)
#         output_dir: Directory to save the file
        
#     Returns:
#         Path to downloaded MP3 file
#     """
#     url = _norm_youtube_url(video_id_or_url)
#     kbps = _audio_quality_to_kbps(quality)
    
#     opts = {
#         'format': 'bestaudio/best',
#         'postprocessors': [
#             {
#                 'key': 'FFmpegExtractAudio',
#                 'preferredcodec': 'mp3',
#                 'preferredquality': kbps,
#             },
#             {'key': 'FFmpegMetadata'},
#             {'key': 'EmbedThumbnail', 'already_have_thumbnail': False},
#         ],
#         'outtmpl': _safe_outtmpl(output_dir),
#         'noprogress': True,
#         'quiet': True,
#         'writethumbnail': True,
#         'extractor_args': {
#             'youtube': {
#                 'player_client': ['android', 'android_embedded']
#             }
#         },
#         'http_headers': {
#             'User-Agent': 'com.google.android.youtube/19.20.34 (Linux; U; Android 11)',
#             'Accept-Language': 'en-US,en;q=0.9',
#         },
#     }
    
#     ffmpeg_loc = _ensure_ffmpeg_location()
#     if ffmpeg_loc:
#         opts['ffmpeg_location'] = ffmpeg_loc

#     os.makedirs(output_dir, exist_ok=True)
    
#     try:
#         with YoutubeDL(opts) as ydl:
#             info = ydl.extract_info(url, download=True)
#             # yt-dlp tells us the actual filename after audio extraction
#             base_path = ydl.prepare_filename(info)
#             # After extraction, it will be .mp3
#             mp3_path = base_path.rsplit('.', 1)[0] + '.mp3'
#             return mp3_path
#     except Exception as e:
#         logger.error(f"Audio download failed: {e}")
#         raise


# # -----------------
# # Video download 
# # -----------------
# def download_video_with_ytdlp(video_id_or_url: str, quality: str, output_dir: str) -> str:
#     """
#     Download video with specified quality.
    
#     Args:
#         video_id_or_url: YouTube video ID or URL
#         quality: Video quality (e.g., '1080p', '720p', '480p', '360p')
#         output_dir: Directory to save the file
        
#     Returns:
#         Path to downloaded MP4 file
#     """
#     url = _norm_youtube_url(video_id_or_url)
#     q = re.sub(r'[^0-9]', '', quality or '')
#     height = int(q) if q.isdigit() else None

#     opts = _common_ydl_opts(output_dir)
    
#     if height:
#         # Prefer requested height, then nearest lower, still merging with audio
#         opts['format'] = (
#             f"bv*[height={height}][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
#             f"bv*[height<={height}][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
#             "bv*+ba/best"
#         )

#     os.makedirs(output_dir, exist_ok=True)

#     try:
#         with YoutubeDL(opts) as ydl:
#             info = ydl.extract_info(url, download=True)
#             # yt-dlp returns final filename here (after remux/merge)
#             fname = ydl.prepare_filename(info)
#             # After remux it might have changed extension to mp4
#             base, _ = os.path.splitext(fname)
#             mp4 = base + '.mp4'
#             return mp4 if os.path.exists(mp4) else fname
#     except Exception as e:
#         logger.error(f"Video download failed: {e}")
#         raise


# # ==================End transcript_utils Module==============
