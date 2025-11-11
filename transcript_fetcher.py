# transcript_fetcher.py — PRODUCTION READY
"""
Smart transcript fetching with multiple strategies to work around cloud IP blocks.
"""
import logging, os
from typing import Optional, List, Dict, Any, Iterable
from youtube_transcript_api import YouTubeTranscriptApi

# --- transcript_fetcher.py (add near top) ---
import base64, tempfile, subprocess, json, re
from typing import Optional, List, Dict, Any


logger = logging.getLogger("youtube_trans_downloader")

PROXY_LIST = os.getenv("YOUTUBE_PROXIES", "").split(",") if os.getenv("YOUTUBE_PROXIES") else []
USE_PROXIES = len(PROXY_LIST) > 0
_EN_PRIORITY = ["en", "en-US", "en-GB", "en-CA", "en-AU", "en-IE", "en-NZ"]

def _get(seg: Any, key: str, default: Any = None) -> Any:
    return seg.get(key, default) if isinstance(seg, dict) else getattr(seg, key, default)

def _sec_to_vtt(ts: float) -> str:
    h = int(ts // 3600); m = int((ts % 3600) // 60); s = int(ts % 60); ms = int((ts - int(ts)) * 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def _sec_to_srt(ts: float) -> str:
    h = int(ts // 3600); m = int((ts % 3600) // 60); s = int(ts % 60); ms = int((ts - int(ts)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def segments_to_vtt(segments) -> str:
    out = ["WEBVTT", "Kind: captions", "Language: en", ""]
    for seg in segments:
        start = _get(seg, "start", 0.0); dur = _get(seg, "duration", 0.0); text = _get(seg, "text", "")
        if not text: continue
        out.append(f"{_sec_to_vtt(start)} --> {_sec_to_vtt(start+dur)}"); out.append(text.replace("\n", " ").strip()); out.append("")
    return "\n".join(out)

def segments_to_srt(segments) -> str:
    out = []
    for i, seg in enumerate(segments, 1):
        start = _get(seg, "start", 0.0); dur = _get(seg, "duration", 0.0); text = _get(seg, "text", "")
        if not text: continue
        out += [str(i), f"{_sec_to_srt(start)} --> {_sec_to_srt(start+dur)}", text.replace("\n", " ").strip(), ""]
    return "\n".join(out)

def _clean_plain_blocks(blocks: List[str]) -> str:
    out, cur, chars = [], [], 0
    for w in " ".join(blocks).split():
        cur.append(w); chars += len(w) + 1
        if chars > 400 and w[-1:] in ".!?":
            out.append(" ".join(cur)); cur, chars = [], 0
    if cur: out.append(" ".join(cur))
    return "\n\n".join(out)

def _format_timestamped(segments) -> str:
    """Format segments as [mm:ss] text, avoiding backslashes inside f-string expressions."""
    lines = []
    for seg in segments:
        start = _get(seg, "start", 0.0)
        raw = _get(seg, "text", "") or ""
        if not raw:
            continue
        t = int(start)
        text_clean = raw.replace("\n", " ")
        lines.append(f"[{t//60:02d}:{t%60:02d}] {text_clean}")
    return "\n".join(lines)


def try_youtube_api_direct(video_id: str, proxies: Optional[Dict] = None):
    try:
        kw = {"languages": _EN_PRIORITY}
        if proxies: kw["proxies"] = proxies
        seg = YouTubeTranscriptApi.get_transcript(video_id, **kw)
        if seg: return seg
    except Exception as e:
        logger.debug("get_transcript failed: %s", e)
    try:
        kw = {}
        if proxies: kw["proxies"] = proxies
        listing = YouTubeTranscriptApi.list_transcripts(video_id, **kw)
        for code in _EN_PRIORITY:
            try:
                t = listing.find_transcript([code]); seg = t.fetch()
                if seg: return seg
            except: pass
        try:
            t = listing.find_generated_transcript(_EN_PRIORITY); seg = t.fetch()
            if seg: return seg
        except: pass
        for t in listing:
            try:
                seg = t.translate("en").fetch()
                if seg: return seg
            except: pass
    except Exception as e:
        logger.debug("list_transcripts failed: %s", e)
    return None

def try_ytdlp_fallback(video_id: str, clean: bool, fmt: Optional[str]) -> Optional[str]:
    try:
        from transcript_utils import get_transcript_with_ytdlp
        return get_transcript_with_ytdlp(video_id, clean=clean, fmt=fmt)
    except Exception as e:
        logger.debug("yt-dlp fallback exception: %s", e)
        return None


#------------------ Begin of Newly Added Functions to support ----------------

def _normalize_video_id_or_url(s: str) -> str:
    """Return a canonical URL for yt-dlp; accept bare IDs, watch URLs, shorts URLs."""
    s = s.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", s):
        return f"https://www.youtube.com/watch?v={s}"
    return s

def _write_cookies_tmp() -> Optional[str]:
    """
    Provide a path to a cookies file for yt-dlp if available.
    Priority:
      - YT_COOKIES_B64 (base64 of a Netscape/Chrome-exported cookies.txt)
      - YT_COOKIES_FILE (path already on disk)
    Returns the path or None.
    """
    b64 = os.getenv("YT_COOKIES_B64", "").strip()
    if b64:
        try:
            raw = base64.b64decode(b64)
            f = tempfile.NamedTemporaryFile(prefix="yt_cookies_", suffix=".txt", delete=False)
            f.write(raw)
            f.flush()
            f.close()
            return f.name
        except Exception as e:
            logger.warning("Failed to decode YT_COOKIES_B64: %s", e)

    fpath = os.getenv("YT_COOKIES_FILE", "").strip()
    if fpath and os.path.exists(fpath):
        return fpath

    return None

def _format_segments_clean(segments: List[Dict[str, Any]]) -> str:
    """Flatten segments into a single clean text blob (no timestamps)."""
    parts = []
    for seg in segments:
        text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
        cleaned = text.replace("\n", " ").strip()
        if cleaned:
            parts.append(cleaned)
    return "\n".join(parts)

def _format_segments_srt(segments: List[Dict[str, Any]]) -> str:
    """Minimal SRT output from a list of {start, duration, text} dicts."""
    def srt_time(t: float) -> str:
        # HH:MM:SS,mmm
        ms = int(round((t - int(t)) * 1000))
        t = int(t)
        h, t = divmod(t, 3600)
        m, s = divmod(t, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    out = []
    for i, seg in enumerate(segments, start=1):
        start = float(seg.get("start", 0.0))
        dur = float(seg.get("duration", 0.0))
        end = start + dur
        text = seg.get("text", "")
        cleaned = text.replace("\n", " ").strip()
        if not cleaned:
            continue
        out.append(str(i))
        out.append(f"{srt_time(start)} --> {srt_time(end)}")
        out.append(cleaned)
        out.append("")  # blank line between cues
    return "\n".join(out).strip()

def _yt_dlp_transcript(video: str, want_format: Optional[str]) -> Optional[str]:
    """
    Use yt-dlp to fetch subtitles. want_format in {None, 'srt', 'vtt'}.
    Returns text content or None if unavailable.
    """
    url = _normalize_video_id_or_url(video)
    cookies_path = _write_cookies_tmp()

    # Decide sub-format flags for yt-dlp
    sub_fmt = "srt" if want_format == "srt" else "vtt" if want_format == "vtt" else "vtt"

    with tempfile.TemporaryDirectory(prefix="ytdlp_subs_") as tmpdir:
        base = os.path.join(tmpdir, "out")
        # yt-dlp arguments: auto subs if manual not present; convert to desired format
        args = [
            "yt-dlp",
            "--skip-download",
            "--no-warnings",
            "--quiet",
            "--no-call-home",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            "--write-auto-subs",
            "--sub-langs", "en.*,en",
            "--sub-format", sub_fmt,
            "--convert-subs", sub_fmt,
            "-o", f"{base}.%(ext)s",
            url,
        ]
        if cookies_path:
            args.extend(["--cookies", cookies_path])

        logger.info("Trying yt-dlp fallback for %s (format: %s)", video, want_format)
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
        except Exception as e:
            logger.error("yt-dlp failed to run: %s", e)
            return None

        if proc.returncode != 0:
            # Helpful for debugging rate limit or bot checks
            if proc.stderr:
                logger.error("yt-dlp stderr: %s", proc.stderr.strip())
            return None

        # Look for produced file
        for ext in [sub_fmt, sub_fmt.upper()]:
            candidate = f"{base}.{ext}"
            if os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()

    return None

# --- PUBLIC ENTRYPOINT: call this from main.py ---
def get_transcript_smart(video_id_or_url: str, clean: bool = True, fmt: Optional[str] = None) -> Optional[str]:
    """
    1) Try youtube_transcript_api (fast, no cookies) → returns clean text or SRT
    2) If blocked/missing, try yt-dlp with cookies (handles bot checks, Shorts, etc.)
    clean=True returns plain text (no timestamps). If fmt in {'srt','vtt'}, returns that format.
    """
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript

    vid = video_id_or_url.strip()
    logger.info("Smart fetch %s (clean=%s, fmt=%s)", vid, clean, fmt)

    # 1) API path (best case)
    try:
        # Bare ID only for the API
        m = re.search(r"[A-Za-z0-9_-]{10,}", vid)
        bare_id = m.group(0) if m else vid

        transcript_list = YouTubeTranscriptApi.list_transcripts(bare_id)
        # Prefer English (manual or auto)
        try:
            transcript = transcript_list.find_manually_created_transcript(['en', 'en-US'])
        except:
            transcript = transcript_list.find_transcript(['en', 'en-US'])
        segments = transcript.fetch()  # list of dicts: start, duration, text

        if clean and fmt is None:
            return _format_segments_clean(segments)
        if fmt == "srt":
            return _format_segments_srt(segments)
        if fmt == "vtt":
            # simple VTT (we can transform SRT to VTT if needed; here we output minimal VTT)
            srt_like = _format_segments_srt(segments).splitlines()
            # Convert SRT to VTT quickly
            vtt = ["WEBVTT", ""]
            i = 0
            while i < len(srt_like):
                line = srt_like[i].strip()
                if re.fullmatch(r"\d+", line):
                    i += 1
                    if i >= len(srt_like): break
                    times = srt_like[i].replace(",", ".")
                    vtt.append(times)
                    i += 1
                    # collect text lines until blank
                    while i < len(srt_like) and srt_like[i].strip():
                        vtt.append(srt_like[i])
                        i += 1
                    vtt.append("")
                else:
                    i += 1
            return "\n".join(vtt).strip()
        # default to clean
        return _format_segments_clean(segments)

    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript) as e:
        logger.info("API transcript not available: %s", e)
    except Exception as e:
        logger.warning("API path failed unexpectedly: %s", e)

    # 2) yt-dlp fallback (needs cookies for bot checks)
    text = _yt_dlp_transcript(vid, fmt if fmt in ("srt", "vtt") else "vtt")
    if not text:
        logger.error("Transcript fetch failed for %s: Could not retrieve transcript (no captions or YouTube blocked our requests).", vid)
        return None

    if clean and fmt is None:
        # Very lightweight VTT/SRT → clean text conversion
        # remove cue headers and timestamps
        cleaned_lines = []
        for line in text.splitlines():
            if re.match(r"^\d+$", line.strip()):
                continue
            if re.search(r"-->\s", line):
                continue
            if line.strip().upper() in ("WEBVTT",):
                continue
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    return text
    
#------------------ End of Newly Added Functions to support ----------------

def get_transcript_smart(video_id: str, clean: bool = True, fmt: Optional[str] = None, use_proxies: bool = USE_PROXIES) -> str:
    logger.info("Smart fetch %s (clean=%s, fmt=%s)", video_id, clean, fmt)

    # 1) Direct API first (fastest if not blocked)
    seg = try_youtube_api_direct(video_id)
    if seg:
        if fmt == "srt": return segments_to_srt(seg)
        if fmt == "vtt": return segments_to_vtt(seg)
        if clean:        return _clean_plain_blocks([_get(s, "text", "").replace("\n", " ") for s in seg])
        return _format_timestamped(seg)

    # 2) Strategy 2: Try yt-dlp fallback (supports txt, srt, vtt)
    logger.info(f"Trying yt-dlp fallback for {video_id} (format: {fmt})")
    ytdlp_result = try_ytdlp_fallback(video_id, clean=clean, fmt=fmt)
    if ytdlp_result:
        logger.info(f"✅ yt-dlp fallback successful for {video_id}")
        return ytdlp_result    

    # 3) Proxy attempts (optional)
    if use_proxies:
        for proxy_url in PROXY_LIST:
            proxies = {"http": proxy_url, "https": proxy_url}
            seg = try_youtube_api_direct(video_id, proxies=proxies)
            if seg:
                if fmt == "srt": return segments_to_srt(seg)
                if fmt == "vtt": return segments_to_vtt(seg)
                if clean:        return _clean_plain_blocks([_get(s, "text", "").replace("\n", " ") for s in seg])
                return _format_timestamped(seg)

    raise Exception("Could not retrieve transcript (no captions or YouTube blocked our requests).")

def fallback_auto_subs_with_ytdlp(video_id: str, lang: str = "en") -> list[dict]:
    import subprocess, json, tempfile, os
    # Ask yt_dlp for automatic subtitles (no download)
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "--skip-download",
        "--write-auto-sub",
        "--sub-langs", lang,
        "--sub-format", "json3",
        "-J"  # dump metadata as JSON (contains subtitles)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:  # no auto subs available
        return []
    data = json.loads(proc.stdout)
    tracks = (data.get("subtitles") or {}) or (data.get("automatic_captions") or {})
    track = (tracks.get(lang) or tracks.get(f"{lang}-orig") or [])
    # Pick first JSON3 track and parse events -> [{'text': ..., 'start': ...}]
    url = next((t.get("url") for t in track if "json3" in (t.get("ext") or "")), None)
    if not url:
        return []
    # Fetch JSON3
    import urllib.request
    with urllib.request.urlopen(url) as r:
        j3 = r.read().decode("utf-8")
    j3 = json.loads(j3)
    out = []
    for ev in j3.get("events", []):
        segs = ev.get("segs") or []
        text = "".join(s.get("utf8","") for s in segs).strip()
        if not text:
            continue
        t = ev.get("tStartMs", 0) // 1000
        out.append({"text": text, "start": t})
    return out

def segments_to_text_timestamped(segments: Iterable[Any]) -> str:
    """
    Convert a sequence of transcript segments into timestamped lines.

    Each segment may be a dict (e.g., {"text": "...", "start": 12.3})
    or an object with .text / .start (or .start_time) attributes.

    Returns a single '\n'-joined string like:
    [00:12] Hello world
    [00:15] Another line
    """
    lines: list[str] = []

    for seg in segments or []:
        # Extract text
        if isinstance(seg, dict):
            text = seg.get("text") or seg.get("utf8") or ""
            start = seg.get("start", seg.get("start_time", seg.get("tStartMs")))
        else:
            text = getattr(seg, "text", "") or getattr(seg, "utf8", "")
            start = getattr(seg, "start", getattr(seg, "start_time", getattr(seg, "tStartMs", None)))

        # Normalize start -> seconds (int)
        if start is None:
            t_seconds = 0
        else:
            try:
                # Handle ms fields (tStartMs) and numeric seconds
                val = float(start)
                # If very large, treat as milliseconds
                t_seconds = int(round(val / 1000.0)) if val > 10_000 else int(round(val))
            except Exception:
                t_seconds = 0

        # Clean text (NO backslashes in f-string expression)
        safe = (text or "").replace("\n", " ").strip()
        if not safe:
            continue

        # Append formatted line
        lines.append(f"[{t_seconds // 60:02d}:{t_seconds % 60:02d}] {safe}")

    return "\n".join(lines)

