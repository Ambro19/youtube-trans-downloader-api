# transcript_fetcher.py — PRODUCTION READY
"""
Smart transcript fetching with multiple strategies to work around cloud IP blocks.
"""
import logging, os
from typing import Optional, List, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi

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


