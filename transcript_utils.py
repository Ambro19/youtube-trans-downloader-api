# transcript_utils.py

import os
import json
import subprocess
import time
import logging
from pathlib import Path

logger = logging.getLogger("transcript_utils")

def get_transcript_with_ytdlp(video_id: str, clean=True, retries=3, wait_sec=1) -> str:
    try:
        output_vtt = f"{video_id}.en.vtt"
        output_json3 = f"{video_id}.en.json3"
        url = f"https://www.youtube.com/watch?v={video_id}"

        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--sub-format", "json3/vtt",
            "--output", "%(id)s",
            url
        ]
        subprocess.run(cmd, capture_output=True, check=False)

        for _ in range(retries):
            if os.path.exists(output_json3):
                with open(output_json3, encoding="utf8") as f:
                    data = json.load(f)
                os.remove(output_json3)

                blocks = []
                for e in data.get("events", []):
                    if "segs" in e and "tStartMs" in e:
                        text = ''.join(s.get("utf8", '') for s in e["segs"] if s.get("utf8"))
                        if text.strip():
                            sec = int(e["tStartMs"] // 1000)
                            ts = f"[{sec//60:02d}:{sec%60:02d}]"
                            blocks.append(f"{ts} {text.strip()}" if not clean else text.strip())
                return "\n".join(blocks) if blocks else None

            time.sleep(wait_sec)

        if os.path.exists(output_vtt):
            with open(output_vtt, encoding="utf8") as f:
                vtt_raw = f.read()
            os.remove(output_vtt)
            return vtt_raw.strip() if vtt_raw else None

        raise FileNotFoundError(f"yt-dlp output not found: {output_json3} or .vtt")

    except Exception as e:
        logger.error(f"yt-dlp fallback failed: {e}")
        return None

def format_transcript_clean(text: str) -> str:
    paragraphs = text.split("\n")
    output = []
    current = []
    for line in paragraphs:
        if line.strip() == "":
            if current:
                output.append(" ".join(current).strip())
                current = []
        else:
            current.append(line.strip())
    if current:
        output.append(" ".join(current).strip())
    return "\n\n".join(output)

def format_transcript_vtt(raw_vtt: str) -> str:
    lines = raw_vtt.strip().splitlines()
    cleaned = ["WEBVTT\n"]
    for line in lines:
        if "-->" in line or line.strip() == "":
            cleaned.append(line)
        elif not line.startswith("WEBVTT") and not line.startswith("Kind") and not line.startswith("Language"):
            cleaned.append(line.strip())
    return "\n".join(cleaned).strip()

#========================
# transcript_utils.py

# import os
# import json
# import subprocess
# import time
# import logging
# from pathlib import Path

# logger = logging.getLogger("transcript_utils")

# def get_transcript_with_ytdlp(video_id: str, clean=True, retries=3, wait_sec=1) -> str:
#     try:
#         output_vtt = f"{video_id}.en.vtt"
#         output_json3 = f"{video_id}.en.json3"
#         url = f"https://www.youtube.com/watch?v={video_id}"

#         cmd = [
#             "yt-dlp",
#             "--skip-download",
#             "--write-auto-sub",
#             "--sub-lang", "en",
#             "--sub-format", "json3/vtt",
#             "--output", "%(id)s",
#             url
#         ]
#         subprocess.run(cmd, capture_output=True, check=False)

#         for _ in range(retries):
#             if os.path.exists(output_json3):
#                 with open(output_json3, encoding="utf8") as f:
#                     data = json.load(f)
#                 os.remove(output_json3)

#                 blocks = []
#                 for e in data.get("events", []):
#                     if "segs" in e and "tStartMs" in e:
#                         text = ''.join(s.get("utf8", '') for s in e["segs"] if s.get("utf8"))
#                         if text.strip():
#                             sec = int(e["tStartMs"] // 1000)
#                             ts = f"[{sec//60:02d}:{sec%60:02d}]"
#                             blocks.append(f"{ts} {text.strip()}" if not clean else text.strip())
#                 return "\n".join(blocks) if blocks else None

#             time.sleep(wait_sec)

#         if os.path.exists(output_vtt):
#             with open(output_vtt, encoding="utf8") as f:
#                 vtt_raw = f.read()
#             os.remove(output_vtt)
#             return vtt_raw.strip() if vtt_raw else None

#         raise FileNotFoundError(f"yt-dlp output not found: {output_json3} or .vtt")

#     except Exception as e:
#         logger.error(f"yt-dlp fallback failed: {e}")
#         return None

# def format_transcript_clean(text: str) -> str:
#     paragraphs = text.split("\n")
#     output = []
#     current = []
#     for line in paragraphs:
#         if line.strip() == "":
#             if current:
#                 output.append(" ".join(current).strip())
#                 current = []
#         else:
#             current.append(line.strip())
#     if current:
#         output.append(" ".join(current).strip())
#     return "\n\n".join(output)

# def format_transcript_vtt(raw_vtt: str) -> str:
#     lines = raw_vtt.strip().splitlines()
#     cleaned = ["WEBVTT\n"]
#     for line in lines:
#         if "-->") in line or line.strip() == "":
#             cleaned.append(line)
#         elif not line.startswith("WEBVTT") and not line.startswith("Kind") and not line.startswith("Language"):
#             cleaned.append(line.strip())
#     return "\n".join(cleaned).strip()

