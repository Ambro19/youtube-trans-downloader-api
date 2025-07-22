#============ IMPLEMENTATION FROM CLAUDIO.AI ============
# transcript_utils.py - Updated for better VTT processing
import os
import json
import subprocess
import time
import logging
from pathlib import Path
import re

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

        # Try JSON3 first (more reliable)
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

        # Fallback to VTT processing
        if os.path.exists(output_vtt):
            with open(output_vtt, encoding="utf8") as f:
                vtt_raw = f.read()
            os.remove(output_vtt)
            
            if clean:
                # Extract just the text content
                return extract_vtt_text(vtt_raw)
            else:
                # Return properly formatted VTT
                return format_transcript_vtt(vtt_raw)

        raise FileNotFoundError(f"yt-dlp output not found: {output_json3} or .vtt")

    except Exception as e:
        logger.error(f"yt-dlp fallback failed: {e}")
        return None

def extract_vtt_text(vtt_content: str) -> str:
    """Extract clean text from VTT content."""
    lines = vtt_content.strip().splitlines()
    text_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip VTT headers, timestamps, and empty lines
        if (line and 
            not line.startswith('WEBVTT') and 
            not line.startswith('Kind:') and 
            not line.startswith('Language:') and 
            not '-->' in line and
            not line.startswith('NOTE') and
            not line.isdigit()):
            text_lines.append(line)
    
    return ' '.join(text_lines)

def format_transcript_vtt(raw_vtt: str) -> str:
    """Format VTT content to proper WEBVTT standard."""
    lines = raw_vtt.strip().splitlines()
    formatted_lines = []
    
    # Add proper headers
    formatted_lines.extend([
        "WEBVTT",
        "Kind: captions",
        "Language: en",
        ""
    ])
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip original headers
        if line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
            i += 1
            continue
            
        # Process timestamp lines
        if '-->' in line:
            # Clean up timestamp format
            timestamp_line = re.sub(r'(\d{2}:\d{2}:\d{2})\.(\d{3})', r'\1.\2', line)
            formatted_lines.append(timestamp_line)
            i += 1
            
            # Get the text content
            text_content = []
            while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                text_line = lines[i].strip()
                if text_line and not text_line.isdigit():
                    text_content.append(text_line)
                i += 1
            
            # Add text content
            if text_content:
                formatted_lines.append(' '.join(text_content))
            
            # Add separator line
            formatted_lines.append("")
        else:
            i += 1
    
    return '\n'.join(formatted_lines)

def format_transcript_clean(text: str) -> str:
    """Format clean transcript with proper paragraph breaks."""
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

