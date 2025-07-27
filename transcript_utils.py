# Enhanced transcript_utils.py - Audio/Video Support + Better Processing
import os
import json
import subprocess
import time
import logging
import tempfile
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger("transcript_utils")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Audio quality settings for yt-dlp
AUDIO_FORMATS = {
    'high': {
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
        'quality': '0',  # Best quality
        'bitrate': '320k'
    },
    'medium': {
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
        'quality': '5',  # Medium quality
        'bitrate': '192k'
    },
    'low': {
        'format': 'worstaudio[ext=m4a]/worstaudio[ext=mp3]/worstaudio',
        'quality': '9',  # Lowest quality
        'bitrate': '96k'
    }
}

# Video quality settings for yt-dlp
VIDEO_FORMATS = {
    '1080p': 'best[height<=1080][ext=mp4]/best[height<=1080]/best[ext=mp4]/best',
    '720p': 'best[height<=720][ext=mp4]/best[height<=720]/best[ext=mp4]/best',
    '480p': 'best[height<=480][ext=mp4]/best[height<=480]/best[ext=mp4]/best',
    '360p': 'best[height<=360][ext=mp4]/best[height<=360]/best[ext=mp4]/best'
}

# =============================================================================
# TRANSCRIPT FUNCTIONS
# =============================================================================

def get_transcript_with_ytdlp(video_id: str, clean=True, retries=3, wait_sec=1) -> Optional[str]:
    """
    Get transcript using yt-dlp as fallback
    
    Args:
        video_id: YouTube video ID
        clean: If True, return clean text. If False, return with timestamps
        retries: Number of retry attempts
        wait_sec: Seconds to wait between retries
    
    Returns:
        Formatted transcript string or None if failed
    """
    try:
        output_vtt = f"{video_id}.en.vtt"
        output_json3 = f"{video_id}.en.json3"
        url = f"https://www.youtube.com/watch?v={video_id}"

        # Enhanced yt-dlp command with better error handling
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--sub-format", "json3/vtt/srt",
            "--output", "%(id)s",
            "--no-warnings",
            "--quiet",
            url
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=30,
            check=False
        )
        
        if result.returncode != 0 and result.stderr:
            logger.warning(f"yt-dlp warning: {result.stderr}")

        # Try JSON3 first (most reliable)
        for attempt in range(retries):
            if os.path.exists(output_json3):
                try:
                    with open(output_json3, encoding="utf8") as f:
                        data = json.load(f)
                    
                    # Clean up file
                    os.remove(output_json3)
                    
                    return _process_json3_transcript(data, clean)
                    
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to parse JSON3 file: {e}")
                    if os.path.exists(output_json3):
                        os.remove(output_json3)
                    break
            
            time.sleep(wait_sec)

        # Fallback to VTT processing
        if os.path.exists(output_vtt):
            try:
                with open(output_vtt, encoding="utf8") as f:
                    vtt_content = f.read()
                
                # Clean up file
                os.remove(output_vtt)
                
                if clean:
                    return extract_vtt_text(vtt_content)
                else:
                    return format_transcript_vtt(vtt_content)
                    
            except IOError as e:
                logger.warning(f"Failed to read VTT file: {e}")
                if os.path.exists(output_vtt):
                    os.remove(output_vtt)

        # Clean up any remaining files
        _cleanup_temp_files(video_id)
        
        logger.error(f"No transcript files found for video: {video_id}")
        return None

    except subprocess.TimeoutExpired:
        logger.error(f"yt-dlp timeout for video: {video_id}")
        _cleanup_temp_files(video_id)
        return None
    except Exception as e:
        logger.error(f"yt-dlp fallback failed for {video_id}: {e}")
        _cleanup_temp_files(video_id)
        return None

def _process_json3_transcript(data: Dict[Any, Any], clean: bool) -> Optional[str]:
    """Process JSON3 transcript data"""
    try:
        blocks = []
        
        for event in data.get("events", []):
            if "segs" in event and "tStartMs" in event:
                # Extract text segments
                text_segments = []
                for seg in event["segs"]:
                    if seg.get("utf8"):
                        text_segments.append(seg["utf8"])
                
                if text_segments:
                    text = ''.join(text_segments).strip()
                    
                    if text:
                        if clean:
                            blocks.append(text)
                        else:
                            # Create timestamp
                            start_ms = event["tStartMs"]
                            seconds = int(start_ms // 1000)
                            timestamp = f"[{seconds//60:02d}:{seconds%60:02d}]"
                            blocks.append(f"{timestamp} {text}")
        
        if blocks:
            return "\n".join(blocks) if not clean else " ".join(blocks)
        
        return None
        
    except Exception as e:
        logger.error(f"Error processing JSON3 transcript: {e}")
        return None

def extract_vtt_text(vtt_content: str) -> str:
    """Extract clean text from VTT content"""
    try:
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
                not line.isdigit() and
                not re.match(r'^\d{2}:\d{2}:\d{2}', line)):
                
                # Remove HTML tags and formatting
                clean_line = re.sub(r'<[^>]+>', '', line)
                clean_line = re.sub(r'\{[^}]+\}', '', clean_line)
                
                if clean_line.strip():
                    text_lines.append(clean_line.strip())
        
        return ' '.join(text_lines)
        
    except Exception as e:
        logger.error(f"Error extracting VTT text: {e}")
        return ""

def format_transcript_vtt(raw_vtt: str) -> str:
    """Format VTT content to proper WEBVTT standard"""
    try:
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
                        # Clean HTML tags and formatting
                        clean_text = re.sub(r'<[^>]+>', '', text_line)
                        clean_text = re.sub(r'\{[^}]+\}', '', clean_text)
                        if clean_text.strip():
                            text_content.append(clean_text.strip())
                    i += 1
                
                # Add text content
                if text_content:
                    formatted_lines.append(' '.join(text_content))
                
                # Add separator line
                formatted_lines.append("")
            else:
                i += 1
        
        return '\n'.join(formatted_lines)
        
    except Exception as e:
        logger.error(f"Error formatting VTT: {e}")
        return raw_vtt

# =============================================================================
# AUDIO DOWNLOAD FUNCTIONS
# =============================================================================

# Replace the audio quality mapping in your transcript_utils.py

def download_audio_with_ytdlp(video_id: str, quality: str = "medium", output_dir: str = "downloads") -> str:
    """
    Download audio from YouTube video using yt-dlp with better quality settings
    """
    # FIXED: Better quality settings for playable audio files
    quality_map = {
        "high": {
            "audio_quality": "0",  # Best quality (was 3)
            "format": "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio"  # Best available
        },
        "medium": {
            "audio_quality": "2",  # Good quality (was 5) 
            "format": "bestaudio[abr<=128]/bestaudio[ext=m4a]/bestaudio"  # Max 128kbps
        },
        "low": {
            "audio_quality": "5",  # Acceptable quality (was 9)
            "format": "bestaudio[abr<=96]/bestaudio[ext=m4a]/bestaudio"   # Max 96kbps
        }
    }
    
    settings = quality_map.get(quality, quality_map["medium"])
    
    # Enhanced command for better compatibility
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", settings["audio_quality"],
        "--format", settings["format"],
        "--output", f"{output_dir}/{video_id}_audio_%(quality)s.%(ext)s",
        "--no-playlist",
        "--no-warnings",
        "--prefer-ffmpeg",  # Prefer FFmpeg for better compatibility
        #"--ffmpeg-location", ffmpeg_path,
        "--embed-metadata",  # Add metadata for better file recognition
        f"https://www.youtube.com/watch?v={video_id}"
    ]
    
    logger.info(f"Starting audio download for {video_id} at {quality} quality")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Run with timeout and better error handling
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300,  # 5 minute timeout
            cwd=output_dir,
            check=True
        )
        
        # Find the output file
        pattern = f"{video_id}_audio_*.mp3"
        audio_files = list(Path(output_dir).glob(pattern))
        
        if audio_files:
            audio_file = str(audio_files[0])
            file_size = os.path.getsize(audio_file)
            
            # Verify the file is not corrupted (minimum size check)
            if file_size < 1000:  # Less than 1KB is definitely corrupted
                logger.error(f"Downloaded file too small ({file_size} bytes), likely corrupted")
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                raise Exception("Downloaded audio file is corrupted (too small)")
            
            logger.info(f"Audio download successful: {os.path.basename(audio_file)} ({file_size} bytes)")
            return audio_file
        else:
            raise Exception("No audio file found after download")
            
    except subprocess.TimeoutExpired:
        logger.error(f"Audio download timed out for {video_id}")
        raise Exception("Download timed out")
    except subprocess.CalledProcessError as e:
        logger.error(f"yt-dlp error: {e.stderr}")
        raise Exception(f"Audio download failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Audio download error: {e}")
        raise

# =============================================================================
# VIDEO DOWNLOAD FUNCTIONS
# =============================================================================

def download_video_with_ytdlp(video_id: str, quality: str = "720p", output_dir: str = None) -> Optional[str]:
    """
    Download video from YouTube using yt-dlp
    
    Args:
        video_id: YouTube video ID
        quality: Video quality ("1080p", "720p", "480p", "360p")
        output_dir: Output directory (uses temp dir if None)
    
    Returns:
        Path to downloaded video file or None if failed
    """
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Get quality format
        video_format = VIDEO_FORMATS.get(quality, VIDEO_FORMATS["720p"])
        
        # Create output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        output_template = os.path.join(output_dir, f"{video_id}_video_{quality}.%(ext)s")
        
        # Enhanced yt-dlp command for video
        cmd = [
            "yt-dlp",
            "--format", video_format,
            "--output", output_template,
            "--no-playlist",
            "--no-warnings",
            "--merge-output-format", "mp4",
            url
        ]
        
        logger.info(f"Starting video download for {video_id} at {quality} quality")
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=600,  # 10 minutes timeout
            check=False
        )
        
        if result.returncode == 0:
            # Find the downloaded file
            for filename in os.listdir(output_dir):
                if filename.startswith(f"{video_id}_video_{quality}") and (
                    filename.endswith('.mp4') or filename.endswith('.mkv') or filename.endswith('.webm')
                ):
                    file_path = os.path.join(output_dir, filename)
                    file_size = os.path.getsize(file_path)
                    
                    logger.info(f"Video download successful: {filename} ({file_size} bytes)")
                    return file_path
        
        logger.error(f"Video download failed for {video_id}: {result.stderr}")
        return None
        
    except subprocess.TimeoutExpired:
        logger.error(f"Video download timeout for {video_id}")
        return None
    except Exception as e:
        logger.error(f"Video download error for {video_id}: {e}")
        return None

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_video_info(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Get video information using yt-dlp
    
    Args:
        video_id: YouTube video ID
    
    Returns:
        Dictionary with video information or None if failed
    """
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-warnings",
            "--no-download",
            url
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False
        )
        
        if result.returncode == 0 and result.stdout:
            video_info = json.loads(result.stdout)
            
            # Extract useful information
            return {
                "id": video_info.get("id"),
                "title": video_info.get("title"),
                "duration": video_info.get("duration"),
                "upload_date": video_info.get("upload_date"),
                "uploader": video_info.get("uploader"),
                "view_count": video_info.get("view_count"),
                "like_count": video_info.get("like_count"),
                "description": video_info.get("description", "")[:500],  # Truncate description
                "thumbnail": video_info.get("thumbnail"),
                "has_subtitles": bool(video_info.get("subtitles")),
                "has_auto_captions": bool(video_info.get("automatic_captions"))
            }
        
        return None
        
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to get video info for {video_id}: {e}")
        return None

def check_ytdlp_availability() -> bool:
    """Check if yt-dlp is available and working"""
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info(f"yt-dlp is available: {version}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"yt-dlp not available: {e}")
        return False

def estimate_file_size(video_id: str, download_type: str, quality: str) -> Optional[int]:
    """
    Estimate file size for a download
    
    Args:
        video_id: YouTube video ID
        download_type: "audio" or "video"
        quality: Quality setting
    
    Returns:
        Estimated file size in bytes or None if cannot estimate
    """
    try:
        video_info = get_video_info(video_id)
        if not video_info or not video_info.get("duration"):
            return None
        
        duration_seconds = video_info["duration"]
        
        if download_type == "audio":
            # Estimate based on bitrate
            bitrates = {"high": 320, "medium": 192, "low": 96}  # kbps
            bitrate = bitrates.get(quality, 192)
            estimated_size = (duration_seconds * bitrate * 1000) // 8  # Convert to bytes
            
        elif download_type == "video":
            # Rough estimates based on quality (very approximate)
            size_per_minute = {
                "1080p": 100 * 1024 * 1024,  # ~100MB per minute
                "720p": 50 * 1024 * 1024,    # ~50MB per minute
                "480p": 25 * 1024 * 1024,    # ~25MB per minute
                "360p": 15 * 1024 * 1024     # ~15MB per minute
            }
            
            rate = size_per_minute.get(quality, size_per_minute["720p"])
            estimated_size = (duration_seconds * rate) // 60
            
        else:
            return None
        
        return int(estimated_size)
        
    except Exception as e:
        logger.error(f"Error estimating file size: {e}")
        return None

def _cleanup_temp_files(video_id: str):
    """Clean up temporary files created during processing"""
    try:
        temp_patterns = [
            f"{video_id}*.vtt",
            f"{video_id}*.json3",
            f"{video_id}*.srt",
            f"{video_id}*.json",
            f"{video_id}*.info.json"
        ]
        
        for pattern in temp_patterns:
            import glob
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
                    
    except Exception as e:
        logger.warning(f"Error cleaning up temp files: {e}")

def validate_video_id(video_id: str) -> bool:
    """Validate YouTube video ID format"""
    if not video_id or len(video_id) != 11:
        return False
    
    # YouTube video IDs contain only alphanumeric characters, hyphens, and underscores
    import string
    valid_chars = string.ascii_letters + string.digits + '-_'
    
    return all(c in valid_chars for c in video_id)

def format_transcript_clean(text: str) -> str:
    """Format clean transcript with proper paragraph breaks"""
    try:
        # Split into sentences and group into paragraphs
        sentences = re.split(r'[.!?]+\s+', text)
        paragraphs = []
        current_paragraph = []
        char_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            current_paragraph.append(sentence)
            char_count += len(sentence) + 1
            
            # Create paragraph break every ~400 characters
            if char_count > 400:
                paragraphs.append('. '.join(current_paragraph) + '.')
                current_paragraph = []
                char_count = 0
        
        # Add remaining sentences
        if current_paragraph:
            paragraphs.append('. '.join(current_paragraph) + '.')
        
        return '\n\n'.join(paragraphs)
        
    except Exception as e:
        logger.error(f"Error formatting clean transcript: {e}")
        return text

# =============================================================================
# TESTING AND DEBUGGING
# =============================================================================

def test_transcript_extraction(video_id: str = "dQw4w9WgXcQ"):
    """Test transcript extraction functions"""
    logger.info(f"Testing transcript extraction for video: {video_id}")
    
    # Test clean transcript
    clean_result = get_transcript_with_ytdlp(video_id, clean=True)
    if clean_result:
        logger.info(f"Clean transcript: {len(clean_result)} characters")
    else:
        logger.error("Clean transcript extraction failed")
    
    # Test timestamped transcript
    timestamped_result = get_transcript_with_ytdlp(video_id, clean=False)
    if timestamped_result:
        logger.info(f"Timestamped transcript: {len(timestamped_result)} characters")
    else:
        logger.error("Timestamped transcript extraction failed")
    
    return clean_result, timestamped_result

if __name__ == "__main__":
    # Test the functions
    logging.basicConfig(level=logging.INFO)
    
    print("Testing yt-dlp availability...")
    if check_ytdlp_availability():
        print("✅ yt-dlp is available")
        
        print("\nTesting transcript extraction...")
        clean, timestamped = test_transcript_extraction()
        
        if clean:
            print(f"✅ Clean transcript extracted ({len(clean)} chars)")
        if timestamped:
            print(f"✅ Timestamped transcript extracted ({len(timestamped)} chars)")
    else:
        print("❌ yt-dlp is not available")

#================================================================
# # transcript_utils.py - Updated for better VTT processing
# import os
# import json
# import subprocess
# import time
# import logging
# from pathlib import Path
# import re

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

#         # Try JSON3 first (more reliable)
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

#         # Fallback to VTT processing
#         if os.path.exists(output_vtt):
#             with open(output_vtt, encoding="utf8") as f:
#                 vtt_raw = f.read()
#             os.remove(output_vtt)
            
#             if clean:
#                 # Extract just the text content
#                 return extract_vtt_text(vtt_raw)
#             else:
#                 # Return properly formatted VTT
#                 return format_transcript_vtt(vtt_raw)

#         raise FileNotFoundError(f"yt-dlp output not found: {output_json3} or .vtt")

#     except Exception as e:
#         logger.error(f"yt-dlp fallback failed: {e}")
#         return None

# def extract_vtt_text(vtt_content: str) -> str:
#     """Extract clean text from VTT content."""
#     lines = vtt_content.strip().splitlines()
#     text_lines = []
    
#     for line in lines:
#         line = line.strip()
#         # Skip VTT headers, timestamps, and empty lines
#         if (line and 
#             not line.startswith('WEBVTT') and 
#             not line.startswith('Kind:') and 
#             not line.startswith('Language:') and 
#             not '-->' in line and
#             not line.startswith('NOTE') and
#             not line.isdigit()):
#             text_lines.append(line)
    
#     return ' '.join(text_lines)

# def format_transcript_vtt(raw_vtt: str) -> str:
#     """Format VTT content to proper WEBVTT standard."""
#     lines = raw_vtt.strip().splitlines()
#     formatted_lines = []
    
#     # Add proper headers
#     formatted_lines.extend([
#         "WEBVTT",
#         "Kind: captions",
#         "Language: en",
#         ""
#     ])
    
#     i = 0
#     while i < len(lines):
#         line = lines[i].strip()
        
#         # Skip original headers
#         if line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
#             i += 1
#             continue
            
#         # Process timestamp lines
#         if '-->' in line:
#             # Clean up timestamp format
#             timestamp_line = re.sub(r'(\d{2}:\d{2}:\d{2})\.(\d{3})', r'\1.\2', line)
#             formatted_lines.append(timestamp_line)
#             i += 1
            
#             # Get the text content
#             text_content = []
#             while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
#                 text_line = lines[i].strip()
#                 if text_line and not text_line.isdigit():
#                     text_content.append(text_line)
#                 i += 1
            
#             # Add text content
#             if text_content:
#                 formatted_lines.append(' '.join(text_content))
            
#             # Add separator line
#             formatted_lines.append("")
#         else:
#             i += 1
    
#     return '\n'.join(formatted_lines)

# def format_transcript_clean(text: str) -> str:
#     """Format clean transcript with proper paragraph breaks."""
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

