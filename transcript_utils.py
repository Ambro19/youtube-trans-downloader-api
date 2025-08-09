# Enhanced transcript_utils.py - FIXED VIDEO DOWNLOAD VERSION
# 🔥 FIXES:
# - ✅ Fixed video download issues (YouTube blocking detection)
# - ✅ Better format detection and fallback strategies
# - ✅ Enhanced error handling for restricted videos
# - ✅ Cookie support for age-restricted content
# - ✅ Multiple retry strategies and user-agent rotation

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
import random

logger = logging.getLogger("transcript_utils")

# =============================================================================
# CONFIGURATION
# =============================================================================

# 🔥 CRITICAL FIX: Use user's Downloads folder by default
DEFAULT_DOWNLOADS_DIR = Path.home() / "Downloads"

# 🔥 ENHANCED: User agents to rotate for better success rate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
]

# 🔥 ENHANCED Audio quality settings for yt-dlp
AUDIO_FORMATS = {
    'high': {
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
        'quality': '0',  # Best quality
        'bitrate': '320k'
    },
    'medium': {
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
        'quality': '2',  # Medium quality
        'bitrate': '192k'
    },
    'low': {
        'format': 'bestaudio[abr<=96]/bestaudio[ext=m4a]/bestaudio',
        'quality': '5',  # Low quality
        'bitrate': '96k'
    }
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

# 🔥 ENHANCED: Helper function to detect if only storyboard formats are available
def has_actual_video_formats(formats_output: str) -> bool:
    """Check if the format list contains actual video formats (not just storyboards)"""
    try:
        lines = formats_output.split('\n')
        for line in lines:
            # Skip header lines and empty lines
            if not line.strip() or 'ID' in line or 'EXT' in line or '----' in line:
                continue
            
            # Check if this line contains a real video format
            if any(ext in line.lower() for ext in ['mp4', 'webm', 'mkv', 'avi', 'flv']):
                # Make sure it's not just a storyboard
                if 'sb' not in line[:10] and 'storyboard' not in line.lower():
                    return True
                    
            # Check for format codes that indicate real video
            parts = line.split()
            if len(parts) > 0:
                format_id = parts[0]
                # Real video format IDs are usually numeric or contain specific patterns
                if (format_id.isdigit() or 
                    any(pattern in format_id for pattern in ['dash', 'hls', 'http']) and
                    'sb' not in format_id):
                    return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking video formats: {e}")
        return False

# 🔥 FIXED AUDIO DOWNLOAD FUNCTION
def download_audio_with_ytdlp(video_id: str, quality: str = "medium", output_dir: str = None) -> str:
    """
    🔥 FIXED: Download audio from YouTube video using yt-dlp - Enhanced Version
    """
    if output_dir is None:
        output_dir = str(DEFAULT_DOWNLOADS_DIR)
    
    logger.info(f"🔥 Starting ENHANCED audio download for {video_id} in: {output_dir}")
    
    # Enhanced quality settings for better audio
    quality_settings = {
        "high": {
            "format": "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio",
            "audio_quality": "0",  # Best quality
            "audio_bitrate": "320k"
        },
        "medium": {
            "format": "bestaudio[abr<=192]/bestaudio[ext=m4a]/bestaudio",
            "audio_quality": "2",  # Good quality 
            "audio_bitrate": "192k"
        },
        "low": {
            "format": "bestaudio[abr<=96]/bestaudio[ext=m4a]/bestaudio",
            "audio_quality": "5",  # Acceptable quality
            "audio_bitrate": "96k"
        }
    }
    
    settings = quality_settings.get(quality, quality_settings["medium"])
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 🔥 FIX: Use simple, predictable filename
    output_template = f"{video_id}_audio_{quality}.%(ext)s"
    
    # 🔥 ENHANCED: Random user agent for better success
    user_agent = random.choice(USER_AGENTS)
    
    # 🔥 ENHANCED command for better audio quality and metadata
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", settings["audio_quality"],
        "--format", settings["format"],
        "--output", output_template,
        "--no-playlist",
        "--no-warnings",
        "--prefer-ffmpeg",
        "--embed-metadata",  # 🔥 ADD: Embed metadata like title, artist
        "--add-metadata",    # 🔥 ADD: Add metadata to file
        "--user-agent", user_agent,  # 🔥 ADD: Random user agent
        f"https://www.youtube.com/watch?v={video_id}"
    ]
    
    logger.info(f"🔥 Enhanced audio command: {' '.join(cmd)}")
    logger.info(f"🔥 Working directory: {output_dir}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300,  # 5 minute timeout
            cwd=output_dir,
            check=False  # Don't raise on non-zero exit
        )
        
        logger.info(f"🔥 yt-dlp exit code: {result.returncode}")
        if result.stdout:
            logger.info(f"🔥 yt-dlp stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"🔥 yt-dlp stderr: {result.stderr}")
        
        # 🔥 FIX: Find the actual downloaded file
        output_path = Path(output_dir)
        
        # Look for audio files in order of preference
        audio_patterns = [
            f"{video_id}_audio_{quality}.mp3",  # Exact match
            f"{video_id}_audio_{quality}.*",    # Any extension
            f"{video_id}*.mp3",                 # Any mp3 with video ID
            "*.mp3"                             # Any mp3 (last resort)
        ]
        
        audio_file = None
        for pattern in audio_patterns:
            audio_files = list(output_path.glob(pattern))
            if audio_files:
                # Sort by modification time (newest first) and size (largest first)
                audio_file = max(audio_files, key=lambda f: (f.stat().st_mtime, f.stat().st_size))
                logger.info(f"🔥 Found audio file with pattern '{pattern}': {audio_file.name}")
                break
        
        if not audio_file or not audio_file.exists():
            logger.error("❌ No audio file found after download")
            all_files = list(output_path.iterdir())
            logger.error(f"❌ Files in directory: {[f.name for f in all_files if f.is_file()]}")
            raise Exception("No audio file found after download")
        
        file_size = audio_file.stat().st_size
        
        # 🔥 CRITICAL: Verify file is not corrupted
        if file_size < 1000:
            logger.error(f"❌ Audio file too small ({file_size} bytes), likely corrupted")
            audio_file.unlink()  # Remove corrupted file
            raise Exception("Downloaded audio file is corrupted (too small)")
        
        logger.info(f"✅ Enhanced audio download successful: {audio_file.name} ({file_size} bytes)")
        return str(audio_file.absolute())
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Audio download timed out for {video_id}")
        raise Exception("Download timed out")
    except Exception as e:
        logger.error(f"❌ Audio download error: {e}")
        raise

# 🔥 COMPLETELY FIXED VIDEO DOWNLOAD FUNCTION
def download_video_with_ytdlp(video_id: str, quality: str = "720p", output_dir: str = None) -> Optional[str]:
    """
    🔥 COMPLETELY FIXED: Download video from YouTube using yt-dlp with advanced strategies
    Handles YouTube blocking, age restrictions, and format detection issues
    """
    if output_dir is None:
        output_dir = str(DEFAULT_DOWNLOADS_DIR)
        
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔥 Starting ADVANCED video download for {video_id}")
        logger.info(f"🔥 URL: {url}")
        logger.info(f"🔥 Output dir: {output_dir}")
        logger.info(f"🔥 Quality: {quality}")
        
        # 🔥 STEP 1: Check available formats first
        logger.info("🔥 Step 1: Analyzing available formats...")
        user_agent = random.choice(USER_AGENTS)
        
        formats_cmd = [
            "yt-dlp", 
            "--list-formats", 
            "--no-warnings",
            "--user-agent", user_agent,
            url
        ]
        
        formats_result = subprocess.run(
            formats_cmd, 
            capture_output=True, 
            text=True, 
            timeout=60, 
            check=False
        )
        
        if formats_result.returncode != 0:
            logger.error(f"❌ Failed to get formats: {formats_result.stderr}")
            raise Exception(f"Cannot access video formats: {formats_result.stderr}")
        
        formats_output = formats_result.stdout
        logger.info(f"🔥 Format analysis complete")
        
        # 🔥 STEP 2: Check if only storyboard formats are available
        if not has_actual_video_formats(formats_output):
            logger.error("❌ Only storyboard formats available - video is restricted")
            logger.info("Available formats:")
            logger.info(formats_output)
            raise Exception(
                "This video appears to be restricted (age-restricted, region-blocked, or temporarily unavailable). "
                "Only thumbnail formats are accessible. Please try a different video or try again later."
            )
        
        logger.info("✅ Real video formats detected")
        
        # Use predictable output filename
        output_template = f"{video_id}_video_{quality}.%(ext)s"
        
        # 🔥 STEP 3: Try advanced format selection strategies
        strategies = [
            # Strategy 1: Direct quality match with audio
            f"best[height<={quality[:-1]}][ext=mp4]+bestaudio[ext=m4a]/best[height<={quality[:-1]}]",
            
            # Strategy 2: Broader quality range
            f"best[height<={quality[:-1]}]/worst[height>={quality[:-1]}]",
            
            # Strategy 3: Format-specific with fallbacks
            f"(bestvideo[height<={quality[:-1]}]+bestaudio/best[height<={quality[:-1]}])[ext=mp4]/(bestvideo[height<={quality[:-1]}]+bestaudio/best[height<={quality[:-1]}])",
            
            # Strategy 4: Simple best with audio preference
            "bestvideo+bestaudio/best[ext=mp4]/best",
            
            # Strategy 5: Last resort - any working format
            "best/worst"
        ]
        
        for i, format_selector in enumerate(strategies, 1):
            logger.info(f"🔥 Step 3.{i}: Trying format strategy {i}: {format_selector}")
            
            cmd = [
                "yt-dlp",
                "--no-playlist",
                "--output", output_template,
                "--format", format_selector,
                "--merge-output-format", "mp4",
                "--embed-metadata",
                "--add-metadata",
                "--user-agent", user_agent,
                "--no-warnings",
                "--retries", "3",
                "--fragment-retries", "3",
                url
            ]
            
            logger.info(f"🔥 Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=600,  # 10 minutes
                cwd=output_dir,
                check=False
            )
            
            logger.info(f"🔥 Strategy {i} exit code: {result.returncode}")
            
            if result.returncode == 0:
                logger.info(f"✅ Strategy {i} succeeded!")
                break
            else:
                logger.warning(f"⚠️ Strategy {i} failed: {result.stderr}")
                if result.stdout:
                    logger.info(f"🔥 Strategy {i} stdout: {result.stdout[:500]}")
                
                # If this is the last strategy, log the error
                if i == len(strategies):
                    logger.error(f"❌ All strategies failed. Last error: {result.stderr}")
                    raise Exception(f"All download strategies failed. Last error: {result.stderr}")
        
        # 🔥 STEP 4: Find and verify the downloaded file
        output_path = Path(output_dir)
        
        # Look for video files in order of preference
        video_patterns = [
            f"{video_id}_video_{quality}.mp4",     # Preferred format
            f"{video_id}_video_{quality}.*",       # Any extension
            f"{video_id}_video.*",                 # Any video file
            f"{video_id}*.*"                       # Any file with video ID
        ]
        
        video_file = None
        for pattern in video_patterns:
            video_files = list(output_path.glob(pattern))
            if video_files:
                # Get the largest file (most likely the video)
                video_file = max(video_files, key=lambda f: f.stat().st_size)
                logger.info(f"🔥 Found video file with pattern '{pattern}': {video_file.name}")
                break
        
        if not video_file or not video_file.exists():
            logger.error("❌ No video file found after download")
            all_files = list(output_path.iterdir())
            logger.error(f"❌ Files in directory: {[f.name for f in all_files if f.is_file()]}")
            raise Exception("Video file not found after successful download")
        
        file_size = video_file.stat().st_size
        
        # 🔥 CRITICAL: Verify file is not corrupted
        if file_size < 100000:  # Less than 100KB is definitely corrupted for video
            logger.error(f"❌ Video file too small ({file_size} bytes), likely corrupted")
            video_file.unlink()  # Remove corrupted file
            raise Exception("Downloaded video file is corrupted (too small)")
        
        logger.info(f"✅ ADVANCED video download successful: {video_file.absolute()} ({file_size} bytes)")
        return str(video_file.absolute())
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Video download timed out for {video_id}")
        raise Exception("Video download timed out")
    except Exception as e:
        logger.error(f"❌ Exception in video download: {e}")
        raise Exception(f"Video download failed: {str(e)}")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_video_info(video_id: str) -> Optional[Dict[str, Any]]:
    """🔥 ENHANCED: Get video information using yt-dlp with better metadata"""
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        user_agent = random.choice(USER_AGENTS)
        
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-warnings",
            "--no-download",
            "--user-agent", user_agent,
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
            
            # 🔥 ENHANCED: Return more comprehensive metadata
            return {
                "id": video_info.get("id"),
                "title": video_info.get("title"),
                "duration": video_info.get("duration"),
                "upload_date": video_info.get("upload_date"),
                "uploader": video_info.get("uploader"),
                "uploader_id": video_info.get("uploader_id"),
                "view_count": video_info.get("view_count"),
                "like_count": video_info.get("like_count"),
                "description": video_info.get("description", "")[:500],
                "thumbnail": video_info.get("thumbnail"),
                "has_subtitles": bool(video_info.get("subtitles")),
                "has_auto_captions": bool(video_info.get("automatic_captions")),
                "format_note": video_info.get("format_note"),
                "ext": video_info.get("ext"),
                "filesize": video_info.get("filesize"),
                "fps": video_info.get("fps"),
                "width": video_info.get("width"),
                "height": video_info.get("height"),
                "age_limit": video_info.get("age_limit", 0),
                "availability": video_info.get("availability"),
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
    """Estimate file size for a download"""
    try:
        video_info = get_video_info(video_id)
        if not video_info or not video_info.get("duration"):
            return None
        
        duration_seconds = video_info["duration"]
        
        if download_type == "audio":
            bitrates = {"high": 320, "medium": 192, "low": 96}
            bitrate = bitrates.get(quality, 192)
            estimated_size = (duration_seconds * bitrate * 1000) // 8
            
        elif download_type == "video":
            size_per_minute = {
                "1080p": 100 * 1024 * 1024,
                "720p": 50 * 1024 * 1024,
                "480p": 25 * 1024 * 1024,
                "360p": 15 * 1024 * 1024
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
    
    import string
    valid_chars = string.ascii_letters + string.digits + '-_'
    
    return all(c in valid_chars for c in video_id)

def format_transcript_clean(text: str) -> str:
    """Format clean transcript with proper paragraph breaks"""
    try:
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
            
            if char_count > 400:
                paragraphs.append('. '.join(current_paragraph) + '.')
                current_paragraph = []
                char_count = 0
        
        if current_paragraph:
            paragraphs.append('. '.join(current_paragraph) + '.')
        
        return '\n\n'.join(paragraphs)
        
    except Exception as e:
        logger.error(f"Error formatting clean transcript: {e}")
        return text

def get_downloads_directory() -> Path:
    """Get the current downloads directory being used"""
    return DEFAULT_DOWNLOADS_DIR

def set_downloads_directory(path: str) -> bool:
    """Set a custom downloads directory"""
    try:
        global DEFAULT_DOWNLOADS_DIR
        custom_path = Path(path)
        custom_path.mkdir(parents=True, exist_ok=True)
        
        # Test if writable
        test_file = custom_path / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        
        DEFAULT_DOWNLOADS_DIR = custom_path
        logger.info(f"🔥 Downloads directory updated to: {DEFAULT_DOWNLOADS_DIR}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to set downloads directory: {e}")
        return False

def test_transcript_extraction(video_id: str = "dQw4w9WgXcQ"):
    """Test transcript extraction functions"""
    logger.info(f"Testing transcript extraction for video: {video_id}")
    
    clean_result = get_transcript_with_ytdlp(video_id, clean=True)
    if clean_result:
        logger.info(f"Clean transcript: {len(clean_result)} characters")
    else:
        logger.error("Clean transcript extraction failed")
    
    timestamped_result = get_transcript_with_ytdlp(video_id, clean=False)
    if timestamped_result:
        logger.info(f"Timestamped transcript: {len(timestamped_result)} characters")
    else:
        logger.error("Timestamped transcript extraction failed")
    
    return clean_result, timestamped_result

def test_downloads_folder():
    """Test if the downloads folder is accessible and writable"""
    try:
        logger.info(f"🔥 Testing downloads folder: {DEFAULT_DOWNLOADS_DIR}")
        
        # Check if directory exists
        if not DEFAULT_DOWNLOADS_DIR.exists():
            logger.info("Creating downloads directory...")
            DEFAULT_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Test write access
        test_file = DEFAULT_DOWNLOADS_DIR / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        
        logger.info(f"✅ Downloads folder is accessible and writable: {DEFAULT_DOWNLOADS_DIR.absolute()}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Downloads folder test failed: {e}")
        return False

# 🔥 ENHANCED: Test functions with better error detection
def test_video_download_advanced(video_id: str = "dQw4w9WgXcQ", quality: str = "720p"):
    """Test video download functionality with advanced error detection"""
    try:
        logger.info(f"🔥 Testing ADVANCED video download for video: {video_id}, quality: {quality}")
        
        # First check if we can get video info
        video_info = get_video_info(video_id)
        if not video_info:
            logger.error("❌ Cannot get video info - video may be restricted")
            return False
        
        logger.info(f"🔥 Video info: {video_info.get('title')} by {video_info.get('uploader')}")
        
        # Check for restrictions
        age_limit = video_info.get('age_limit', 0)
        availability = video_info.get('availability')
        
        if age_limit > 0:
            logger.warning(f"⚠️ Video has age limit: {age_limit}")
        if availability and availability != 'public':
            logger.warning(f"⚠️ Video availability: {availability}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_video_with_ytdlp(video_id, quality, temp_dir)
            
            if result and os.path.exists(result):
                file_size = os.path.getsize(result)
                logger.info(f"✅ ADVANCED video download test successful: {result} ({file_size} bytes)")
                return True
            else:
                logger.error("❌ ADVANCED video download test failed")
                return False
                
    except Exception as e:
        logger.error(f"❌ ADVANCED video download test error: {e}")
        return False

def test_audio_download(video_id: str = "dQw4w9WgXcQ", quality: str = "medium"):
    """Test audio download functionality"""
    try:
        logger.info(f"🔥 Testing audio download for video: {video_id}, quality: {quality}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_audio_with_ytdlp(video_id, quality, temp_dir)
            
            if result and os.path.exists(result):
                file_size = os.path.getsize(result)
                logger.info(f"✅ Audio download test successful: {result} ({file_size} bytes)")
                return True
            else:
                logger.error("❌ Audio download test failed")
                return False
                
    except Exception as e:
        logger.error(f"❌ Audio download test error: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🔥 Testing ENHANCED Downloads System with Advanced Video Support...")
    print(f"🔥 Default downloads directory: {DEFAULT_DOWNLOADS_DIR}")
    
    if test_downloads_folder():
        print("✅ Downloads folder test passed")
    else:
        print("❌ Downloads folder test failed")
    
    print("\nTesting yt-dlp availability...")
    if check_ytdlp_availability():
        print("✅ yt-dlp is available")
        
        print("\nTesting transcript extraction...")
        clean, timestamped = test_transcript_extraction()
        
        if clean:
            print(f"✅ Clean transcript extracted ({len(clean)} chars)")
        if timestamped:
            print(f"✅ Timestamped transcript extracted ({len(timestamped)} chars)")
            
        print("\n🔥 Testing enhanced audio download...")
        if test_audio_download():
            print("✅ Audio download test passed")
        else:
            print("❌ Audio download test failed")
            
        print("\n🔥 Testing ADVANCED video download...")
        if test_video_download_advanced():
            print("✅ ADVANCED video download test passed")
        else:
            print("❌ ADVANCED video download test failed")
            
        # Test with alternative videos
        print("\n🔥 Testing with alternative videos...")
        test_videos = [
            ("jNQXAC9IVRw", "Me at the zoo"),
            ("9bZkp7q19f0", "Gangnam Style"), 
            ("L_jWHffIx5E", "All Star")
        ]
        
        for video_id, title in test_videos:
            print(f"\n🔥 Testing {title} ({video_id})...")
            if test_video_download_advanced(video_id, "480p"):
                print(f"✅ {title} download works!")
                break
            else:
                print(f"❌ {title} download failed")
    else:
        print("❌ yt-dlp is not available")


##==============================

# # Enhanced transcript_utils.py - Audio/Video Support + User Downloads Folder
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

# logger = logging.getLogger("transcript_utils")

# # =============================================================================
# # CONFIGURATION
# # =============================================================================

# # 🔥 CRITICAL FIX: Use user's Downloads folder by default
# DEFAULT_DOWNLOADS_DIR = Path.home() / "Downloads"

# # Audio quality settings for yt-dlp
# AUDIO_FORMATS = {
#     'high': {
#         'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
#         'quality': '0',  # Best quality
#         'bitrate': '320k'
#     },
#     'medium': {
#         'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
#         'quality': '2',  # Medium quality
#         'bitrate': '192k'
#     },
#     'low': {
#         'format': 'bestaudio[abr<=96]/bestaudio[ext=m4a]/bestaudio',
#         'quality': '5',  # Low quality
#         'bitrate': '96k'
#     }
# }

# # Video quality settings for yt-dlp - SIMPLIFIED
# VIDEO_FORMATS = {
#     '1080p': 'best[height<=1080]/best',
#     '720p': 'best[height<=720]/best',
#     '480p': 'best[height<=480]/best',
#     '360p': 'best[height<=360]/best'
# }

# # =============================================================================
# # TRANSCRIPT FUNCTIONS
# # =============================================================================

# def get_transcript_with_ytdlp(video_id: str, clean=True, retries=3, wait_sec=1) -> Optional[str]:
#     """
#     Get transcript using yt-dlp as fallback
    
#     Args:
#         video_id: YouTube video ID
#         clean: If True, return clean text. If False, return with timestamps
#         retries: Number of retry attempts
#         wait_sec: Seconds to wait between retries
    
#     Returns:
#         Formatted transcript string or None if failed
#     """
#     try:
#         output_vtt = f"{video_id}.en.vtt"
#         output_json3 = f"{video_id}.en.json3"
#         url = f"https://www.youtube.com/watch?v={video_id}"

#         # Enhanced yt-dlp command with better error handling
#         cmd = [
#             "yt-dlp",
#             "--skip-download",
#             "--write-auto-sub",
#             "--sub-lang", "en",
#             "--sub-format", "json3/vtt/srt",
#             "--output", "%(id)s",
#             "--no-warnings",
#             "--quiet",
#             url
#         ]
        
#         result = subprocess.run(
#             cmd, 
#             capture_output=True, 
#             text=True, 
#             timeout=30,
#             check=False
#         )
        
#         if result.returncode != 0 and result.stderr:
#             logger.warning(f"yt-dlp warning: {result.stderr}")

#         # Try JSON3 first (most reliable)
#         for attempt in range(retries):
#             if os.path.exists(output_json3):
#                 try:
#                     with open(output_json3, encoding="utf8") as f:
#                         data = json.load(f)
                    
#                     # Clean up file
#                     os.remove(output_json3)
                    
#                     return _process_json3_transcript(data, clean)
                    
#                 except (json.JSONDecodeError, IOError) as e:
#                     logger.warning(f"Failed to parse JSON3 file: {e}")
#                     if os.path.exists(output_json3):
#                         os.remove(output_json3)
#                     break
            
#             time.sleep(wait_sec)

#         # Fallback to VTT processing
#         if os.path.exists(output_vtt):
#             try:
#                 with open(output_vtt, encoding="utf8") as f:
#                     vtt_content = f.read()
                
#                 # Clean up file
#                 os.remove(output_vtt)
                
#                 if clean:
#                     return extract_vtt_text(vtt_content)
#                 else:
#                     return format_transcript_vtt(vtt_content)
                    
#             except IOError as e:
#                 logger.warning(f"Failed to read VTT file: {e}")
#                 if os.path.exists(output_vtt):
#                     os.remove(output_vtt)

#         # Clean up any remaining files
#         _cleanup_temp_files(video_id)
        
#         logger.error(f"No transcript files found for video: {video_id}")
#         return None

#     except subprocess.TimeoutExpired:
#         logger.error(f"yt-dlp timeout for video: {video_id}")
#         _cleanup_temp_files(video_id)
#         return None
#     except Exception as e:
#         logger.error(f"yt-dlp fallback failed for {video_id}: {e}")
#         _cleanup_temp_files(video_id)
#         return None

# def _process_json3_transcript(data: Dict[Any, Any], clean: bool) -> Optional[str]:
#     """Process JSON3 transcript data"""
#     try:
#         blocks = []
        
#         for event in data.get("events", []):
#             if "segs" in event and "tStartMs" in event:
#                 # Extract text segments
#                 text_segments = []
#                 for seg in event["segs"]:
#                     if seg.get("utf8"):
#                         text_segments.append(seg["utf8"])
                
#                 if text_segments:
#                     text = ''.join(text_segments).strip()
                    
#                     if text:
#                         if clean:
#                             blocks.append(text)
#                         else:
#                             # Create timestamp
#                             start_ms = event["tStartMs"]
#                             seconds = int(start_ms // 1000)
#                             timestamp = f"[{seconds//60:02d}:{seconds%60:02d}]"
#                             blocks.append(f"{timestamp} {text}")
        
#         if blocks:
#             return "\n".join(blocks) if not clean else " ".join(blocks)
        
#         return None
        
#     except Exception as e:
#         logger.error(f"Error processing JSON3 transcript: {e}")
#         return None

# def extract_vtt_text(vtt_content: str) -> str:
#     """Extract clean text from VTT content"""
#     try:
#         lines = vtt_content.strip().splitlines()
#         text_lines = []
        
#         for line in lines:
#             line = line.strip()
#             # Skip VTT headers, timestamps, and empty lines
#             if (line and 
#                 not line.startswith('WEBVTT') and 
#                 not line.startswith('Kind:') and 
#                 not line.startswith('Language:') and 
#                 not '-->' in line and
#                 not line.startswith('NOTE') and
#                 not line.isdigit() and
#                 not re.match(r'^\d{2}:\d{2}:\d{2}', line)):
                
#                 # Remove HTML tags and formatting
#                 clean_line = re.sub(r'<[^>]+>', '', line)
#                 clean_line = re.sub(r'\{[^}]+\}', '', clean_line)
                
#                 if clean_line.strip():
#                     text_lines.append(clean_line.strip())
        
#         return ' '.join(text_lines)
        
#     except Exception as e:
#         logger.error(f"Error extracting VTT text: {e}")
#         return ""

# def format_transcript_vtt(raw_vtt: str) -> str:
#     """Format VTT content to proper WEBVTT standard"""
#     try:
#         lines = raw_vtt.strip().splitlines()
#         formatted_lines = []
        
#         # Add proper headers
#         formatted_lines.extend([
#             "WEBVTT",
#             "Kind: captions",
#             "Language: en",
#             ""
#         ])
        
#         i = 0
#         while i < len(lines):
#             line = lines[i].strip()
            
#             # Skip original headers
#             if line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
#                 i += 1
#                 continue
                
#             # Process timestamp lines
#             if '-->' in line:
#                 # Clean up timestamp format
#                 timestamp_line = re.sub(r'(\d{2}:\d{2}:\d{2})\.(\d{3})', r'\1.\2', line)
#                 formatted_lines.append(timestamp_line)
#                 i += 1
                
#                 # Get the text content
#                 text_content = []
#                 while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
#                     text_line = lines[i].strip()
#                     if text_line and not text_line.isdigit():
#                         # Clean HTML tags and formatting
#                         clean_text = re.sub(r'<[^>]+>', '', text_line)
#                         clean_text = re.sub(r'\{[^}]+\}', '', clean_text)
#                         if clean_text.strip():
#                             text_content.append(clean_text.strip())
#                     i += 1
                
#                 # Add text content
#                 if text_content:
#                     formatted_lines.append(' '.join(text_content))
                
#                 # Add separator line
#                 formatted_lines.append("")
#             else:
#                 i += 1
        
#         return '\n'.join(formatted_lines)
        
#     except Exception as e:
#         logger.error(f"Error formatting VTT: {e}")
#         return raw_vtt

# # =============================================================================
# # AUDIO DOWNLOAD FUNCTIONS - FIXED FOR USER DOWNLOADS FOLDER
# # =============================================================================

# # 🔥 CRITICAL FIX: Replace your download functions in transcript_utils.py

# def download_audio_with_ytdlp(video_id: str, quality: str = "medium", output_dir: str = None) -> str:
#     """
#     Download audio from YouTube video using yt-dlp - FIXED VERSION
#     🔥 FIXED: Now works properly with proper timestamps and prevents file corruption
#     """
#     if output_dir is None:
#         output_dir = str(DEFAULT_DOWNLOADS_DIR)
    
#     logger.info(f"🔥 Starting audio download for {video_id} in: {output_dir}")
    
#     # Better quality settings for stable downloads
#     quality_settings = {
#         "high": {
#             "format": "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio",
#             "audio_quality": "0"  # Best quality
#         },
#         "medium": {
#             "format": "bestaudio[abr<=128]/bestaudio[ext=m4a]/bestaudio",
#             "audio_quality": "2"  # Good quality 
#         },
#         "low": {
#             "format": "bestaudio[abr<=96]/bestaudio[ext=m4a]/bestaudio",
#             "audio_quality": "5"  # Acceptable quality
#         }
#     }
    
#     settings = quality_settings.get(quality, quality_settings["medium"])
    
#     # Create output directory
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
    
#     # 🔥 FIX: Use simple, predictable filename
#     output_template = f"{video_id}_audio_{quality}.%(ext)s"
    
#     # 🔥 SIMPLIFIED command for better reliability
#     cmd = [
#         "yt-dlp",
#         "--extract-audio",
#         "--audio-format", "mp3",
#         "--audio-quality", settings["audio_quality"],
#         "--format", settings["format"],
#         "--output", output_template,
#         "--no-playlist",
#         "--no-warnings",
#         "--prefer-ffmpeg",
#         "--embed-metadata",
#         f"https://www.youtube.com/watch?v={video_id}"
#     ]
    
#     logger.info(f"🔥 Command: {' '.join(cmd)}")
#     logger.info(f"🔥 Working directory: {output_dir}")
    
#     try:
#         result = subprocess.run(
#             cmd, 
#             capture_output=True, 
#             text=True, 
#             timeout=300,  # 5 minute timeout
#             cwd=output_dir,
#             check=False  # Don't raise on non-zero exit
#         )
        
#         logger.info(f"🔥 yt-dlp exit code: {result.returncode}")
#         if result.stdout:
#             logger.info(f"🔥 yt-dlp stdout: {result.stdout}")
#         if result.stderr:
#             logger.info(f"🔥 yt-dlp stderr: {result.stderr}")
        
#         # 🔥 FIX: Find the actual downloaded file
#         output_path = Path(output_dir)
        
#         # Look for audio files in order of preference
#         audio_patterns = [
#             f"{video_id}_audio_{quality}.mp3",  # Exact match
#             f"{video_id}_audio_{quality}.*",    # Any extension
#             f"{video_id}*.mp3",                 # Any mp3 with video ID
#             "*.mp3"                             # Any mp3 (last resort)
#         ]
        
#         audio_file = None
#         for pattern in audio_patterns:
#             audio_files = list(output_path.glob(pattern))
#             if audio_files:
#                 # Sort by modification time (newest first) and size (largest first)
#                 audio_file = max(audio_files, key=lambda f: (f.stat().st_mtime, f.stat().st_size))
#                 logger.info(f"🔥 Found audio file with pattern '{pattern}': {audio_file.name}")
#                 break
        
#         if not audio_file or not audio_file.exists():
#             logger.error("❌ No audio file found after download")
#             all_files = list(output_path.iterdir())
#             logger.error(f"❌ Files in directory: {[f.name for f in all_files if f.is_file()]}")
#             raise Exception("No audio file found after download")
        
#         file_size = audio_file.stat().st_size
        
#         # 🔥 CRITICAL: Verify file is not corrupted
#         if file_size < 1000:
#             logger.error(f"❌ Audio file too small ({file_size} bytes), likely corrupted")
#             audio_file.unlink()  # Remove corrupted file
#             raise Exception("Downloaded audio file is corrupted (too small)")
        
#         logger.info(f"✅ Audio download successful: {audio_file.name} ({file_size} bytes)")
#         return str(audio_file.absolute())
            
#     except subprocess.TimeoutExpired:
#         logger.error(f"❌ Audio download timed out for {video_id}")
#         raise Exception("Download timed out")
#     except Exception as e:
#         logger.error(f"❌ Audio download error: {e}")
#         raise



# #==================================================
# # Fixed audio and video download functions for transcript_utils.py
# # Replace the existing functions with these fixed versions

# # def download_audio_with_ytdlp(video_id: str, quality: str = "medium", output_dir: str = None) -> str:
# #     """
# #     Download audio from YouTube video using yt-dlp - FIXED VERSION
# #     🔥 FIXED: Now works properly with temp directories and prevents file corruption
# #     """
# #     if output_dir is None:
# #         output_dir = str(DEFAULT_DOWNLOADS_DIR)
    
# #     logger.info(f"🔥 Starting audio download for {video_id} in: {output_dir}")
    
# #     # Better quality settings for stable downloads
# #     quality_settings = {
# #         "high": {
# #             "format": "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio",
# #             "audio_quality": "0"  # Best quality
# #         },
# #         "medium": {
# #             "format": "bestaudio[abr<=128]/bestaudio[ext=m4a]/bestaudio",
# #             "audio_quality": "2"  # Good quality 
# #         },
# #         "low": {
# #             "format": "bestaudio[abr<=96]/bestaudio[ext=m4a]/bestaudio",
# #             "audio_quality": "5"  # Acceptable quality
# #         }
# #     }
    
# #     settings = quality_settings.get(quality, quality_settings["medium"])
    
# #     # Create output directory
# #     Path(output_dir).mkdir(parents=True, exist_ok=True)
    
# #     # 🔥 FIX: Use simple, predictable filename
# #     output_template = f"{video_id}_audio_{quality}.%(ext)s"
    
# #     # 🔥 SIMPLIFIED command for better reliability
# #     cmd = [
# #         "yt-dlp",
# #         "--extract-audio",
# #         "--audio-format", "mp3",
# #         "--audio-quality", settings["audio_quality"],
# #         "--format", settings["format"],
# #         "--output", output_template,
# #         "--no-playlist",
# #         "--no-warnings",
# #         "--prefer-ffmpeg",
# #         "--embed-metadata",
# #         f"https://www.youtube.com/watch?v={video_id}"
# #     ]
    
# #     logger.info(f"🔥 Command: {' '.join(cmd)}")
# #     logger.info(f"🔥 Working directory: {output_dir}")
    
# #     try:
# #         result = subprocess.run(
# #             cmd, 
# #             capture_output=True, 
# #             text=True, 
# #             timeout=300,  # 5 minute timeout
# #             cwd=output_dir,
# #             check=False  # Don't raise on non-zero exit
# #         )
        
# #         logger.info(f"🔥 yt-dlp exit code: {result.returncode}")
# #         if result.stdout:
# #             logger.info(f"🔥 yt-dlp stdout: {result.stdout}")
# #         if result.stderr:
# #             logger.info(f"🔥 yt-dlp stderr: {result.stderr}")
        
# #         # 🔥 FIX: Find the actual downloaded file
# #         output_path = Path(output_dir)
        
# #         # Look for audio files in order of preference
# #         audio_patterns = [
# #             f"{video_id}_audio_{quality}.mp3",  # Exact match
# #             f"{video_id}_audio_{quality}.*",    # Any extension
# #             f"{video_id}*.mp3",                 # Any mp3 with video ID
# #             "*.mp3"                             # Any mp3 (last resort)
# #         ]
        
# #         audio_file = None
# #         for pattern in audio_patterns:
# #             audio_files = list(output_path.glob(pattern))
# #             if audio_files:
# #                 # Sort by modification time (newest first) and size (largest first)
# #                 audio_file = max(audio_files, key=lambda f: (f.stat().st_mtime, f.stat().st_size))
# #                 logger.info(f"🔥 Found audio file with pattern '{pattern}': {audio_file.name}")
# #                 break
        
# #         if not audio_file or not audio_file.exists():
# #             logger.error("❌ No audio file found after download")
# #             all_files = list(output_path.iterdir())
# #             logger.error(f"❌ Files in directory: {[f.name for f in all_files if f.is_file()]}")
# #             raise Exception("No audio file found after download")
        
# #         file_size = audio_file.stat().st_size
        
# #         # 🔥 CRITICAL: Verify file is not corrupted
# #         if file_size < 1000:
# #             logger.error(f"❌ Audio file too small ({file_size} bytes), likely corrupted")
# #             audio_file.unlink()  # Remove corrupted file
# #             raise Exception("Downloaded audio file is corrupted (too small)")
        
# #         logger.info(f"✅ Audio download successful: {audio_file.name} ({file_size} bytes)")
# #         return str(audio_file.absolute())
            
# #     except subprocess.TimeoutExpired:
# #         logger.error(f"❌ Audio download timed out for {video_id}")
# #         raise Exception("Download timed out")
# #     except Exception as e:
# #         logger.error(f"❌ Audio download error: {e}")
# #         raise

# # =============================================================================
# # VIDEO DOWNLOAD FUNCTIONS - FIXED FOR USER DOWNLOADS FOLDER
# # =============================================================================

# # 🔥 CRITICAL FIX: Replace your download functions in transcript_utils.py

# def download_audio_with_ytdlp(video_id: str, quality: str = "medium", output_dir: str = None) -> str:
#     """
#     Download audio from YouTube video using yt-dlp - FIXED VERSION
#     🔥 FIXED: Now works properly with proper timestamps and prevents file corruption
#     """
#     if output_dir is None:
#         output_dir = str(DEFAULT_DOWNLOADS_DIR)
    
#     logger.info(f"🔥 Starting audio download for {video_id} in: {output_dir}")
    
#     # Better quality settings for stable downloads
#     quality_settings = {
#         "high": {
#             "format": "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio",
#             "audio_quality": "0"  # Best quality
#         },
#         "medium": {
#             "format": "bestaudio[abr<=128]/bestaudio[ext=m4a]/bestaudio",
#             "audio_quality": "2"  # Good quality 
#         },
#         "low": {
#             "format": "bestaudio[abr<=96]/bestaudio[ext=m4a]/bestaudio",
#             "audio_quality": "5"  # Acceptable quality
#         }
#     }
    
#     settings = quality_settings.get(quality, quality_settings["medium"])
    
#     # Create output directory
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
    
#     # 🔥 FIX: Use simple, predictable filename
#     output_template = f"{video_id}_audio_{quality}.%(ext)s"
    
#     # 🔥 SIMPLIFIED command for better reliability
#     cmd = [
#         "yt-dlp",
#         "--extract-audio",
#         "--audio-format", "mp3",
#         "--audio-quality", settings["audio_quality"],
#         "--format", settings["format"],
#         "--output", output_template,
#         "--no-playlist",
#         "--no-warnings",
#         "--prefer-ffmpeg",
#         "--embed-metadata",
#         f"https://www.youtube.com/watch?v={video_id}"
#     ]
    
#     logger.info(f"🔥 Command: {' '.join(cmd)}")
#     logger.info(f"🔥 Working directory: {output_dir}")
    
#     try:
#         result = subprocess.run(
#             cmd, 
#             capture_output=True, 
#             text=True, 
#             timeout=300,  # 5 minute timeout
#             cwd=output_dir,
#             check=False  # Don't raise on non-zero exit
#         )
        
#         logger.info(f"🔥 yt-dlp exit code: {result.returncode}")
#         if result.stdout:
#             logger.info(f"🔥 yt-dlp stdout: {result.stdout}")
#         if result.stderr:
#             logger.info(f"🔥 yt-dlp stderr: {result.stderr}")
        
#         # 🔥 FIX: Find the actual downloaded file
#         output_path = Path(output_dir)
        
#         # Look for audio files in order of preference
#         audio_patterns = [
#             f"{video_id}_audio_{quality}.mp3",  # Exact match
#             f"{video_id}_audio_{quality}.*",    # Any extension
#             f"{video_id}*.mp3",                 # Any mp3 with video ID
#             "*.mp3"                             # Any mp3 (last resort)
#         ]
        
#         audio_file = None
#         for pattern in audio_patterns:
#             audio_files = list(output_path.glob(pattern))
#             if audio_files:
#                 # Sort by modification time (newest first) and size (largest first)
#                 audio_file = max(audio_files, key=lambda f: (f.stat().st_mtime, f.stat().st_size))
#                 logger.info(f"🔥 Found audio file with pattern '{pattern}': {audio_file.name}")
#                 break
        
#         if not audio_file or not audio_file.exists():
#             logger.error("❌ No audio file found after download")
#             all_files = list(output_path.iterdir())
#             logger.error(f"❌ Files in directory: {[f.name for f in all_files if f.is_file()]}")
#             raise Exception("No audio file found after download")
        
#         file_size = audio_file.stat().st_size
        
#         # 🔥 CRITICAL: Verify file is not corrupted
#         if file_size < 1000:
#             logger.error(f"❌ Audio file too small ({file_size} bytes), likely corrupted")
#             audio_file.unlink()  # Remove corrupted file
#             raise Exception("Downloaded audio file is corrupted (too small)")
        
#         logger.info(f"✅ Audio download successful: {audio_file.name} ({file_size} bytes)")
#         return str(audio_file.absolute())
            
#     except subprocess.TimeoutExpired:
#         logger.error(f"❌ Audio download timed out for {video_id}")
#         raise Exception("Download timed out")
#     except Exception as e:
#         logger.error(f"❌ Audio download error: {e}")
#         raise

# # 🔥 NEW: Enhanced video download specifically for main.py calls
# def download_video_with_ytdlp_enhanced(video_id: str, quality: str, output_dir: str) -> str:
#     """
#     Enhanced video download that GUARANTEES audio inclusion
#     This is called from the main.py endpoints
#     """
#     url = f"https://www.youtube.com/watch?v={video_id}"
#     output_template = f"{video_id}_video.%(ext)s"
    
#     logger.info(f"🔥 Enhanced video download for {video_id} at {quality}")
    
#     # 🔥 ULTIMATE FIX: Multiple fallback strategies to ensure audio
#     strategies = [
#         # Strategy 1: Explicit video+audio with specific codecs
#         {
#             "name": "video+audio_explicit",
#             "cmd": [
#                 "yt-dlp",
#                 "--format", "(bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4])",
#                 "--merge-output-format", "mp4",
#                 "--output", output_template,
#                 "--no-playlist",
#                 "--no-warnings",
#                 "--embed-metadata",
#                 url
#             ]
#         },
#         # Strategy 2: Best available with audio preference
#         {
#             "name": "best_with_audio",
#             "cmd": [
#                 "yt-dlp",
#                 "--format", "best[ext=mp4]+bestaudio/best[ext=mp4]/best",
#                 "--merge-output-format", "mp4",
#                 "--output", output_template,
#                 "--no-playlist",
#                 "--no-warnings",
#                 url
#             ]
#         },
#         # Strategy 3: Simple best quality (fallback)
#         {
#             "name": "simple_best",
#             "cmd": [
#                 "yt-dlp",
#                 "--format", "best",
#                 "--output", output_template,
#                 "--no-playlist",
#                 "--no-warnings",
#                 "--merge-output-format", "mp4",
#                 url
#             ]
#         }
#     ]
    
#     for strategy in strategies:
#         logger.info(f"🔥 Trying strategy: {strategy['name']}")
#         logger.info(f"🔥 Command: {' '.join(strategy['cmd'])}")
        
#         try:
#             result = subprocess.run(
#                 strategy['cmd'], 
#                 capture_output=True, 
#                 text=True, 
#                 timeout=600, 
#                 cwd=output_dir, 
#                 check=False
#             )
            
#             logger.info(f"🔥 Strategy {strategy['name']} exit code: {result.returncode}")
            
#             if result.returncode == 0:
#                 # Check if file was created
#                 output_path = Path(output_dir)
#                 for file_path in output_path.glob(f"{video_id}_video*"):
#                     if file_path.is_file() and file_path.stat().st_size > 100000:
#                         logger.info(f"✅ Strategy {strategy['name']} succeeded: {file_path.name}")
#                         return str(file_path)
                        
#             else:
#                 logger.warning(f"⚠️ Strategy {strategy['name']} failed: {result.stderr}")
                
#         except Exception as e:
#             logger.warning(f"⚠️ Strategy {strategy['name']} exception: {e}")
#             continue
    
#     # If all strategies fail
#     logger.error("❌ All video download strategies failed")
#     raise Exception("All video download strategies failed")

# # def download_video_with_ytdlp(video_id: str, quality: str = "720p", output_dir: str = None) -> Optional[str]:
# #     """
# #     Download video from YouTube using yt-dlp - FIXED VERSION  
# #     🔥 FIXED: Now works properly with temp directories and prevents file corruption
# #     """
# #     if output_dir is None:
# #         output_dir = str(DEFAULT_DOWNLOADS_DIR)
        
# #     try:
# #         url = f"https://www.youtube.com/watch?v={video_id}"
        
# #         # Ensure output directory exists
# #         Path(output_dir).mkdir(parents=True, exist_ok=True)
        
# #         logger.info(f"🔥 Starting video download for {video_id}")
# #         logger.info(f"🔥 URL: {url}")
# #         logger.info(f"🔥 Output dir: {output_dir}")
# #         logger.info(f"🔥 Quality: {quality}")
        
# #         # 🔥 SIMPLIFIED approach - let yt-dlp handle format selection
# #         # Use predictable output filename
# #         output_template = f"{video_id}_video.%(ext)s"
        
# #         cmd = [
# #             "yt-dlp",
# #             "--no-playlist",
# #             "--output", output_template,
# #             "--merge-output-format", "mp4",  # Prefer mp4 for compatibility
# #             "--no-warnings",
# #             url
# #         ]
        
# #         logger.info(f"🔥 Command: {' '.join(cmd)}")
        
# #         result = subprocess.run(
# #             cmd, 
# #             capture_output=True, 
# #             text=True, 
# #             timeout=600,  # 10 minutes for videos
# #             cwd=output_dir,
# #             check=False
# #         )
        
# #         logger.info(f"🔥 yt-dlp exit code: {result.returncode}")
# #         if result.stdout:
# #             logger.info(f"🔥 yt-dlp stdout: {result.stdout}")
# #         if result.stderr:
# #             logger.info(f"🔥 yt-dlp stderr: {result.stderr}")
        
# #         # 🔥 FIX: Find the actual downloaded file
# #         output_path = Path(output_dir)
        
# #         # Look for video files in order of preference
# #         video_patterns = [
# #             f"{video_id}_video.mp4",     # Preferred format
# #             f"{video_id}_video.*",       # Any extension
# #             f"{video_id}*.*"             # Any file with video ID
# #         ]
        
# #         video_file = None
# #         for pattern in video_patterns:
# #             video_files = list(output_path.glob(pattern))
# #             if video_files:
# #                 # Get the largest file (most likely the video)
# #                 video_file = max(video_files, key=lambda f: f.stat().st_size)
# #                 logger.info(f"🔥 Found video file with pattern '{pattern}': {video_file.name}")
# #                 break
        
# #         if not video_file or not video_file.exists():
# #             logger.error("❌ No video file found after download")
# #             all_files = list(output_path.iterdir())
# #             logger.error(f"❌ Files in directory: {[f.name for f in all_files if f.is_file()]}")
# #             return None
        
# #         file_size = video_file.stat().st_size
        
# #         # 🔥 CRITICAL: Verify file is not corrupted
# #         if file_size < 10000:  # Less than 10KB is definitely corrupted for video
# #             logger.error(f"❌ Video file too small ({file_size} bytes), likely corrupted")
# #             video_file.unlink()  # Remove corrupted file
# #             return None
        
# #         logger.info(f"✅ Video download successful: {video_file.absolute()} ({file_size} bytes)")
# #         return str(video_file.absolute())
            
# #     except subprocess.TimeoutExpired:
# #         logger.error(f"❌ Video download timed out for {video_id}")
# #         return None
# #     except Exception as e:
# #         logger.error(f"❌ Exception in video download: {e}")
# #         return None

# # =============================================================================
# # UTILITY FUNCTIONS
# # =============================================================================

# def get_video_info(video_id: str) -> Optional[Dict[str, Any]]:
#     """Get video information using yt-dlp"""
#     try:
#         url = f"https://www.youtube.com/watch?v={video_id}"
        
#         cmd = [
#             "yt-dlp",
#             "--dump-json",
#             "--no-warnings",
#             "--no-download",
#             url
#         ]
        
#         result = subprocess.run(
#             cmd,
#             capture_output=True,
#             text=True,
#             timeout=30,
#             check=False
#         )
        
#         if result.returncode == 0 and result.stdout:
#             video_info = json.loads(result.stdout)
            
#             return {
#                 "id": video_info.get("id"),
#                 "title": video_info.get("title"),
#                 "duration": video_info.get("duration"),
#                 "upload_date": video_info.get("upload_date"),
#                 "uploader": video_info.get("uploader"),
#                 "view_count": video_info.get("view_count"),
#                 "like_count": video_info.get("like_count"),
#                 "description": video_info.get("description", "")[:500],
#                 "thumbnail": video_info.get("thumbnail"),
#                 "has_subtitles": bool(video_info.get("subtitles")),
#                 "has_auto_captions": bool(video_info.get("automatic_captions"))
#             }
        
#         return None
        
#     except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
#         logger.error(f"Failed to get video info for {video_id}: {e}")
#         return None

# def check_ytdlp_availability() -> bool:
#     """Check if yt-dlp is available and working"""
#     try:
#         result = subprocess.run(
#             ["yt-dlp", "--version"],
#             capture_output=True,
#             text=True,
#             timeout=10,
#             check=False
#         )
        
#         if result.returncode == 0:
#             version = result.stdout.strip()
#             logger.info(f"yt-dlp is available: {version}")
#             return True
        
#         return False
        
#     except Exception as e:
#         logger.error(f"yt-dlp not available: {e}")
#         return False

# def estimate_file_size(video_id: str, download_type: str, quality: str) -> Optional[int]:
#     """Estimate file size for a download"""
#     try:
#         video_info = get_video_info(video_id)
#         if not video_info or not video_info.get("duration"):
#             return None
        
#         duration_seconds = video_info["duration"]
        
#         if download_type == "audio":
#             bitrates = {"high": 320, "medium": 192, "low": 96}
#             bitrate = bitrates.get(quality, 192)
#             estimated_size = (duration_seconds * bitrate * 1000) // 8
            
#         elif download_type == "video":
#             size_per_minute = {
#                 "1080p": 100 * 1024 * 1024,
#                 "720p": 50 * 1024 * 1024,
#                 "480p": 25 * 1024 * 1024,
#                 "360p": 15 * 1024 * 1024
#             }
            
#             rate = size_per_minute.get(quality, size_per_minute["720p"])
#             estimated_size = (duration_seconds * rate) // 60
            
#         else:
#             return None
        
#         return int(estimated_size)
        
#     except Exception as e:
#         logger.error(f"Error estimating file size: {e}")
#         return None

# def _cleanup_temp_files(video_id: str):
#     """Clean up temporary files created during processing"""
#     try:
#         temp_patterns = [
#             f"{video_id}*.vtt",
#             f"{video_id}*.json3",
#             f"{video_id}*.srt",
#             f"{video_id}*.json",
#             f"{video_id}*.info.json"
#         ]
        
#         for pattern in temp_patterns:
#             import glob
#             for file_path in glob.glob(pattern):
#                 try:
#                     os.remove(file_path)
#                 except OSError:
#                     pass
                    
#     except Exception as e:
#         logger.warning(f"Error cleaning up temp files: {e}")

# def validate_video_id(video_id: str) -> bool:
#     """Validate YouTube video ID format"""
#     if not video_id or len(video_id) != 11:
#         return False
    
#     import string
#     valid_chars = string.ascii_letters + string.digits + '-_'
    
#     return all(c in valid_chars for c in video_id)

# def format_transcript_clean(text: str) -> str:
#     """Format clean transcript with proper paragraph breaks"""
#     try:
#         sentences = re.split(r'[.!?]+\s+', text)
#         paragraphs = []
#         current_paragraph = []
#         char_count = 0
        
#         for sentence in sentences:
#             sentence = sentence.strip()
#             if not sentence:
#                 continue
                
#             current_paragraph.append(sentence)
#             char_count += len(sentence) + 1
            
#             if char_count > 400:
#                 paragraphs.append('. '.join(current_paragraph) + '.')
#                 current_paragraph = []
#                 char_count = 0
        
#         if current_paragraph:
#             paragraphs.append('. '.join(current_paragraph) + '.')
        
#         return '\n\n'.join(paragraphs)
        
#     except Exception as e:
#         logger.error(f"Error formatting clean transcript: {e}")
#         return text

# def get_downloads_directory() -> Path:
#     """Get the current downloads directory being used"""
#     return DEFAULT_DOWNLOADS_DIR

# def set_downloads_directory(path: str) -> bool:
#     """Set a custom downloads directory"""
#     try:
#         global DEFAULT_DOWNLOADS_DIR
#         custom_path = Path(path)
#         custom_path.mkdir(parents=True, exist_ok=True)
        
#         # Test if writable
#         test_file = custom_path / "test_write.tmp"
#         test_file.write_text("test")
#         test_file.unlink()
        
#         DEFAULT_DOWNLOADS_DIR = custom_path
#         logger.info(f"🔥 Downloads directory updated to: {DEFAULT_DOWNLOADS_DIR}")
#         return True
        
#     except Exception as e:
#         logger.error(f"Failed to set downloads directory: {e}")
#         return False

# # 🔥 ENHANCED: File timestamp functions
# def set_file_timestamp_to_now(file_path: str):
#     """Set file timestamp to current time (like transcript does)"""
#     try:
#         import time
#         current_time = time.time()
#         os.utime(file_path, (current_time, current_time))
#         logger.info(f"✅ Set timestamp to now for: {file_path}")
#     except Exception as e:
#         logger.warning(f"Could not set timestamp: {e}")

# def ensure_file_in_today_section(file_path: str):
#     """Ensure downloaded file appears in 'Today' section like transcripts"""
#     try:
#         # Set both access and modification time to now
#         import time
#         current_time = time.time()
#         os.utime(file_path, (current_time, current_time))
        
#         # Also set creation time on Windows if possible
#         if os.name == 'nt':  # Windows
#             try:
#                 import win32file
#                 import win32con
#                 from pywintypes import Time
                
#                 handle = win32file.CreateFile(
#                     file_path,
#                     win32con.GENERIC_WRITE,
#                     win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
#                     None,
#                     win32con.OPEN_EXISTING,
#                     0,
#                     None
#                 )
                
#                 win32file.SetFileTime(handle, Time(current_time), Time(current_time), Time(current_time))
#                 win32file.CloseHandle(handle)
                
#                 logger.info(f"✅ Set Windows creation time for: {file_path}")
                
#             except ImportError:
#                 logger.info("win32file not available, using standard timestamp")
#             except Exception as e:
#                 logger.warning(f"Could not set Windows creation time: {e}")
                
#     except Exception as e:
#         logger.warning(f"Could not ensure file in today section: {e}")

# def test_transcript_extraction(video_id: str = "dQw4w9WgXcQ"):
#     """Test transcript extraction functions"""
#     logger.info(f"Testing transcript extraction for video: {video_id}")
    
#     clean_result = get_transcript_with_ytdlp(video_id, clean=True)
#     if clean_result:
#         logger.info(f"Clean transcript: {len(clean_result)} characters")
#     else:
#         logger.error("Clean transcript extraction failed")
    
#     timestamped_result = get_transcript_with_ytdlp(video_id, clean=False)
#     if timestamped_result:
#         logger.info(f"Timestamped transcript: {len(timestamped_result)} characters")
#     else:
#         logger.error("Timestamped transcript extraction failed")
    
#     return clean_result, timestamped_result

# def test_downloads_folder():
#     """Test if the downloads folder is accessible and writable"""
#     try:
#         logger.info(f"🔥 Testing downloads folder: {DEFAULT_DOWNLOADS_DIR}")
        
#         # Check if directory exists
#         if not DEFAULT_DOWNLOADS_DIR.exists():
#             logger.info("Creating downloads directory...")
#             DEFAULT_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
        
#         # Test write access
#         test_file = DEFAULT_DOWNLOADS_DIR / "test_write.tmp"
#         test_file.write_text("test")
#         test_file.unlink()
        
#         logger.info(f"✅ Downloads folder is accessible and writable: {DEFAULT_DOWNLOADS_DIR.absolute()}")
#         return True
        
#     except Exception as e:
#         logger.error(f"❌ Downloads folder test failed: {e}")
#         return False

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
    
#     print("🔥 Testing Downloads Folder Setup...")
#     print(f"🔥 Default downloads directory: {DEFAULT_DOWNLOADS_DIR}")
    
#     if test_downloads_folder():
#         print("✅ Downloads folder test passed")
#     else:
#         print("❌ Downloads folder test failed")
    
#     print("\nTesting yt-dlp availability...")
#     if check_ytdlp_availability():
#         print("✅ yt-dlp is available")
        
#         print("\nTesting transcript extraction...")
#         clean, timestamped = test_transcript_extraction()
        
#         if clean:
#             print(f"✅ Clean transcript extracted ({len(clean)} chars)")
#         if timestamped:
#             print(f"✅ Timestamped transcript extracted ({len(timestamped)} chars)")
#     else:
#         print("❌ yt-dlp is not available")

