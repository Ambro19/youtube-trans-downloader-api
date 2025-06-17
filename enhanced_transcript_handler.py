# Create enhanced_transcript_handler.py in your backend directory

import logging
import re
from typing import List, Dict, Optional, Tuple
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from youtube_transcript_api.formatters import TextFormatter
import requests
import json
from urllib.parse import parse_qs, urlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTranscriptHandler:
    def __init__(self):
        self.text_formatter = TextFormatter()
        
    def extract_video_id(self, url_or_id: str) -> str:
        """Extract video ID from various YouTube URL formats"""
        if not url_or_id:
            raise ValueError("URL or ID is required")
        
        # If it's already a video ID (11 characters, alphanumeric)
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id.strip()):
            return url_or_id.strip()
        
        # Extract from various YouTube URL formats
        patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract video ID from: {url_or_id}")
    
    def get_transcript_with_fallbacks(self, video_id: str) -> Tuple[List[Dict], str]:
        """
        Try multiple methods to get transcript with comprehensive error handling
        Returns: (transcript_data, method_used)
        """
        methods = [
            ("primary_api", self._get_transcript_primary),
            ("alternative_languages", self._get_transcript_alternative_languages),
            ("auto_generated", self._get_transcript_auto_generated),
            ("debug_method", self._debug_transcript_availability)
        ]
        
        last_error = None
        
        for method_name, method_func in methods:
            try:
                logger.info(f"Trying method: {method_name} for video {video_id}")
                result = method_func(video_id)
                if result:
                    logger.info(f"✅ Success with method: {method_name}")
                    return result, method_name
            except Exception as e:
                logger.warning(f"❌ Method {method_name} failed: {str(e)}")
                last_error = e
                continue
        
        # If all methods fail, raise the last error with context
        raise Exception(f"All transcript methods failed. Last error: {str(last_error)}")
    
    def _get_transcript_primary(self, video_id: str) -> List[Dict]:
        """Primary method using YouTubeTranscriptApi"""
        try:
            # Try to get transcript in preferred languages
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=['en', 'en-US', 'en-GB']
            )
            return transcript
        except Exception as e:
            logger.error(f"Primary method failed: {e}")
            raise
    
    def _get_transcript_alternative_languages(self, video_id: str) -> List[Dict]:
        """Try to get transcript in any available language"""
        try:
            # Get list of available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try manually created transcripts first
            for transcript in transcript_list:
                if not transcript.is_generated:
                    logger.info(f"Found manual transcript in language: {transcript.language}")
                    return transcript.fetch()
            
            # Fall back to auto-generated transcripts
            for transcript in transcript_list:
                if transcript.is_generated:
                    logger.info(f"Found auto-generated transcript in language: {transcript.language}")
                    return transcript.fetch()
                    
        except Exception as e:
            logger.error(f"Alternative languages method failed: {e}")
            raise
    
    def _get_transcript_auto_generated(self, video_id: str) -> List[Dict]:
        """Specifically try to get auto-generated transcripts"""
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Look for auto-generated transcripts
            for transcript in transcript_list:
                if transcript.is_generated:
                    logger.info(f"Using auto-generated transcript: {transcript.language}")
                    return transcript.fetch()
            
            raise NoTranscriptFound(video_id)
            
        except Exception as e:
            logger.error(f"Auto-generated method failed: {e}")
            raise
    
    def _debug_transcript_availability(self, video_id: str) -> Optional[List[Dict]]:
        """Debug method to check what transcripts are available"""
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            available_transcripts = []
            for transcript in transcript_list:
                transcript_info = {
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                }
                available_transcripts.append(transcript_info)
                logger.info(f"Available transcript: {transcript_info}")
            
            # If we found any transcripts, try the first one
            if available_transcripts:
                first_transcript = transcript_list._transcripts[0]
                return first_transcript.fetch()
                
        except Exception as e:
            logger.error(f"Debug method failed: {e}")
            raise
    
    def check_video_exists(self, video_id: str) -> Dict:
        """Check if video exists and has captions available"""
        try:
            # Simple check using YouTube oembed API
            response = requests.get(
                f"https://www.youtube.com/oembed?url=http://www.youtube.com/watch?v={video_id}&format=json",
                timeout=10
            )
            
            if response.status_code == 200:
                video_info = response.json()
                return {
                    "exists": True,
                    "title": video_info.get("title", "Unknown"),
                    "author": video_info.get("author_name", "Unknown")
                }
            else:
                return {"exists": False, "error": "Video not found or private"}
                
        except Exception as e:
            logger.error(f"Video check failed: {e}")
            return {"exists": False, "error": str(e)}
    
    def format_transcript_response(self, transcript_data: List[Dict], format_type: str = "clean") -> Dict:
        """Format transcript data for API response"""
        if not transcript_data:
            raise ValueError("No transcript data to format")
        
        if format_type == "clean":
            # Clean format - just text
            text_content = " ".join([item.get('text', '') for item in transcript_data])
            return {
                "content": text_content,
                "format": "clean",
                "entry_count": len(transcript_data)
            }
        
        elif format_type == "unclean":
            # Unclean format - with timestamps
            formatted_lines = []
            for item in transcript_data:
                start_time = item.get('start', 0)
                text = item.get('text', '')
                timestamp = self._seconds_to_timestamp(start_time)
                formatted_lines.append(f"[{timestamp}] {text}")
            
            return {
                "content": "\n".join(formatted_lines),
                "format": "unclean",
                "entry_count": len(transcript_data)
            }
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

# Create global instance
transcript_handler = EnhancedTranscriptHandler()


# Enhanced error response helper
def create_error_response(error_type: str, message: str, video_id: str = None, suggestions: List[str] = None) -> Dict:
    """Create standardized error response"""
    response = {
        "success": False,
        "error_type": error_type,
        "message": message,
        "timestamp": "2025-06-17T00:00:00Z"  # You can use datetime.now().isoformat()
    }
    
    if video_id:
        response["video_id"] = video_id
    
    if suggestions:
        response["suggestions"] = suggestions
    
    return response