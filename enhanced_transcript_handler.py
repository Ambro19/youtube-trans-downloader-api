# enhanced_transcript_handler.py - FIXED VERSION

import logging
import re
from typing import List, Dict, Optional, Tuple
import requests
import json
import xml.etree.ElementTree as ET
from urllib.parse import unquote
import time

# Set up logging WITHOUT emojis to avoid Unicode encoding issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTranscriptHandler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
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
        Get transcript using fallback methods - FIXED VERSION
        Returns: (transcript_data, method_used)
        """
        methods = [
            ("youtube_transcript_api", self._get_transcript_youtube_api),
            ("direct_http_v1", self._get_transcript_direct_http_v1),
            ("direct_http_v2", self._get_transcript_direct_http_v2),
        ]
        
        last_error = None
        
        for method_name, method_func in methods:
            try:
                logger.info(f"Trying method: {method_name} for video {video_id}")
                result = method_func(video_id)
                if result:
                    logger.info(f"SUCCESS with method: {method_name}")
                    return result, method_name
            except Exception as e:
                logger.warning(f"Method {method_name} failed: {str(e)}")
                last_error = e
                # Add small delay between attempts
                time.sleep(0.5)
                continue
        
        # If all methods fail, raise the last error with context
        raise Exception(f"All transcript methods failed. Last error: {str(last_error)}")
    
    def _get_transcript_youtube_api(self, video_id: str) -> List[Dict]:
        """Primary method using youtube-transcript-api"""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            logger.info(f"Fetching transcript using YouTube Transcript API for video: {video_id}")
            
            # Try to get English transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
            )
            
            if not transcript_list:
                raise Exception("No transcript data returned from API")
            
            logger.info(f"Successfully retrieved {len(transcript_list)} transcript segments")
            return transcript_list
            
        except Exception as e:
            logger.warning(f"YouTube Transcript API failed: {str(e)}")
            raise Exception(f"YouTube Transcript API failed: {str(e)}")
    
    def _get_transcript_direct_http_v1(self, video_id: str) -> List[Dict]:
        """Direct HTTP method - IMPROVED VERSION"""
        try:
            # Step 1: Get the YouTube video page
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            logger.info(f"Fetching video page: {video_url}")
            
            response = self.session.get(video_url, timeout=15)
            
            if response.status_code != 200:
                raise Exception(f"Video page not accessible (Status: {response.status_code})")
            
            page_content = response.text
            logger.info(f"Got page content: {len(page_content)} characters")
            
            # Step 2: Look for player response data
            player_response_patterns = [
                r'var ytInitialPlayerResponse = ({.*?});',
                r'"playerResponse":"({.*?})"',
                r'ytInitialPlayerResponse"?\s*[:=]\s*({.+?})(?:[;,]|</script>)',
            ]
            
            player_data = None
            for pattern in player_response_patterns:
                matches = re.finditer(pattern, page_content, re.DOTALL)
                for match in matches:
                    try:
                        json_str = match.group(1)
                        # Clean up the JSON string
                        if json_str.startswith('"') and json_str.endswith('"'):
                            json_str = json_str[1:-1].replace('\\"', '"')
                        
                        player_data = json.loads(json_str)
                        if 'captions' in player_data:
                            break
                    except (json.JSONDecodeError, ValueError):
                        continue
                
                if player_data and 'captions' in player_data:
                    break
            
            if not player_data or 'captions' not in player_data:
                raise Exception("No player response or captions found in video page")
            
            # Step 3: Extract caption tracks
            captions_data = player_data.get('captions', {})
            caption_tracks = captions_data.get('playerCaptionsTracklistRenderer', {}).get('captionTracks', [])
            
            if not caption_tracks:
                raise Exception("No caption tracks found in player data")
            
            # Step 4: Find the best caption track (prefer English manual, then auto)
            best_caption = None
            
            # First try: Manual English captions
            for caption in caption_tracks:
                lang_code = caption.get('languageCode', '').lower()
                if lang_code.startswith('en') and not caption.get('kind', '').startswith('asr'):
                    best_caption = caption
                    logger.info(f"Using manual English caption")
                    break
            
            # Second try: Auto-generated English captions
            if not best_caption:
                for caption in caption_tracks:
                    lang_code = caption.get('languageCode', '').lower()
                    if lang_code.startswith('en'):
                        best_caption = caption
                        logger.info(f"Using auto-generated English caption")
                        break
            
            # Third try: Any available caption
            if not best_caption and caption_tracks:
                best_caption = caption_tracks[0]
                logger.info(f"Using first available caption")
            
            if not best_caption or 'baseUrl' not in best_caption:
                raise Exception("No usable caption track found")
            
            caption_url = best_caption['baseUrl']
            logger.info(f"Fetching captions from URL")
            
            # Step 5: Fetch the caption XML
            caption_response = self.session.get(caption_url, timeout=10)
            
            if caption_response.status_code != 200:
                raise Exception(f"Failed to download caption file (Status: {caption_response.status_code})")
            
            # Step 6: Parse the XML caption file
            return self._parse_caption_xml(caption_response.content)
                
        except Exception as e:
            logger.error(f"Direct HTTP v1 method failed: {str(e)}")
            raise
    
    def _get_transcript_direct_http_v2(self, video_id: str) -> List[Dict]:
        """Alternative direct HTTP method with different approach"""
        try:
            # Try different YouTube URL patterns
            urls_to_try = [
                f"https://www.youtube.com/watch?v={video_id}",
                f"https://m.youtube.com/watch?v={video_id}",
            ]
            
            for video_url in urls_to_try:
                try:
                    logger.info(f"Trying URL: {video_url}")
                    response = self.session.get(video_url, timeout=10)
                    
                    if response.status_code == 200:
                        page_content = response.text
                        
                        # Look for caption tracks using different patterns
                        caption_patterns = [
                            r'"captionTracks":\s*(\[.*?\])',
                            r'"captions".*?"captionTracks":\s*(\[.*?\])',
                            r'captionTracks["\']:\s*(\[.*?\])',
                        ]
                        
                        for pattern in caption_patterns:
                            match = re.search(pattern, page_content, re.DOTALL)
                            if match:
                                try:
                                    caption_tracks = json.loads(match.group(1))
                                    if caption_tracks:
                                        return self._process_caption_tracks(caption_tracks)
                                except (json.JSONDecodeError, ValueError):
                                    continue
                except Exception as e:
                    logger.warning(f"Failed to process URL {video_url}: {str(e)}")
                    continue
            
            raise Exception("Could not extract captions using alternative method")
            
        except Exception as e:
            logger.error(f"Direct HTTP v2 method failed: {str(e)}")
            raise
    
    def _parse_caption_xml(self, xml_content: bytes) -> List[Dict]:
        """Parse caption XML content"""
        try:
            root = ET.fromstring(xml_content)
            transcript_data = []
            
            for text_elem in root.findall('.//text'):
                start_time = float(text_elem.get('start', '0'))
                duration = float(text_elem.get('dur', '3'))
                text_content = text_elem.text or ''
                
                if text_content.strip():
                    # Decode HTML entities and clean up
                    text_content = unquote(text_content)
                    text_content = (text_content
                                   .replace('&amp;', '&')
                                   .replace('&lt;', '<')
                                   .replace('&gt;', '>')
                                   .replace('&quot;', '"')
                                   .replace('&#39;', "'")
                                   .replace('\n', ' ')
                                   .strip())
                    
                    # Remove HTML tags
                    text_content = re.sub(r'<[^>]+>', '', text_content)
                    text_content = re.sub(r'\s+', ' ', text_content).strip()
                    
                    if text_content:
                        transcript_data.append({
                            'text': text_content,
                            'start': start_time,
                            'duration': duration
                        })
            
            if not transcript_data:
                raise Exception("Transcript file contains no readable text")
            
            logger.info(f"Extracted {len(transcript_data)} transcript segments")
            return transcript_data
            
        except ET.ParseError as e:
            raise Exception(f"Failed to parse caption XML: {str(e)}")
    
    def _process_caption_tracks(self, caption_tracks: List[Dict]) -> List[Dict]:
        """Process caption tracks and fetch the actual transcript"""
        for track in caption_tracks:
            if 'baseUrl' in track:
                try:
                    caption_url = track['baseUrl']
                    response = self.session.get(caption_url, timeout=10)
                    
                    if response.status_code == 200:
                        return self._parse_caption_xml(response.content)
                except Exception as e:
                    logger.warning(f"Failed to process caption track: {str(e)}")
                    continue
        
        raise Exception("No processable caption tracks found")
    
    def check_video_exists(self, video_id: str) -> Dict:
        """Check if video exists and has captions available"""
        try:
            # Simple check using YouTube oembed API
            response = self.session.get(
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
        "detail": message  # For backward compatibility
    }
    
    if video_id:
        response["video_id"] = video_id
    
    if suggestions:
        response["suggestions"] = suggestions
    
    return response