# test_transcript.py - Debug transcript API issues
import sys
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.exceptions import *

def test_video_ids():
    """Test multiple video IDs to see which ones work"""
    
    test_videos = [
        ("dQw4w9WgXcQ", "Rick Astley - Never Gonna Give You Up"),
        ("jNQXAC9IVRw", "Me at the zoo (first YouTube video)"),
        ("ZbZSe6N_BXs", "Sample video"),
        ("9bZkp7q19f0", "PSY - GANGNAM STYLE"),
        ("kffacxfA7G4", "Baby Shark Dance")
    ]
    
    print("🧪 Testing YouTube Transcript API with multiple videos...\n")
    
    for video_id, title in test_videos:
        print(f"🔍 Testing: {video_id} ({title})")
        
        try:
            # Test basic connectivity to YouTube
            response = requests.get(f"https://www.youtube.com/watch?v={video_id}", timeout=10)
            print(f"   ✅ YouTube page accessible: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Can't reach YouTube: {e}")
            continue
        
        try:
            # Try to list available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            available_transcripts = []
            
            for transcript in transcript_list:
                available_transcripts.append({
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated
                })
            
            print(f"   ✅ Available transcripts: {len(available_transcripts)}")
            for t in available_transcripts[:3]:  # Show first 3
                print(f"      - {t['language']} ({t['language_code']}) {'[Auto]' if t['is_generated'] else '[Manual]'}")
            
            # Try to get English transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            print(f"   ✅ Got transcript: {len(transcript)} segments")
            
            # Show first segment
            if transcript:
                first_segment = transcript[0]
                print(f"   📝 First segment: \"{first_segment['text'][:50]}...\"")
                print(f"   ✅ SUCCESS: {video_id} works!\n")
                return video_id  # Return the first working video ID
            
        except TranscriptsDisabled:
            print(f"   ❌ Transcripts disabled for {video_id}")
        except NoTranscriptFound:
            print(f"   ❌ No transcript found for {video_id}")
        except VideoUnavailable:
            print(f"   ❌ Video unavailable: {video_id}")
        except Exception as e:
            print(f"   ❌ Error: {type(e).__name__}: {e}")
        
        print()
    
    print("❌ No working video IDs found!")
    return None

def test_alternative_method(video_id):
    """Test the alternative HTTP method"""
    print(f"🔄 Testing alternative method for {video_id}...")
    
    try:
        import re
        import json
        from urllib.parse import unquote
        
        # Get YouTube video page
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        print(f"   📡 Fetching: {video_url}")
        response = requests.get(video_url, headers=headers, timeout=15)
        print(f"   📄 Page size: {len(response.text)} characters")
        
        # Look for captions
        caption_patterns = [
            r'"captionTracks":\[(.*?)\]',
            r'"captions".*?"playerCaptionsTracklistRenderer".*?"captionTracks":\[(.*?)\]'
        ]
        
        for i, pattern in enumerate(caption_patterns):
            match = re.search(pattern, response.text)
            if match:
                print(f"   ✅ Found captions with pattern {i+1}")
                return True
        
        print(f"   ❌ No caption tracks found in page HTML")
        
        # Check if it's a valid YouTube page
        if "youtube.com" in response.text and "watch" in response.text:
            print(f"   ✅ Valid YouTube page")
        else:
            print(f"   ❌ Not a valid YouTube page")
            
        return False
        
    except Exception as e:
        print(f"   ❌ Alternative method failed: {e}")
        return False

def main():
    print("🚀 YouTube Transcript API Debugging Tool\n")
    
    # Test network connectivity
    print("🌐 Testing network connectivity...")
    try:
        response = requests.get("https://httpbin.org/get", timeout=10)
        print(f"   ✅ Internet connection: OK ({response.status_code})")
    except Exception as e:
        print(f"   ❌ Internet connection failed: {e}")
        return
    
    try:
        response = requests.get("https://www.youtube.com", timeout=10)
        print(f"   ✅ YouTube accessible: OK ({response.status_code})")
    except Exception as e:
        print(f"   ❌ YouTube not accessible: {e}")
        return
    
    print()
    
    # Test library version
    try:
        from youtube_transcript_api import __version__
        print(f"📚 YouTube Transcript API version: {__version__}")
    except:
        print(f"📚 YouTube Transcript API version: Unknown")
    
    print()
    
    # Test specific video IDs
    working_video = test_video_ids()
    
    if working_video:
        print(f"🎯 Found working video: {working_video}")
        test_alternative_method(working_video)
    else:
        print("🔄 Testing alternative method anyway...")
        test_alternative_method("dQw4w9WgXcQ")

if __name__ == "__main__":
    main()