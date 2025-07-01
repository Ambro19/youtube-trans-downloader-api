# test_simple.py - Simple debug script without exceptions import
import requests

def test_basic_connectivity():
    """Test basic network connectivity"""
    print("🌐 Testing network connectivity...")
    
    try:
        response = requests.get("https://httpbin.org/get", timeout=10)
        print(f"   ✅ Internet: OK ({response.status_code})")
    except Exception as e:
        print(f"   ❌ Internet failed: {e}")
        return False
    
    try:
        response = requests.get("https://www.youtube.com", timeout=10)
        print(f"   ✅ YouTube: OK ({response.status_code})")
    except Exception as e:
        print(f"   ❌ YouTube failed: {e}")
        return False
    
    return True

def test_youtube_transcript_api():
    """Test the YouTube transcript API installation"""
    print("\n📚 Testing YouTube Transcript API...")
    
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        print("   ✅ YouTubeTranscriptApi imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import YouTubeTranscriptApi: {e}")
        return False
    
    try:
        from youtube_transcript_api import __version__
        print(f"   📖 Version: {__version__}")
    except ImportError:
        print("   ⚠️  Version info not available")
    
    return True

def test_simple_transcript(video_id="jNQXAC9IVRw"):
    """Test getting a transcript for a simple video"""
    print(f"\n🧪 Testing transcript for video: {video_id}")
    
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Try to get transcript
        print(f"   🔍 Attempting to get transcript...")
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        if transcript:
            print(f"   ✅ Success! Got {len(transcript)} segments")
            if len(transcript) > 0:
                first_segment = transcript[0]
                print(f"   📝 First segment: \"{first_segment.get('text', 'No text')[:50]}...\"")
            return True
        else:
            print(f"   ❌ No transcript returned")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {type(e).__name__}: {str(e)}")
        return False

def test_multiple_videos():
    """Test multiple video IDs"""
    print("\n🎯 Testing multiple video IDs...")
    
    test_videos = [
        "jNQXAC9IVRw",  # First YouTube video - should have transcripts
        "ZbZSe6N_BXs",  # From your examples
        "9bZkp7q19f0",  # Gangnam Style - very popular, should have transcripts
        "dQw4w9WgXcQ"   # Rick Roll - the one that failed
    ]
    
    working_videos = []
    
    for video_id in test_videos:
        print(f"\n   Testing: {video_id}")
        if test_simple_transcript(video_id):
            working_videos.append(video_id)
    
    print(f"\n📊 Results: {len(working_videos)}/{len(test_videos)} videos worked")
    if working_videos:
        print(f"   ✅ Working videos: {', '.join(working_videos)}")
    else:
        print(f"   ❌ No videos worked!")
    
    return working_videos

def test_alternative_method(video_id="jNQXAC9IVRw"):
    """Test alternative HTTP method"""
    print(f"\n🔄 Testing alternative method for {video_id}...")
    
    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        print(f"   📡 Fetching: {video_url}")
        response = requests.get(video_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            print(f"   ✅ Page loaded: {len(response.text)} characters")
            
            # Look for caption indicators
            if '"captions"' in response.text:
                print(f"   ✅ Found captions in page HTML")
                return True
            else:
                print(f"   ❌ No captions found in page HTML")
                return False
        else:
            print(f"   ❌ Failed to load page: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Alternative method failed: {e}")
        return False

def main():
    print("🚀 YouTube Transcript API Simple Debug Tool\n")
    
    # Test basic connectivity
    if not test_basic_connectivity():
        print("\n❌ Basic connectivity failed. Check your internet connection.")
        return
    
    # Test API installation
    if not test_youtube_transcript_api():
        print("\n❌ YouTube Transcript API not properly installed.")
        print("💡 Try: pip uninstall youtube-transcript-api && pip install youtube-transcript-api==0.6.1")
        return
    
    # Test simple transcript
    working_videos = test_multiple_videos()
    
    # Test alternative method
    test_alternative_method()
    
    print(f"\n🎯 Summary:")
    if working_videos:
        print(f"   ✅ Found {len(working_videos)} working video(s)")
        print(f"   💡 Use one of these in your app: {working_videos[0]}")
    else:
        print(f"   ❌ No videos worked with the API")
        print(f"   💡 Try reinstalling: pip install youtube-transcript-api==0.6.1")

if __name__ == "__main__":
    main()