# test_working_alternative.py - Test the working alternative method
import requests
import re
import json
import xml.etree.ElementTree as ET
from urllib.parse import unquote

def test_working_alternative_method(video_id="jNQXAC9IVRw"):
    """Test the working alternative method"""
    print(f"🧪 Testing working alternative method for: {video_id}")
    
    try:
        # Step 1: Get YouTube page
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        print(f"   📡 Fetching: {video_url}")
        response = requests.get(video_url, headers=headers, timeout=20)
        print(f"   📄 Page size: {len(response.text)} characters")
        
        # Step 2: Look for baseUrl directly (simpler approach)
        baseurl_pattern = r'"baseUrl":\s*"([^"]*timedtext[^"]*)"'
        baseurl_matches = re.findall(baseurl_pattern, response.text)
        
        if baseurl_matches:
            caption_url = baseurl_matches[0].replace('\\u0026', '&').replace('\\/', '/')
            print(f"   ✅ Found caption URL: {caption_url[:100]}...")
            
            # Step 3: Fetch captions
            print(f"   📥 Fetching caption XML...")
            caption_response = requests.get(caption_url, headers=headers, timeout=15)
            print(f"   📄 Caption XML size: {len(caption_response.content)} bytes")
            
            # Step 4: Parse XML
            try:
                root = ET.fromstring(caption_response.content)
                text_elements = root.findall('.//text')
                print(f"   📝 Found {len(text_elements)} text segments")
                
                if len(text_elements) > 0:
                    # Show first few segments
                    print(f"   📖 Sample segments:")
                    for i, elem in enumerate(text_elements[:3]):
                        text = elem.text or ''
                        start = elem.get('start', '0')
                        print(f"      [{start}s] {text[:50]}...")
                    
                    print(f"   ✅ SUCCESS! Working alternative method extracted {len(text_elements)} segments")
                    return True
                else:
                    print(f"   ❌ No text segments found in XML")
                    return False
                    
            except ET.ParseError as e:
                print(f"   ❌ XML parsing failed: {e}")
                print(f"   📄 Raw XML preview: {caption_response.text[:200]}...")
                return False
        else:
            print(f"   ❌ No caption URL found in page")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("🚀 Testing Working Alternative Transcript Method\n")
    
    test_videos = [
        "jNQXAC9IVRw",  # First YouTube video
        "ZbZSe6N_BXs",  # From examples
        "dQw4w9WgXcQ"   # Rick Roll
    ]
    
    working_count = 0
    for video_id in test_videos:
        if test_working_alternative_method(video_id):
            working_count += 1
        print()
    
    print(f"🎯 Results: {working_count}/{len(test_videos)} videos worked with alternative method")
    
    if working_count > 0:
        print("✅ The working alternative method is functional!")
        print("💡 Update your main.py to use the working alternative method")
    else:
        print("❌ Alternative method needs more work")

if __name__ == "__main__":
    main()