# quick_test.py - Run this to verify your fix works
# Save this as a separate file and run it to test your transcript handler

import sys
import os
sys.path.append('.')  # Add current directory to path

try:
    from enhanced_transcript_handler import transcript_handler
    print("✅ Enhanced transcript handler imported successfully")
except Exception as e:
    print(f"❌ Failed to import enhanced_transcript_handler: {e}")
    exit(1)

# Test videos that are KNOWN to have captions
test_videos = [
    {
        "id": "ZbZSe6N_BXs", 
        "name": "Pharrell Williams - Happy",
        "description": "Popular music video with reliable captions"
    },
    {
        "id": "kqtD5dpn9C8",
        "name": "Python vs JavaScript", 
        "description": "Programming tutorial with captions"
    },
    {
        "id": "jNQXAC9IVRw",
        "name": "Me at the zoo",
        "description": "First YouTube video (may not have captions)"
    }
]

print("\n🧪 TESTING YOUTUBE TRANSCRIPT DOWNLOADER FIX")
print("=" * 50)

for i, video in enumerate(test_videos, 1):
    print(f"\n{i}. Testing: {video['name']} ({video['id']})")
    print(f"   Description: {video['description']}")
    
    # Test 1: Video exists check
    try:
        video_check = transcript_handler.check_video_exists(video['id'])
        if video_check.get('exists', False):
            print(f"   ✅ Video exists: {video_check.get('title', 'Unknown title')}")
        else:
            print(f"   ❌ Video check failed: {video_check.get('error', 'Unknown error')}")
            continue
    except Exception as e:
        print(f"   ❌ Video check error: {e}")
        continue
    
    # Test 2: Transcript extraction
    try:
        transcript_data, method_used = transcript_handler.get_transcript_with_fallbacks(video['id'])
        print(f"   ✅ Transcript retrieved using: {method_used}")
        print(f"   📊 Total segments: {len(transcript_data)}")
        
        # Show sample text
        if transcript_data:
            sample_text = transcript_data[0].get('text', '')[:100]
            print(f"   📝 Sample: '{sample_text}...'")
            
        # Test formatting
        try:
            clean_result = transcript_handler.format_transcript_response(transcript_data, "clean")
            unclean_result = transcript_handler.format_transcript_response(transcript_data, "unclean")
            print(f"   ✅ Clean format: {len(clean_result['content'])} characters")
            print(f"   ✅ Unclean format: {len(unclean_result['content'])} characters")
        except Exception as format_error:
            print(f"   ❌ Formatting error: {format_error}")
            
    except Exception as e:
        print(f"   ❌ Transcript extraction failed: {e}")

print("\n" + "=" * 50)
print("🎯 RECOMMENDATION:")
print("✅ Use videos that show '✅ Transcript retrieved' for testing your app")
print("❌ Avoid videos that show '❌ Transcript extraction failed'")

print("\n🔧 NEXT STEPS:")
print("1. If all tests passed: Your fix is working! Test in your app.")
print("2. If tests failed: Check your enhanced_transcript_handler.py file")
print("3. Make sure to restart your FastAPI server after changes")

print("\n📝 TO TEST IN YOUR APP:")
print("- Go to: http://localhost:3000/download")
print("- Use a working video ID from above")
print("- Should see transcript displayed like in Picture #4")