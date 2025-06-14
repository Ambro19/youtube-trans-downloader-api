# Quick test script to verify the alternative method works
# Save as test_transcript.py and run: python test_transcript.py

import requests
import json

def test_transcript_endpoints():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Transcript Functionality\n")
    
    # Test 1: Network connectivity
    print("1️⃣ Testing network connectivity...")
    try:
        response = requests.get(f"{base_url}/debug/network")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Basic internet: {data.get('basic_internet', {}).get('status', 'Unknown')}")
            print(f"   ✅ YouTube connectivity: {data.get('youtube_connectivity', {}).get('status', 'Unknown')}")
        else:
            print(f"   ❌ Network test failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Network test error: {e}")
    
    # Test 2: Alternative transcript method
    print("\n2️⃣ Testing alternative transcript method...")
    test_video_id = "ZbZSe6N_BXs"
    try:
        response = requests.get(f"{base_url}/debug/alternative_method/{test_video_id}")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"   ✅ Alternative method works!")
                print(f"   📊 Found {data.get('transcript_segments', 0)} segments")
                print(f"   📝 Sample: {data.get('sample_text', 'No sample')[:100]}...")
            else:
                print(f"   ❌ Alternative method failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ Alternative test failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Alternative test error: {e}")
    
    # Test 3: Library method (for comparison)
    print("\n3️⃣ Testing library method...")
    try:
        response = requests.get(f"{base_url}/debug/transcript_raw/{test_video_id}")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"   ✅ Library method works!")
                print(f"   📊 Found {data.get('transcript_segments', 0)} segments")
            else:
                print(f"   ❌ Library method failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ Library test failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Library test error: {e}")
    
    print("\n" + "="*50)
    print("🎯 SUMMARY:")
    print("If Alternative Method ✅ = Your transcript downloads should work!")
    print("If Network Test ❌ = Check firewall/antivirus/proxy settings")
    print("If All Tests ❌ = Network connectivity issues")
    print("="*50)

if __name__ == "__main__":
    test_transcript_endpoints()