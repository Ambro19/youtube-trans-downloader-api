"""
Diagnostic script to check proxy configuration
Run this in your backend directory to verify everything is set up correctly
"""
import os
import sys

print("=" * 60)
print("🔍 PROXY CONFIGURATION DIAGNOSTICS")
print("=" * 60)

# Check 1: Environment variables
print("\n1️⃣ Checking Environment Variables:")
print("-" * 60)

env_vars = {
    "PROXY_ENABLED": os.getenv("PROXY_ENABLED"),
    "PROXY_HOST": os.getenv("PROXY_HOST"),
    "PROXY_PORT": os.getenv("PROXY_PORT"),
    "PROXY_USERNAME": os.getenv("PROXY_USERNAME"),
    "PROXY_PASSWORD": os.getenv("PROXY_PASSWORD"),
}

for key, value in env_vars.items():
    if value:
        if key == "PROXY_PASSWORD":
            print(f"   ✅ {key} = {value[:5]}****{value[-3:] if len(value) > 8 else '***'}")
        else:
            print(f"   ✅ {key} = {value}")
    else:
        print(f"   ❌ {key} = NOT SET")

# Check 2: .env file exists
print("\n2️⃣ Checking .env File:")
print("-" * 60)

if os.path.exists(".env"):
    print("   ✅ .env file exists")
    with open(".env", "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]
        proxy_lines = [l for l in lines if "PROXY" in l]
        if proxy_lines:
            print(f"   ✅ Found {len(proxy_lines)} PROXY lines in .env:")
            for line in proxy_lines:
                if "PASSWORD" in line:
                    key, val = line.split("=", 1)
                    print(f"      {key}={val[:5]}****")
                else:
                    print(f"      {line}")
        else:
            print("   ⚠️  No PROXY variables found in .env")
else:
    print("   ❌ .env file NOT FOUND")
    print("   💡 Create a .env file with proxy variables")

# Check 3: transcript_utils.py has proxy code
print("\n3️⃣ Checking transcript_utils.py:")
print("-" * 60)

try:
    with open("transcript_utils.py", "r") as f:
        content = f.read()
        if "PROXY_ENABLED" in content and "Using proxy" in content:
            print("   ✅ Proxy code found in transcript_utils.py")
        else:
            print("   ❌ Proxy code NOT found in transcript_utils.py")
            print("   💡 You need to replace with the proxy-enabled version!")
except FileNotFoundError:
    print("   ❌ transcript_utils.py NOT FOUND")

# Check 4: Can import the module
print("\n4️⃣ Checking Module Import:")
print("-" * 60)

try:
    from transcript_utils import _get_cookies_file, get_transcript_with_ytdlp
    print("   ✅ Successfully imported transcript_utils functions")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")

# Check 5: Test proxy configuration
print("\n5️⃣ Testing Proxy Configuration:")
print("-" * 60)

proxy_enabled = os.getenv("PROXY_ENABLED", "false").lower() == "true"

if proxy_enabled:
    print("   ✅ PROXY_ENABLED = true")
    
    required = ["PROXY_HOST", "PROXY_PORT", "PROXY_USERNAME", "PROXY_PASSWORD"]
    missing = [k for k in required if not os.getenv(k)]
    
    if missing:
        print(f"   ❌ Missing required variables: {', '.join(missing)}")
    else:
        print("   ✅ All required proxy variables are set")
        
        # Try to construct proxy URL
        host = os.getenv("PROXY_HOST")
        port = os.getenv("PROXY_PORT")
        user = os.getenv("PROXY_USERNAME")
        password = os.getenv("PROXY_PASSWORD")
        
        proxy_url = f"http://{user}:{password}@{host}:{port}"
        print(f"   🌐 Proxy URL would be: http://{user}:****@{host}:{port}")
else:
    print("   ❌ PROXY_ENABLED is not 'true'")
    print(f"      Current value: '{os.getenv('PROXY_ENABLED')}'")
    print("   💡 Set PROXY_ENABLED=true in your .env file")

# Summary
print("\n" + "=" * 60)
print("📋 SUMMARY")
print("=" * 60)

all_vars_set = all(env_vars.values())
proxy_code_exists = "PROXY_ENABLED" in content if 'content' in locals() else False
proxy_properly_enabled = proxy_enabled and all_vars_set

if proxy_properly_enabled and proxy_code_exists:
    print("   ✅ PROXY IS FULLY CONFIGURED!")
    print("   ✅ Ready for production deployment")
elif not proxy_code_exists:
    print("   ❌ PROXY CODE MISSING in transcript_utils.py")
    print("   💡 Replace with transcript_utils_with_proxy.py")
elif not proxy_enabled:
    print("   ❌ PROXY NOT ENABLED")
    print("   💡 Set PROXY_ENABLED=true in .env")
elif not all_vars_set:
    print("   ❌ PROXY VARIABLES INCOMPLETE")
    print("   💡 Add all 5 proxy variables to .env")
else:
    print("   ⚠️  PARTIAL CONFIGURATION")
    print("   💡 Review the issues above")

print("=" * 60)