#!/usr/bin/env python3
"""
Deployment Health Checker
Run this to diagnose issues with your YouTube Content Downloader deployment
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_item(name: str, condition: bool, error_msg: str = ""):
    """Print check result with emoji"""
    if condition:
        print(f"✅ {name}")
        return True
    else:
        print(f"❌ {name}")
        if error_msg:
            print(f"   ↳ {error_msg}")
        return False

def main():
    print("=" * 60)
    print("YouTube Content Downloader - Health Check")
    print("=" * 60)
    print()
    
    all_good = True
    
    # Check Python version
    py_version = sys.version_info
    all_good &= check_item(
        f"Python {py_version.major}.{py_version.minor}.{py_version.micro}",
        py_version >= (3, 8),
        "Need Python 3.8 or higher"
    )
    
    # Check required files
    required_files = [
        "backend/main.py",
        "backend/transcript_utils.py",
        "backend/models.py",
        "backend/apt.txt",
        "requirements.txt",
    ]
    
    for file_path in required_files:
        exists = Path(file_path).exists()
        all_good &= check_item(f"File: {file_path}", exists, f"Missing: {file_path}")
    
    # Check environment variables
    print()
    print("Environment Variables:")
    print("-" * 40)
    
    env_vars = {
        "SECRET_KEY": "Required for JWT tokens",
        "DATABASE_URL": "Database connection string",
        "FRONTEND_URL": "Frontend URL for CORS",
        "APP_ENV": "Should be 'production' in prod",
    }
    
    for var, description in env_vars.items():
        value = os.getenv(var)
        has_value = bool(value)
        all_good &= check_item(
            f"{var}: {'✓' if has_value else 'missing'}",
            has_value,
            description
        )
    
    # Check optional but recommended
    print()
    print("Optional (Recommended):")
    print("-" * 40)
    
    optional_vars = {
        "YTDLP_COOKIES_B64": "For restricted content",
        "STRIPE_SECRET_KEY": "For payments",
        "SENDGRID_API_KEY": "For email",
    }
    
    for var, description in optional_vars.items():
        value = os.getenv(var)
        has_value = bool(value)
        emoji = "✅" if has_value else "⚠️ "
        print(f"{emoji} {var}: {'configured' if has_value else 'not set'}")
        if not has_value:
            print(f"   ↳ {description}")
    
    # Check system commands
    print()
    print("System Commands:")
    print("-" * 40)
    
    commands = {
        "yt-dlp": "YouTube downloader",
        "ffmpeg": "Media processing",
        "python3": "Python runtime",
    }
    
    for cmd, description in commands.items():
        try:
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            has_cmd = result.returncode == 0
            version_info = ""
            if has_cmd:
                # Try to extract version
                if "version" in result.stdout.lower():
                    lines = result.stdout.split('\n')
                    version_info = f" ({lines[0][:50]})"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            has_cmd = False
            version_info = ""
        
        check_item(f"{cmd}{version_info}", has_cmd, f"Install {cmd} - {description}")
    
    # Check Python packages
    print()
    print("Python Packages:")
    print("-" * 40)
    
    packages = [
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "yt_dlp",
        "youtube_transcript_api",
    ]
    
    for pkg in packages:
        try:
            __import__(pkg)
            check_item(f"Package: {pkg}", True)
        except ImportError:
            check_item(f"Package: {pkg}", False, f"Install with: pip install {pkg}")
    
    # Summary
    print()
    print("=" * 60)
    if all_good:
        print("✅ All critical checks passed!")
        print("Your deployment should be ready.")
    else:
        print("❌ Some checks failed.")
        print("Review the issues above and fix them.")
        print()
        print("Common fixes:")
        print("1. Set missing environment variables in Render dashboard")
        print("2. Ensure apt.txt includes 'ffmpeg'")
        print("3. Ensure requirements.txt includes 'yt-dlp>=2024.12.13'")
        print("4. Run: pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())