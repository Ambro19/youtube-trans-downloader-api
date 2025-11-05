#!/usr/bin/env python3
"""
YCD App - Post-Deployment Verification Script
Tests all critical fixes to ensure proper deployment
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    """Print section header."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{text:^70}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")

def print_success(text):
    """Print success message."""
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    """Print error message."""
    print(f"{RED}❌ {text}{RESET}")

def print_warning(text):
    """Print warning message."""
    print(f"{YELLOW}⚠️  {text}{RESET}")

def print_info(text):
    """Print info message."""
    print(f"{BLUE}ℹ️  {text}{RESET}")

def check_file_exists(filepath):
    """Check if file exists."""
    if Path(filepath).exists():
        print_success(f"Found: {filepath}")
        return True
    else:
        print_error(f"Missing: {filepath}")
        return False

def check_function_signature(module_path, function_name, expected_params):
    """Check if function has correct signature."""
    try:
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        func = getattr(module, function_name)
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        if params == expected_params:
            print_success(f"{function_name}() has correct signature: {params}")
            return True
        else:
            print_error(f"{function_name}() signature mismatch!")
            print_error(f"  Expected: {expected_params}")
            print_error(f"  Found: {params}")
            return False
    except Exception as e:
        print_error(f"Error checking {function_name}: {e}")
        return False

def check_function_exists(module_path, function_name):
    """Check if function exists in module."""
    try:
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, function_name):
            print_success(f"Function {function_name}() exists")
            return True
        else:
            print_error(f"Function {function_name}() not found")
            return False
    except Exception as e:
        print_error(f"Error checking {function_name}: {e}")
        return False

def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print_success(f"FFmpeg available: {version}")
            return True
        else:
            print_error("FFmpeg not working correctly")
            return False
    except FileNotFoundError:
        print_error("FFmpeg not found in PATH")
        return False
    except Exception as e:
        print_error(f"Error checking FFmpeg: {e}")
        return False

def check_ytdlp():
    """Check if yt-dlp is available."""
    try:
        import yt_dlp
        print_success(f"yt-dlp available: {yt_dlp.version.__version__}")
        return True
    except ImportError:
        print_error("yt-dlp not installed")
        return False

def check_youtube_transcript_api():
    """Check if youtube-transcript-api is available."""
    try:
        import youtube_transcript_api
        print_success("youtube-transcript-api available")
        return True
    except ImportError:
        print_error("youtube-transcript-api not installed")
        return False

def main():
    """Main verification routine."""
    print_header("YCD App - Post-Deployment Verification")
    print_info("This script verifies that all critical fixes are properly deployed\n")
    
    # Track results
    all_checks_passed = True
    
    # 1. Check Files
    print_header("1. Checking File Deployment")
    files_ok = True
    files_ok &= check_file_exists("backend/transcript_utils.py")
    files_ok &= check_file_exists("backend/batch.py")
    files_ok &= check_file_exists("backend/BatchJobs.py")
    
    if not files_ok:
        print_error("\nSome required files are missing!")
        print_warning("Please deploy all files from the fix package\n")
        all_checks_passed = False
    
    # 2. Check Function Signatures
    print_header("2. Checking Function Signatures")
    
    if Path("backend/transcript_utils.py").exists():
        # Check audio function
        audio_ok = check_function_signature(
            "backend/transcript_utils.py",
            "download_audio_with_ytdlp",
            ["video_id_or_url", "quality", "output_dir"]
        )
        
        # Check video function
        video_ok = check_function_signature(
            "backend/transcript_utils.py",
            "download_video_with_ytdlp",
            ["video_id_or_url", "quality", "output_dir"]
        )
        
        # Check _safe_outtmpl exists
        outtmpl_ok = check_function_exists(
            "backend/transcript_utils.py",
            "_safe_outtmpl"
        )
        
        if not (audio_ok and video_ok and outtmpl_ok):
            print_error("\nFunction signature issues detected!")
            print_warning("transcript_utils.py may not be properly deployed\n")
            all_checks_passed = False
    else:
        print_error("Cannot verify signatures - transcript_utils.py missing")
        all_checks_passed = False
    
    # 3. Check Batch Module
    print_header("3. Checking Batch Module")
    
    if Path("backend/batch.py").exists():
        # Check get_formatted_transcript exists
        formatted_ok = check_function_exists(
            "backend/batch.py",
            "get_formatted_transcript"
        )
        
        if not formatted_ok:
            print_error("\nBatch transcript formatting may not work correctly!")
            print_warning("batch.py may not be properly deployed\n")
            all_checks_passed = False
    else:
        print_error("Cannot verify batch module - batch.py missing")
        all_checks_passed = False
    
    # 4. Check Dependencies
    print_header("4. Checking Dependencies")
    deps_ok = True
    deps_ok &= check_ffmpeg()
    deps_ok &= check_ytdlp()
    deps_ok &= check_youtube_transcript_api()
    
    if not deps_ok:
        print_error("\nSome dependencies are missing!")
        print_warning("Install missing dependencies with:")
        print_warning("  pip install --upgrade yt-dlp youtube-transcript-api --break-system-packages\n")
        all_checks_passed = False
    
    # 5. Final Summary
    print_header("Verification Summary")
    
    if all_checks_passed:
        print_success("🎉 All checks passed!")
        print_success("Your YCD App is properly configured and ready to use.\n")
        print_info("Next steps:")
        print_info("  1. Start your application: python run.py")
        print_info("  2. Test audio download")
        print_info("  3. Test video download")
        print_info("  4. Test batch transcript formatting\n")
        return 0
    else:
        print_error("⚠️  Some checks failed!")
        print_warning("Please review the errors above and fix them before proceeding.\n")
        print_info("Common fixes:")
        print_info("  - Ensure all files are in backend/ directory")
        print_info("  - Install missing dependencies")
        print_info("  - Restart your application\n")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nVerification cancelled by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        sys.exit(1)