"""
🔥 FULLY PATCHED run.py - Fixed for Mobile Connections
=====================================================
This script properly imports and starts the FastAPI application
with all the fixed dependencies and error handling.
CRITICAL: This runs on 0.0.0.0:8000 to allow mobile connections!
"""
import sys
import os
import logging
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("youtube_trans_downloader")

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import fastapi
        logger.info("✅ FastAPI available")
    except ImportError:
        missing_deps.append("fastapi")
    
    try:
        import uvicorn
        logger.info("✅ Uvicorn available")
    except ImportError:
        missing_deps.append("uvicorn")
    
    try:
        import sqlalchemy
        logger.info("✅ SQLAlchemy available")
    except ImportError:
        missing_deps.append("sqlalchemy")
    
    try:
        import youtube_transcript_api
        logger.info("✅ YouTube Transcript API available")
    except ImportError:
        missing_deps.append("youtube-transcript-api")
    
    try:
        import yt_dlp
        logger.info("✅ yt-dlp available")
    except ImportError:
        missing_deps.append("yt-dlp")
    
    try:
        import stripe
        logger.info("✅ Stripe available")
    except ImportError:
        missing_deps.append("stripe")
    
    if missing_deps:
        logger.error(f"❌ Missing dependencies: {missing_deps}")
        logger.error("Please install them with: pip install " + " ".join(missing_deps))
        return False
    
    return True

def check_files():
    """Check if all required files exist"""
    required_files = [
        "main.py",
        "models.py", 
        "transcript_utils.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            logger.info(f"✅ Found {file}")
    
    if missing_files:
        logger.error(f"❌ Missing files: {missing_files}")
        return False
    
    return True

def check_network_config():
    """Check network configuration for mobile access"""
    import socket
    
    # Get the local IP address
    try:
        # Connect to a remote server to get the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        
        logger.info(f"🔥 Local IP Address: {local_ip}")
        logger.info(f"🔥 Mobile devices should connect to: http://{local_ip}:8000")
        
        return local_ip
    except Exception as e:
        logger.warning(f"Could not determine local IP: {e}")
        return "192.168.1.185"  # fallback

def main():
    """Main function to start the application"""
    try:
        logger.info("🔥 MOBILE-READY STARTUP SEQUENCE")
        logger.info("Environment: development")
        logger.info("Starting YouTube Transcript Downloader API")
        logger.info("Environment variables loaded from .env file")
        logger.info("Using SQLite database for development")
        
        # Check network configuration
        local_ip = check_network_config()
        
        # Check dependencies
        if not check_dependencies():
            logger.error("❌ Dependency check failed")
            sys.exit(1)
        
        # Check files
        if not check_files():
            logger.error("❌ File check failed")
            sys.exit(1)
        
        # Import and start the application
        try:
            from main import app
            logger.info("✅ Application imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import application: {e}")
            logger.error("Make sure you have the updated main.py and models.py files")
            sys.exit(1)
        
        # 🔥 CRITICAL: Start server on 0.0.0.0:8000 for mobile access
        import uvicorn
        
        logger.info("🔥 STARTING SERVER FOR MOBILE ACCESS")
        logger.info("🔥 Host: 0.0.0.0 (allows connections from any device)")
        logger.info("🔥 Port: 8000")
        logger.info(f"🔥 Mobile URL: http://{local_ip}:8000")
        logger.info("🔥 Localhost URL: http://localhost:8000")
        logger.info("🔥 Server starting with reload enabled...")
        
        # Print mobile connection instructions
        print("\n" + "="*60)
        print("🔥 MOBILE CONNECTION INSTRUCTIONS")
        print("="*60)
        print(f"📱 On your mobile device, open browser and go to:")
        print(f"   http://{local_ip}:8000")
        print(f"💻 On this computer, use:")
        print(f"   http://localhost:8000")
        print("🔥 Make sure both devices are on the same WiFi network!")
        print("="*60 + "\n")
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",  # 🔥 CRITICAL - MUST be 0.0.0.0 for mobile access
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        logger.error(f"❌ Error details: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#================ =====================  =============

# """
# 🔥 FULLY PATCHED run.py - Fixed for Mobile Connections
# =====================================================
# This script properly imports and starts the FastAPI application
# with all the fixed dependencies and error handling.
# CRITICAL: This runs on 0.0.0.0:8000 to allow mobile connections!
# """
# import sys
# import os
# import logging
# from pathlib import Path

# # Add the current directory to Python path
# current_dir = Path(__file__).parent
# sys.path.insert(0, str(current_dir))

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger("youtube_trans_downloader")

# def check_dependencies():
#     """Check if all required dependencies are available"""
#     missing_deps = []
    
#     try:
#         import fastapi
#         logger.info("✅ FastAPI available")
#     except ImportError:
#         missing_deps.append("fastapi")
    
#     try:
#         import uvicorn
#         logger.info("✅ Uvicorn available")
#     except ImportError:
#         missing_deps.append("uvicorn")
    
#     try:
#         import sqlalchemy
#         logger.info("✅ SQLAlchemy available")
#     except ImportError:
#         missing_deps.append("sqlalchemy")
    
#     try:
#         import youtube_transcript_api
#         logger.info("✅ YouTube Transcript API available")
#     except ImportError:
#         missing_deps.append("youtube-transcript-api")
    
#     try:
#         import yt_dlp
#         logger.info("✅ yt-dlp available")
#     except ImportError:
#         missing_deps.append("yt-dlp")
    
#     try:
#         import stripe
#         logger.info("✅ Stripe available")
#     except ImportError:
#         missing_deps.append("stripe")
    
#     if missing_deps:
#         logger.error(f"❌ Missing dependencies: {missing_deps}")
#         logger.error("Please install them with: pip install " + " ".join(missing_deps))
#         return False
    
#     return True

# def check_files():
#     """Check if all required files exist"""
#     required_files = [
#         "main.py",
#         "models.py", 
#         "transcript_utils.py"
#     ]
    
#     missing_files = []
#     for file in required_files:
#         if not os.path.exists(file):
#             missing_files.append(file)
#         else:
#             logger.info(f"✅ Found {file}")
    
#     if missing_files:
#         logger.error(f"❌ Missing files: {missing_files}")
#         return False
    
#     return True

# def check_network_config():
#     """Check network configuration for mobile access"""
#     import socket
    
#     # Get the local IP address
#     try:
#         # Connect to a remote server to get the local IP
#         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         s.connect(("8.8.8.8", 80))
#         local_ip = s.getsockname()[0]
#         s.close()
        
#         logger.info(f"🔥 Local IP Address: {local_ip}")
#         logger.info(f"🔥 Mobile devices should connect to: http://{local_ip}:8000")
        
#         return local_ip
#     except Exception as e:
#         logger.warning(f"Could not determine local IP: {e}")
#         return "192.168.1.185"  # fallback

# def main():
#     """Main function to start the application"""
#     try:
#         logger.info("🔥 MOBILE-READY STARTUP SEQUENCE")
#         logger.info("Environment: development")
#         logger.info("Starting YouTube Transcript Downloader API")
#         logger.info("Environment variables loaded from .env file")
#         logger.info("Using SQLite database for development")
        
#         # Check network configuration
#         local_ip = check_network_config()
        
#         # Check dependencies
#         if not check_dependencies():
#             logger.error("❌ Dependency check failed")
#             sys.exit(1)
        
#         # Check files
#         if not check_files():
#             logger.error("❌ File check failed")
#             sys.exit(1)
        
#         # Import and start the application
#         try:
#             from main import app
#             logger.info("✅ Application imported successfully")
#         except ImportError as e:
#             logger.error(f"❌ Failed to import application: {e}")
#             logger.error("Make sure you have the updated main.py and models.py files")
#             sys.exit(1)
        
#         # 🔥 CRITICAL: Start server on 0.0.0.0:8000 for mobile access
#         import uvicorn
        
#         logger.info("🔥 STARTING SERVER FOR MOBILE ACCESS")
#         logger.info("🔥 Host: 0.0.0.0 (allows connections from any device)")
#         logger.info("🔥 Port: 8000")
#         logger.info(f"🔥 Mobile URL: http://{local_ip}:8000")
#         logger.info("🔥 Localhost URL: http://localhost:8000")
#         logger.info("🔥 Server starting with reload enabled...")
        
#         # Print mobile connection instructions
#         print("\n" + "="*60)
#         print("🔥 MOBILE CONNECTION INSTRUCTIONS")
#         print("="*60)
#         print(f"📱 On your mobile device, open browser and go to:")
#         print(f"   http://{local_ip}:8000")
#         print(f"💻 On this computer, use:")
#         print(f"   http://localhost:8000")
#         print("🔥 Make sure both devices are on the same WiFi network!")
#         print("="*60 + "\n")
        
#         uvicorn.run(
#             "main:app",
#             host="0.0.0.0",  # 🔥 CRITICAL - MUST be 0.0.0.0 for mobile access
#             port=8000,
#             reload=True,
#             log_level="info"
#         )
        
#     except KeyboardInterrupt:
#         logger.info("👋 Server stopped by user")
#     except Exception as e:
#         logger.error(f"❌ Server failed to start: {e}")
#         logger.error(f"❌ Error details: {str(e)}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()

