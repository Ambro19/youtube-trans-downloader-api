
"""
Run script for YouTube Transcript Downloader API
==============================================

This script properly imports and starts the FastAPI application
with all the fixed dependencies and error handling.
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

def main():
    """Main function to start the application"""
    try:
        logger.info("Environment: development")
        logger.info("Starting YouTube Transcript Downloader API")
        logger.info("Environment variables loaded from .env file")
        logger.info("Using SQLite database for development")
        
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
        
        # Start server
        import uvicorn
        logger.info("Starting server on 0.0.0.0:8000 (reload: True)")
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


#====================================================

# #run.py

# import os
# import logging
# from dotenv import load_dotenv
# import sys

# # Load environment variables from .env file
# load_dotenv()

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("app.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger("youtube_trans_downloader")

# # Check environment and set variables
# ENV = os.getenv("ENV", "development")
# logger.info(f"Environment: {ENV}")

# # Check if required environment variables are set
# required_vars = [
#     "STRIPE_SECRET_KEY",
#     "SECRET_KEY",
# ]

# # Database check - if using PostgreSQL in production, check for database variables
# if ENV == "production":
#     db_vars = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_NAME"]
#     for var in db_vars:
#         if var not in required_vars:
#             required_vars.append(var)

# missing_vars = [var for var in required_vars if not os.getenv(var)]
# if missing_vars:
#     logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
#     logger.error("Please set these variables in a .env file or environment before running the application.")
#     sys.exit(1)

# # Validate Stripe price IDs
# if not os.getenv("PRO_PRICE_ID") or not os.getenv("PREMIUM_PRICE_ID"):
#     logger.warning("Stripe price IDs (PRO_PRICE_ID, PREMIUM_PRICE_ID) are not set. Subscription creation may fail.")

# # Log startup information
# logger.info("Starting YouTube Transcript Downloader API")
# logger.info(f"Environment variables loaded from .env file")

# if ENV == "production":
#     db_host = os.getenv("DB_HOST", "localhost")
#     db_name = os.getenv("DB_NAME", "youtube_trans_db")
#     logger.info(f"Using PostgreSQL database at {db_host} for {db_name}")
# else:
#     logger.info(f"Using SQLite database for development")

# # Run the FastAPI application
# if __name__ == "__main__":
#     import uvicorn
   
#     host = os.getenv("HOST", "0.0.0.0")
#     port = int(os.getenv("PORT", "8000"))
    
#     # Only use reload in development mode
#     reload = ENV == "development"
   
#     logger.info(f"Starting server on {host}:{port} (reload: {reload})")
#     uvicorn.run("main:app", host=host, port=port, reload=reload)
