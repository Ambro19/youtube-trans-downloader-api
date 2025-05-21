import os
import logging
from dotenv import load_dotenv
import sys

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("youtube_trans_downloader")

# Check environment and set variables
ENV = os.getenv("ENV", "development")
logger.info(f"Environment: {ENV}")

# Check if required environment variables are set
required_vars = [
    "STRIPE_SECRET_KEY",
    "SECRET_KEY",
]

# Database check - if using PostgreSQL in production, check for database variables
if ENV == "production":
    db_vars = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_NAME"]
    for var in db_vars:
        if var not in required_vars:
            required_vars.append(var)

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    logger.error("Please set these variables in a .env file or environment before running the application.")
    sys.exit(1)

# Validate Stripe price IDs
if not os.getenv("BASIC_PRICE_ID") or not os.getenv("PREMIUM_PRICE_ID"):
    logger.warning("Stripe price IDs (BASIC_PRICE_ID, PREMIUM_PRICE_ID) are not set. Subscription creation may fail.")

# Log startup information
logger.info("Starting YouTube Transcript Downloader API")
logger.info(f"Environment variables loaded from .env file")

if ENV == "production":
    db_host = os.getenv("DB_HOST", "localhost")
    db_name = os.getenv("DB_NAME", "youtube_trans_db")
    logger.info(f"Using PostgreSQL database at {db_host} for {db_name}")
else:
    logger.info(f"Using SQLite database for development")

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
   
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # Only use reload in development mode
    reload = ENV == "development"
   
    logger.info(f"Starting server on {host}:{port} (reload: {reload})")
    uvicorn.run("main:app", host=host, port=port, reload=reload)