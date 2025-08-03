# Simple main.py to test without path issues
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="YouTube Transcript Downloader API")

# Simple CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "YouTube Content Downloader API (Simple Test)", 
        "status": "running",
        "note": "This is a simplified version to test startup"
    }

if __name__ == "__main__":
    import uvicorn
    print("🔥 Starting SIMPLE server on 0.0.0.0:8000")
    
    uvicorn.run(
        "main_simple:app", 
        host="0.0.0.0",
        port=8000, 
        reload=True
    )