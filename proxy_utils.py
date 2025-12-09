import os
import subprocess
import uuid
from yt_dlp.utils import DownloadError

def get_proxy_env():
    if os.getenv("PROXY_ENABLED", "false").lower() == "true":
        return {
            "http_proxy": f"http://{os.getenv('PROXY_USERNAME')}:{os.getenv('PROXY_PASSWORD')}@{os.getenv('PROXY_HOST')}:{os.getenv('PROXY_PORT')}",
            "https_proxy": f"http://{os.getenv('PROXY_USERNAME')}:{os.getenv('PROXY_PASSWORD')}@{os.getenv('PROXY_HOST')}:{os.getenv('PROXY_PORT')}",
        }
    return {}

def download_with_fallback(video_id, quality="high"):
    filename = f"{uuid.uuid4()}.mp3"
    output_path = os.path.join("downloads", filename)
    proxy_env = get_proxy_env()

    ytdlp_cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", quality,
        "-o", output_path
    ]

    try:
        subprocess.run(ytdlp_cmd, check=True, env={**os.environ, **proxy_env})
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"yt-dlp failed with proxy: {e}. Trying fallback...")

        # Try fallback without proxy
        try:
            subprocess.run(ytdlp_cmd, check=True, env=os.environ)
            return output_path
        except subprocess.CalledProcessError as e2:
            raise Exception(f"Download failed with and without proxy. {e2}")
