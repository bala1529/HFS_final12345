import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent

    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-in-production")

    UPLOAD_FOLDER = os.environ.get(
        "UPLOAD_FOLDER", str(BASE_DIR / "uploads")
    )
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

    # File type configuration
    ALLOWED_EXTENSIONS = {
        "txt",
        "pdf",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "wav",
        "mp3",
        "flac",
        "m4a",
        "ogg",
        "mp4",
        "avi",
        "mov",
        "mkv",
    }

    AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "m4a", "ogg"}
    VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
    IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

    # Feedback / SMTP configuration (Gmail with app password)
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "yourgmail@gmail.com")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "your-app-password")
    FEEDBACK_RECIPIENT = os.environ.get(
        "FEEDBACK_RECIPIENT", "huntforsolution@gmail.com"
    )

    # VirusTotal API key for URL checker – get free key: https://www.virustotal.com/gui/my-apikey
    # Always set via environment in production.
    VT_API_KEY = os.environ.get("VT_API_KEY", "")

