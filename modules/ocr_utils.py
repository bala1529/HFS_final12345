from __future__ import annotations

from pathlib import Path
import os
import shutil

from PIL import Image
import pytesseract

# On Windows, explicitly set the Tesseract executable if it's installed
# in the default location but not found on PATH.
if os.name == "nt":
    if not shutil.which("tesseract"):
        default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_path):
            pytesseract.pytesseract.tesseract_cmd = default_path

# Prefix returned when OCR fails so callers can detect it
OCR_FAILED_PREFIX = "[OCR failed:"


def is_ocr_failed(extracted_text: str) -> bool:
    """True if the string is an OCR error message, not real extracted text."""
    if not extracted_text:
        return False
    t = extracted_text.strip()
    return t.startswith(OCR_FAILED_PREFIX) or "tesseract" in t.lower()


def is_no_text(extracted_text: str) -> bool:
    """True when OCR succeeded but no readable text was found."""
    return not extracted_text or not extracted_text.strip()


def extract_text_from_image(image_path: Path) -> str:
    """
    Extract text from an image using Tesseract OCR.
    On failure returns a message starting with [OCR failed: ...]; use is_ocr_failed() to detect.
    """
    try:
        image = Image.open(str(image_path))
        text = pytesseract.image_to_string(image)
        return (text or "").strip()
    except Exception as exc:
        return f"[OCR failed: {exc}]"