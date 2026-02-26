from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import joblib

from modules import url_checker

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

_VECTORIZER = None
_MODEL = None

# Extract URLs from text: https://... or http://... or www....
_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    re.IGNORECASE,
)


def _normalize_for_url_extraction(text: str) -> str:
    """
    OCR often splits URLs across lines, e.g.:
      'http://\\ncardssbi.com/'
    Normalize common break patterns so URL extraction can still detect links.
    """
    if not text:
        return ""

    # Join scheme with next token(s): http://\nexample.com -> http://example.com
    text = re.sub(r"(https?://)\s+", r"\1", text, flags=re.IGNORECASE)

    # Join www. with next token(s): www.\nexample.com -> www.example.com
    text = re.sub(r"(www\.)\s+", r"\1", text, flags=re.IGNORECASE)

    # Collapse spaces around dots in URL-ish contexts (helps OCR like 'sbi . com')
    text = re.sub(r"(\b(?:https?://|www\.)[^\s]{0,200})\s*\.\s*([A-Za-z0-9])", r"\1.\2", text)
    return text


def _load_models() -> None:
    global _VECTORIZER, _MODEL
    if _VECTORIZER is not None and _MODEL is not None:
        return

    vectorizer_path = MODELS_DIR / "vectorizer.pkl"
    model_path = MODELS_DIR / "spam_model.pkl"

    if not vectorizer_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Required model files not found in {MODELS_DIR}. "
            "Expected 'vectorizer.pkl' and 'spam_model.pkl'."
        )

    _VECTORIZER = joblib.load(vectorizer_path)
    _MODEL = joblib.load(model_path)


def _predict_proba(message: str) -> Optional[float]:
    try:
        _load_models()
    except Exception:
        return None

    if not message:
        return 0.0

    features = _VECTORIZER.transform([message])
    proba = getattr(_MODEL, "predict_proba", None)
    if proba is None:
        pred = _MODEL.predict(features)[0]
        return float(pred)

    spam_prob = proba(features)[0][1]
    return float(spam_prob)


def _extract_urls(text: str) -> list[str]:
    """Return list of URL strings found in text."""
    if not text or not text.strip():
        return []
    normalized = _normalize_for_url_extraction(text)
    return _URL_PATTERN.findall(normalized)


def _domain_from_url(url: str) -> str:
    """Normalize URL to domain (lowercase, no www)."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        parsed = urlparse(url)
        domain = (parsed.netloc or parsed.path).split(":")[0].lower().replace("www.", "")
        return domain or ""
    except Exception:
        return ""


def analyze_text(message: str) -> dict:
    """
    Analyze SMS / text: spam model + link check.
    If risk > 20% → Fake/Dangerous with increased percentage.
    If message contains a link, check domain against trusted/blacklist; fake link → Fake.
    """
    spam_prob = _predict_proba(message)

    if spam_prob is None:
        return {
            "risk": 50.0,
            "confidence": 30.0,
            "details": "Spam detection model not available. Default heuristic result used.",
        }

    risk = max(0.0, min(100.0, spam_prob * 100.0))
    confidence = 80.0
    details_parts = []

    # Check for links in message
    urls = _extract_urls(message)
    link_fake = False
    link_untrusted = False
    link_trusted = False
    checked_links = []

    for raw_url in urls:
        domain = _domain_from_url(raw_url)
        if not domain:
            continue
        verdict = url_checker.check_domain_verdict(domain)
        checked_links.append((raw_url, domain, verdict))
        if verdict == "blacklisted":
            link_fake = True
        elif verdict == "unknown":
            link_untrusted = True
        else:
            link_trusted = True

    if link_fake:
        risk = 100.0
        details_parts.append("Links in message:")
        idx = 1
        for raw_url, domain, _ in checked_links:
            verdict = url_checker.check_domain_verdict(domain)
            if verdict == "blacklisted":
                details_parts.append(f"  {idx}. [FAKE] {raw_url} (domain: {domain})")
                idx += 1
    elif link_untrusted:
        risk = max(risk, 55.0)
        details_parts.append("Links in message:")
        idx = 1
        for raw_url, domain, v in checked_links:
            if v == "unknown":
                details_parts.append(f"  {idx}. [UNTRUSTED] {raw_url} (domain: {domain})")
                idx += 1
    elif link_trusted and checked_links:
        details_parts.append("Links in message:")
        idx = 1
        for raw_url, domain, v in checked_links:
            if v == "trusted":
                details_parts.append(f"  {idx}. [TRUSTED] {raw_url} (domain: {domain})")
                idx += 1

    # NOTE: We keep the 20% threshold logic in app.py for status,
    # but do not mention it explicitly in the explanation text.
    label = "SPAM / FRAUDULENT" if risk >= 50 else ("FAKE" if risk > 20 else "LIKELY SAFE")

    # Format original text in an ordered way (line by line)
    lines = [ln.strip() for ln in str(message).splitlines() if ln.strip()]
    if not lines:
        formatted_text_block = "[no text]"
    elif len(lines) == 1:
        formatted_text_block = f"1. {lines[0]}"
    else:
        formatted_text_block = "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(lines))

    details_parts = [
        f"Message classified as: {label}",
        "Extracted / original text (ordered):",
        formatted_text_block,
    ] + details_parts

    details = "\n".join(details_parts)

    return {
        "risk": float(risk),
        "confidence": float(confidence),
        "details": details,
    }
