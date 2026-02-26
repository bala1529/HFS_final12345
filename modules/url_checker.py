from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
import whois  # python-whois

try:
    from config import Config
    VT_API_KEY = getattr(Config, "VT_API_KEY", "") or ""
except Exception:
    import os
    VT_API_KEY = os.environ.get("VT_API_KEY", "")

# Known-bad: phishing, piracy, malicious. Match if domain contains any of these.
BLACKLIST = [
    "malicious-site.com",
    "phishing-test.xyz",
    "isaimini.spot",
    "isaimini.com",
    "netmirror.gg",
]

# Known-good: major brands, tech, gov, education, news, finance, health.
# Result = only domains that match this list get "Safe"; others are evaluated by signals.
TRUSTED_DOMAINS = [
    # Tech & search
    "google.com", "google.co.in", "youtube.com", "gmail.com", "drive.google.com",
    "microsoft.com", "outlook.com", "live.com", "bing.com", "linkedin.com",
    "apple.com", "icloud.com", "github.com", "gitlab.com", "stackoverflow.com",
    "mozilla.org", "firefox.com", "duckduckgo.com", "wikipedia.org", "wikimedia.org",
    "cloudflare.com", "wordpress.com", "medium.com", "reddit.com", "quora.com",
    "yahoo.com", "brave.com", "dropbox.com", "zoom.us", "slack.com", "notion.so",
    "atlassian.com", "trello.com", "figma.com", "canva.com", "adobe.com",
    "salesforce.com", "ibm.com", "oracle.com", "cisco.com", "intel.com", "amd.com",
    "nvidia.com", "samsung.com", "sony.com", "hp.com", "dell.com", "lenovo.com",
    "asus.com", "acer.com", "qualcomm.com",
    # Social & messaging
    "facebook.com", "fb.com", "instagram.com", "whatsapp.com", "wa.me",
    "twitter.com", "x.com", "telegram.org", "discord.com", "snapchat.com",
    "pinterest.com", "tumblr.com", "vk.com", "weibo.com",
    # E‑commerce & payments
    "amazon.com", "amazon.in", "amazon.co.uk", "flipkart.com", "ebay.com",
    "paypal.com", "stripe.com", "aliexpress.com", "alibaba.com",
    "walmart.com", "target.com", "bestbuy.com", "shopify.com",
    # Streaming & media
    "netflix.com", "spotify.com", "disneyplus.com", "hulu.com", "primevideo.com",
    "twitch.tv", "vimeo.com", "soundcloud.com", "bandcamp.com", "imdb.com",
    # News & media
    "bbc.com", "bbc.co.uk", "cnn.com", "reuters.com", "apnews.com", "npr.org",
    "nytimes.com", "washingtonpost.com", "theguardian.com", "wsj.com",
    "bloomberg.com", "forbes.com", "economist.com", "ft.com", "aljazeera.com",
    "ndtv.com", "indiatimes.com", "thehindu.com", "indianexpress.com",
    "moneycontrol.com", "economictimes.indiatimes.com", "hindustantimes.com",
    # Gov & education
    "gov.in", "gov.uk", "usa.gov", "gov.au", "canada.ca", "europa.eu",
    "who.int", "cdc.gov", "nih.gov", "un.org", "w3.org",
    "edu", "ac.in", "ac.uk", "edu.au", "coursera.org", "edx.org", "udemy.com",
    "khanacademy.org", "ted.com", "khanacademy.org",
    # Indian banks & finance
    "hdfcbank.com", "icicibank.com", "sbi.co.in", "axisbank.com", "kotak.com",
    "pnb.co.in", "bankofbaroda.in", "canarabank.com", "idfcfirstbank.com",
    "paytm.com", "phonepe.com", "gpay.google.com", "cred.club",
    # Other
    "weather.com", "accuweather.com", "goodreads.com", "rottentomatoes.com",
    "patreon.com", "kickstarter.com", "gofundme.com", "indiegogo.com",
]

# Shorteners: often used in phishing; increase risk when domain is unknown.
SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly"]

# TLDs often used for spam/phishing; extra caution when domain is unknown.
SUSPICIOUS_TLDS = (".xyz", ".top", ".work", ".click", ".link", ".gq", ".ml", ".cf", ".tk", ".ga")


def _domain_age_in_days(domain_info) -> float:
    created = domain_info.get("creation_date")
    if isinstance(created, list):
        created = created[0]
    if not created:
        return 0.0
    if isinstance(created, datetime):
        if created.tzinfo is not None:
            created = created.astimezone(timezone.utc).replace(tzinfo=None)
        now = datetime.utcnow()
        return (now.date() - created.date()).days
    try:
        parsed = datetime.strptime(str(created), "%Y-%m-%d")
        return (datetime.utcnow().date() - parsed.date()).days
    except Exception:
        return 0.0


def _get_vt_domain_info(domain: str) -> tuple[str, str, str, int]:
    """Returns (owner, age_str, vt_status, malicious_count)."""
    owner = "Not Public"
    age = "Not Available"
    vt_status = "Not Checked"
    malicious_count = 0

    if not VT_API_KEY or VT_API_KEY.strip() == "PASTE_YOUR_VIRUSTOTAL_API_KEY_HERE":
        return owner, age, "Not configured (set VT_API_KEY in config.py or env)", 0

    try:
        headers = {"x-apikey": VT_API_KEY}
        vt_url = f"https://www.virustotal.com/api/v3/domains/{domain}"
        r = requests.get(vt_url, headers=headers, timeout=10)

        if r.status_code != 200:
            return owner, age, "VT Not Available", 0

        data = r.json()
        if "data" not in data:
            return owner, age, vt_status, malicious_count

        attrs = data["data"]["attributes"]
        owner = attrs.get("registrar", owner)

        if "creation_date" in attrs:
            creation = datetime.fromtimestamp(attrs["creation_date"])
            age_days = (datetime.now() - creation).days
            age = f"{age_days // 365} years ({age_days} days)"

        stats = attrs.get("last_analysis_stats", {})
        malicious_count = stats.get("malicious", 0)
        if malicious_count > 0:
            vt_status = f"⚠ Flagged ({malicious_count} engines)"
        else:
            vt_status = "Clean"

        return owner, age, vt_status, malicious_count
    except Exception:
        return owner, age, "VT Failed", 0


def _is_trusted(domain: str) -> bool:
    return any(t in domain for t in TRUSTED_DOMAINS)


def _is_blacklisted(domain: str) -> bool:
    return any(b in domain for b in BLACKLIST)


def check_domain_verdict(domain: str) -> str:
    """
    Check a domain against trusted/blacklist. Used by SMS checker for links in text.
    Returns "trusted" | "blacklisted" | "unknown".
    """
    if not domain or not domain.strip():
        return "unknown"
    d = domain.lower().strip().replace("www.", "").split("/")[0].split(":")[0]
    if _is_blacklisted(d):
        return "blacklisted"
    if _is_trusted(d):
        return "trusted"
    return "unknown"


def _is_shortener(domain: str) -> bool:
    return any(s in domain for s in SHORTENERS)


def _has_suspicious_tld(domain: str) -> bool:
    return any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS)


def analyze_url(url: str) -> dict:
    """
    Analyze URL using: trusted list, blacklist, VirusTotal, WHOIS, HTTPS, age,
    shorteners, TLD. Only domains in the trusted list get Safe by default;
    others are scored from multiple signals.
    Returns {"risk": float, "confidence": float, "details": str}.
    """
    if not url or not url.strip():
        return {
            "risk": 100.0,
            "confidence": 100.0,
            "details": "No URL provided.",
        }

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    domain = (parsed.netloc or parsed.path).split(":")[0].lower().replace("www.", "")

    details_parts = []
    risk = 0.0
    confidence = 70.0

    # 1) Blacklist → always high risk
    if _is_blacklisted(domain):
        return {
            "risk": 100.0,
            "confidence": 95.0,
            "details": (
                f"Domain '{domain}' is on the block list (known bad).\n"
                "These domains are not in the trusted set and are marked as risky."
            ),
        }

    # 2) Trusted list → low risk unless VT strongly says otherwise
    is_trusted = _is_trusted(domain)
    if is_trusted:
        details_parts.append("Domain is in the trusted list (known legitimate site).")
        # Start safe; only VT can push risk up a bit
        risk = 5.0
        confidence = 90.0
    else:
        # 3) Unknown domain → score from many signals (start higher)
        risk = 45.0
        confidence = 75.0
        details_parts.append("Domain is not in the trusted list (unknown site).")

    # Gather WHOIS (owner + age)
    owner = "N/A"
    age_str = "Not available"
    age_days = -1.0
    try:
        domain_info = whois.whois(domain)
        if getattr(domain_info, "org", None):
            owner = domain_info.org
        elif getattr(domain_info, "registrar", None):
            owner = domain_info.registrar
        if owner is None:
            owner = "N/A"

        creation = getattr(domain_info, "creation_date", None)
        if isinstance(creation, list):
            creation = creation[0] if creation else None
        if creation:
            if isinstance(creation, datetime):
                age_days = (datetime.now() - creation).days
                age_str = f"{age_days // 365} years ({age_days} days)"
            else:
                # Try parsing string dates from WHOIS
                for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%Y.%m.%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%SZ"):
                    try:
                        dt = datetime.strptime(str(creation).strip()[:10], fmt) if len(str(creation)) >= 10 else None
                        if dt:
                            age_days = (datetime.now() - dt).days
                            age_str = f"{age_days // 365} years ({age_days} days)"
                            break
                    except Exception:
                        continue
    except Exception:
        pass

    # VirusTotal
    vt_owner, vt_age, vt_status, malicious_count = _get_vt_domain_info(domain)
    if owner in ("N/A", "Privacy Protected", None):
        owner = vt_owner
    if age_str in ("N/A", "Not available", "Not Available") and vt_age not in ("N/A", "Not Available", "Not available"):
        age_str = vt_age
    if age_days < 0 and vt_age and "days" in str(vt_age):
        try:
            age_days = int(vt_age.split("(")[1].split(" ")[0])
        except Exception:
            age_days = 0

    # Apply VT to risk
    if malicious_count > 0:
        if is_trusted:
            risk += 15.0
            details_parts.append("VirusTotal reported flags; trusted domain (possible false positive).")
        else:
            risk += 40.0
            details_parts.append("VirusTotal reported malicious flags.")

    # HTTPS
    if parsed.scheme != "https":
        risk += 20.0
        details_parts.append("Connection is not HTTPS (insecure).")
    else:
        if not is_trusted:
            risk -= 10.0
        details_parts.append("HTTPS: Yes")

    # Domain age (only for unknown)
    if not is_trusted and age_days >= 0:
        if age_days < 30:
            risk += 25.0
            details_parts.append("Domain very new (< 30 days) – higher risk.")
        elif age_days < 365:
            risk += 10.0
            details_parts.append("Domain less than 1 year old.")

    # Shortener
    if _is_shortener(domain):
        risk += 15.0
        details_parts.append("URL uses a shortener (often used in phishing).")

    # Suspicious TLD (unknown only)
    if not is_trusted and _has_suspicious_tld(domain):
        risk += 10.0
        details_parts.append("Domain uses a high-risk TLD.")

    # WHOIS: has real org (unknown only)
    if not is_trusted and owner not in ("N/A", "Not Public", "Privacy Protected", None):
        risk -= 5.0

    # Reachability (optional signal)
    try:
        requests.get(url, timeout=5)
    except Exception:
        if not is_trusted:
            risk += 5.0
        details_parts.append("URL could not be reached (timeout or connection failed).")

    # Clamp risk and decide status (35+ → Fake so app can show higher %)
    risk = max(0.0, min(100.0, risk))
    if risk <= 15:
        status = "Safe"
    elif risk < 35:
        status = "Low Risk"
    elif risk <= 60:
        status = "Fake"
    else:
        status = "Dangerous"

    if VT_API_KEY:
        confidence = min(95.0, confidence + 10.0)

    # Ensure Age and VirusTotal are always visible
    if age_str in ("N/A", "Not available", "Not Available"):
        age_str = "Not available (set VT_API_KEY in config.py for VirusTotal age)"
    details_parts = [
        f"Domain: {domain}",
        f"Status: {status}",
        f"Owner/Registrar: {owner}",
        f"Age: {age_str}",
        f"VirusTotal: {vt_status}",
    ] + [p for p in details_parts if not p.startswith("VirusTotal:")]

    return {
        "risk": float(risk),
        "confidence": float(confidence),
        "details": "\n".join(details_parts),
    }
