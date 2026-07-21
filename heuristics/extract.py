"""
Heuristic feature extraction for scam message/URL detection.
Fast, explainable, zero-cost checks that run before the LLM call.
"""
import re
import difflib
from urllib.parse import urlparse

# --- Reference data -----------------------------------------------------

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc",
}

SUSPICIOUS_TLDS = {".xyz", ".top", ".club", ".info", ".click", ".loan", ".work", ".gq", ".tk"}

# Brands commonly impersonated in Indian SMS/WhatsApp scams
KNOWN_BRANDS = {
    "sbi", "hdfc", "icicibank", "icici", "axisbank", "axis", "kotak",
    "paytm", "phonepe", "googlepay", "amazon", "flipkart", "irctc",
    "indiapost", "airtel", "jio", "vodafone", "whatsapp",
}

URGENCY_PHRASES = [
    r"\bact now\b", r"\bimmediately\b", r"\burgent\b", r"\bexpire[sd]?\b",
    r"\bwithin \d+ hours?\b", r"\blast chance\b", r"\bfinal notice\b",
    r"\bblocked\b", r"\bsuspend(ed)?\b", r"\bverify (your )?account\b",
]

CREDENTIAL_REQUEST_PHRASES = [
    r"\botp\b", r"\bpin\b", r"\bcvv\b", r"\bupi\s?id\b", r"\bpassword\b",
    r"\baadhaar\b", r"\bkyc\b", r"\bbank details\b", r"\bshare.{0,15}(otp|pin|code)\b",
]

PRIZE_PHRASES = [
    r"\byou('ve| have) won\b", r"\blottery\b", r"\bcongratulations\b.{0,20}\bwon\b",
    r"\bclaim (your )?(prize|reward)\b", r"\bcash prize\b", r"\bfree gift\b",
]

PHONE_REGEX = re.compile(r"(\+?91[\-\s]?)?[6-9]\d{9}\b")
URL_REGEX = re.compile(r"https?://[^\s]+|www\.[^\s]+")


def _extract_urls(text: str, provided_urls=None):
    urls = set(provided_urls or [])
    urls.update(URL_REGEX.findall(text))
    return list(urls)


def _analyze_url(url: str) -> dict:
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":")[0]

    flags = []

    if host in URL_SHORTENERS:
        flags.append(f"shortened_url:{host}")

    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host):
        flags.append("ip_based_url")

    for tld in SUSPICIOUS_TLDS:
        if host.endswith(tld):
            flags.append(f"suspicious_tld:{tld}")

    # crude lookalike/typosquat check against known brands
    host_core = host.replace("www.", "").split(".")[0]
    for brand in KNOWN_BRANDS:
        if host_core == brand:
            break  # exact match, not a lookalike
        ratio = difflib.SequenceMatcher(None, host_core, brand).ratio()
        if 0.75 <= ratio < 1.0:
            flags.append(f"lookalike_domain:{host_core}~{brand}")

    return {"url": url, "host": host, "flags": flags}


def extract_heuristics(text: str, urls: list = None) -> dict:
    """
    Returns structured heuristic signals for a message.
    risk_score is a simple count of distinct triggered categories -
    NOT a calibrated probability, just a fast pre-filter signal.
    """
    text_lower = text.lower()
    risk_factors = []

    if any(re.search(p, text_lower) for p in URGENCY_PHRASES):
        risk_factors.append("urgency_language")

    if any(re.search(p, text_lower) for p in CREDENTIAL_REQUEST_PHRASES):
        risk_factors.append("credential_or_otp_request")

    if any(re.search(p, text_lower) for p in PRIZE_PHRASES):
        risk_factors.append("prize_or_lottery_bait")

    phones = PHONE_REGEX.findall(text)
    if phones:
        risk_factors.append("contains_phone_number")

    all_urls = _extract_urls(text, urls)
    url_analyses = [_analyze_url(u) for u in all_urls]
    for ua in url_analyses:
        risk_factors.extend(ua["flags"])

    return {
        "risk_score": len(risk_factors),
        "risk_factors": risk_factors,
        "urls_found": all_urls,
        "url_analysis": url_analyses,
    }
