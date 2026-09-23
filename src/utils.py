"""
URL utilities for PhishGuard AI:
Validation, parsing, canonicalization, and normalization helpers.
"""

import re
from urllib.parse import urlparse, urlunparse
from typing import Tuple, Optional


IPV4_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)

IPV6_PATTERN = re.compile(
    r"^\[?(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}\]?$"
)


def normalize_url(raw_url: str) -> str:
    """
    Clean, canonicalize, and normalize URL for robust analysis.
    Adds scheme if missing, strips whitespace, and standardizes 'www.' host prefixes
    to prevent domain-form bias across web addresses.
    """
    if not isinstance(raw_url, str):
        return ""
    
    url = raw_url.strip()
    if not url:
        return ""
    
    # If no scheme is present, add http:// for uniform parsing
    if not (url.startswith("http://") or url.startswith("https://") or url.startswith("ftp://")):
        url = "http://" + url
    
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower() if parsed.scheme else "http"
        netloc = parsed.netloc.lower() if parsed.netloc else ""

        # Normalize www. prefix on hostname (e.g. www.google.com -> google.com)
        if netloc.startswith("www."):
            netloc = netloc[4:]

        path = parsed.path
        return urlunparse((scheme, netloc, path, parsed.params, parsed.query, parsed.fragment))
    except Exception:
        return url


def is_valid_url(raw_url: str) -> bool:
    """
    Check if the input string can be parsed into a viable URL or hostname.
    """
    if not isinstance(raw_url, str):
        return False
    
    cleaned = raw_url.strip()
    if not cleaned or len(cleaned) < 3 or " " in cleaned:
        return False
    
    normalized = normalize_url(cleaned)
    try:
        parsed = urlparse(normalized)
        return bool(parsed.netloc or parsed.path)
    except Exception:
        return False


def parse_url_components(url: str) -> Tuple[str, str, str, str, str]:
    """
    Safely extract scheme, netloc, hostname, path, and query components.
    """
    normalized = normalize_url(url)
    try:
        parsed = urlparse(normalized)
        scheme = parsed.scheme.lower() if parsed.scheme else ""
        netloc = parsed.netloc.lower() if parsed.netloc else ""
        hostname = (parsed.hostname or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
        return scheme, netloc, hostname, path, query
    except Exception:
        return "", "", "", "", ""


def is_ip_host(netloc_or_host: str) -> int:
    """
    Check if the hostname or netloc contains an IPv4 or IPv6 address.
    Phishing URLs frequently use raw IP addresses to bypass domain reputation systems.
    """
    if not netloc_or_host:
        return 0
    # Strip user:pass if present
    host = netloc_or_host.split("@")[-1]
    # Strip port if present
    if ":" in host and not host.startswith("["):
        host = host.split(":")[0]
    host = host.strip("[]")

    if IPV4_PATTERN.match(host) or IPV6_PATTERN.match(host):
        return 1
    return 0
