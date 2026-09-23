"""
Feature Extraction Engine for PhishGuard AI:
Extracts lexical, structural, statistical, and keyword features from raw URLs.
Includes industry-standard domain normalization (handling www. prefixes and subdomains).
"""

import math
import re
from typing import Dict, List, Any, Iterable
import pandas as pd

from src.utils import normalize_url, parse_url_components, is_ip_host


# Known URL shortening domains frequently leveraged by threat actors
KNOWN_SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "mcaf.ee", "rebrand.ly", "shorte.st",
    "trib.al", "clck.ru", "cutt.ly", "rb.gy"
}

# Suspicious keywords targeted in credential harvesting & financial spoofing
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "verification", "update", "banking",
    "secure", "security", "account", "password", "credential", "confirm",
    "confirmation", "wallet", "paypal", "ebay", "appleid", "support",
    "auth", "authentication", "billing", "service", "recover", "webscr"
]


def calculate_entropy(text: str) -> float:
    """
    Compute Shannon entropy of the string.
    High entropy often indicates obfuscation, randomized subdomains, or encrypted slugs.
    """
    if not text:
        return 0.0
    freq: Dict[str, int] = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    
    length = len(text)
    entropy = 0.0
    for count in freq.values():
        prob = count / length
        entropy -= prob * math.log2(prob)
    return round(entropy, 4)


class URLFeatureExtractor:
    """
    Extracts rich numerical features from any raw URL string.
    Applies domain normalization to prevent www vs apex domain bias.
    """
    
    FEATURE_NAMES = [
        "url_length",
        "dot_count",
        "hyphen_count",
        "slash_count",
        "question_count",
        "equal_count",
        "at_count",
        "exclamation_count",
        "ampersand_count",
        "percent_count",
        "double_slash_path",
        "is_https",
        "is_ip",
        "subdomain_count",
        "domain_length",
        "digit_count",
        "digit_ratio",
        "letter_count",
        "letter_ratio",
        "special_char_count",
        "special_char_ratio",
        "url_entropy",
        "is_shortener",
        "suspicious_keyword_count",
        "has_suspicious_keyword"
    ]

    def __init__(self):
        pass

    def extract_features(self, raw_url: str) -> Dict[str, Any]:
        """
        Extract feature dictionary for a single URL string.
        """
        url = str(raw_url).strip()
        url_lower = url.lower()
        url_len = max(len(url), 1)

        scheme, netloc, hostname, path, query = parse_url_components(url)
        host_target = (hostname if hostname else netloc).lower()
        
        # Domain normalization: strip leading www. for uniform evaluation
        host_normalized = re.sub(r"^www\.", "", host_target)

        # 1. Lexical & Character counts
        dot_count = url.count(".")
        hyphen_count = url.count("-")
        slash_count = url.count("/")
        question_count = url.count("?")
        equal_count = url.count("=")
        at_count = url.count("@")
        exclamation_count = url.count("!")
        ampersand_count = url.count("&")
        percent_count = url.count("%")

        # 2. Structural checks
        # Path contains // after the scheme
        double_slash_path = 1 if "//" in (path + ("?" + query if query else "")) else 0
        is_https = 1 if scheme == "https" else 0
        is_ip = is_ip_host(host_target)

        # Subdomains calculation on normalized host (e.g. login.evil.com -> 1 subdomain)
        domain_parts = host_normalized.split(".") if host_normalized else []
        subdomain_count = max(0, len(domain_parts) - 2) if len(domain_parts) >= 2 else 0
        domain_length = len(host_normalized)

        # 3. Statistical character distributions
        digits = sum(c.isdigit() for c in url)
        letters = sum(c.isalpha() for c in url)
        special_chars = url_len - (digits + letters)

        digit_ratio = round(digits / url_len, 4)
        letter_ratio = round(letters / url_len, 4)
        special_char_ratio = round(special_chars / url_len, 4)

        # 4. Information theory / Entropy
        entropy = calculate_entropy(url)

        # 5. Shortener detection
        is_shortener = 1 if any(shortener in host_normalized for shortener in KNOWN_SHORTENERS) else 0

        # 6. Keyword matching
        keyword_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
        has_suspicious_keyword = 1 if keyword_hits > 0 else 0

        return {
            "url_length": url_len,
            "dot_count": dot_count,
            "hyphen_count": hyphen_count,
            "slash_count": slash_count,
            "question_count": question_count,
            "equal_count": equal_count,
            "at_count": at_count,
            "exclamation_count": exclamation_count,
            "ampersand_count": ampersand_count,
            "percent_count": percent_count,
            "double_slash_path": double_slash_path,
            "is_https": is_https,
            "is_ip": is_ip,
            "subdomain_count": subdomain_count,
            "domain_length": domain_length,
            "digit_count": digits,
            "digit_ratio": digit_ratio,
            "letter_count": letters,
            "letter_ratio": letter_ratio,
            "special_char_count": special_chars,
            "special_char_ratio": special_char_ratio,
            "url_entropy": entropy,
            "is_shortener": is_shortener,
            "suspicious_keyword_count": keyword_hits,
            "has_suspicious_keyword": has_suspicious_keyword
        }

    def transform(self, urls: Iterable[str]) -> pd.DataFrame:
        """
        Extract features for an iterable of URLs and return a clean DataFrame.
        """
        records = [self.extract_features(u) for u in urls]
        return pd.DataFrame(records, columns=self.FEATURE_NAMES)

    def explain_risk_indicators(self, features: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Translate extracted features into human-readable cybersecurity risk indicators.
        """
        indicators = []

        if features.get("is_ip", 0) == 1:
            indicators.append({
                "severity": "CRITICAL",
                "title": "Raw IP Address in Hostname",
                "desc": "The URL uses an IP address instead of a domain name, a common phishing tactic to evade domain registry reputation checks."
            })

        if features.get("at_count", 0) > 0:
            indicators.append({
                "severity": "CRITICAL",
                "title": "Embedded '@' Symbol in URL",
                "desc": "The presence of '@' in a URL causes browsers to ignore everything preceding it, obscuring the true destination."
            })

        if features.get("double_slash_path", 0) == 1:
            indicators.append({
                "severity": "HIGH",
                "title": "Double Slash in Path Redirection",
                "desc": "Double slashes inside the URL path are commonly used to redirect visitors to an external attacker-controlled landing page."
            })

        if features.get("suspicious_keyword_count", 0) >= 2:
            indicators.append({
                "severity": "HIGH",
                "title": f"Multiple Credential/Auth Keywords ({features['suspicious_keyword_count']})",
                "desc": "Contains targeted sensitive keywords (e.g. login, verify, secure, banking, paypal) characteristic of credential harvesting."
            })
        elif features.get("suspicious_keyword_count", 0) == 1:
            indicators.append({
                "severity": "MEDIUM",
                "title": "Suspicious Authentication Keyword Detected",
                "desc": "Contains a sensitive keyword often associated with authentication or account management."
            })

        if features.get("is_shortener", 0) == 1:
            indicators.append({
                "severity": "HIGH",
                "title": "URL Shortening Service",
                "desc": "The link uses a known URL shortening service, masking the final destination and bypassing perimeter filters."
            })

        if features.get("url_length", 0) > 85:
            indicators.append({
                "severity": "MEDIUM",
                "title": f"Excessive URL Length ({features['url_length']} chars)",
                "desc": "Abnormally long URLs are frequently used in phishing campaigns to hide suspicious tokens or embed payloads."
            })

        if features.get("subdomain_count", 0) >= 2:
            indicators.append({
                "severity": "MEDIUM",
                "title": f"High Subdomain Count ({features['subdomain_count']} subdomains)",
                "desc": "Multiple levels of subdomains are often created to mimic brand names on free hosting platforms."
            })

        if features.get("dot_count", 0) >= 4:
            indicators.append({
                "severity": "MEDIUM",
                "title": f"Excessive Dots in URL ({features['dot_count']} dots)",
                "desc": "A high number of dot delimiters often signifies deep subdomains or confusing hostnames."
            })

        if features.get("url_entropy", 0.0) >= 4.4:
            indicators.append({
                "severity": "MEDIUM",
                "title": f"High Character Entropy ({features['url_entropy']})",
                "desc": "Character distribution indicates significant randomness or obfuscated encoding."
            })

        if features.get("is_https", 0) == 0:
            indicators.append({
                "severity": "LOW",
                "title": "Plain HTTP Protocol",
                "desc": "The URL does not use encrypted HTTPS communication."
            })

        if not indicators:
            indicators.append({
                "severity": "SAFE",
                "title": "Standard Lexical Profile",
                "desc": "No anomalous lexical anomalies, raw IP hosts, shorteners, or deceptive redirection patterns were identified."
            })

        return indicators
