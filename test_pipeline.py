"""
Automated Verification Suite for PhishGuard AI.
Tests feature extraction, data loading, error handling, and model inference.
"""

import sys
import os

from src.features import URLFeatureExtractor
from src.utils import is_valid_url, normalize_url, is_ip_host


def test_utils():
    print("[*] Testing URL utils...")
    assert is_valid_url("https://www.google.com") == True
    assert is_valid_url("google.com") == True
    assert is_valid_url("http://192.168.1.1/login") == True
    assert is_valid_url("") == False
    assert is_valid_url("   ") == False
    assert is_valid_url("invalid with spaces") == False

    assert is_ip_host("192.168.1.1") == 1
    assert is_ip_host("google.com") == 0
    assert is_ip_host("10.0.0.1:8080") == 1
    print("  [+] Utils tests passed successfully.")


def test_feature_extractor():
    print("[*] Testing URLFeatureExtractor...")
    extractor = URLFeatureExtractor()
    assert len(extractor.FEATURE_NAMES) == 25

    # Test legitimate URL
    legit_url = "https://www.google.com/search?q=machine+learning"
    f_legit = extractor.extract_features(legit_url)
    assert f_legit["is_https"] == 1
    assert f_legit["is_ip"] == 0
    assert f_legit["url_length"] == len(legit_url)
    assert f_legit["dot_count"] >= 2
    assert f_legit["suspicious_keyword_count"] == 0

    # Test phishing URL with IP, @, double slash, and sensitive keyword
    phish_url = "http://admin:pass@192.168.1.100//paypal-login/update-verify.php"
    f_phish = extractor.extract_features(phish_url)
    assert f_phish["is_https"] == 0
    assert f_phish["is_ip"] == 1
    assert f_phish["at_count"] == 1
    assert f_phish["double_slash_path"] == 1
    assert f_phish["suspicious_keyword_count"] >= 2

    # Check risk indicator explanations
    reasons = extractor.explain_risk_indicators(f_phish)
    severities = [r["severity"] for r in reasons]
    assert "CRITICAL" in severities or "HIGH" in severities
    print("  [+] Feature extraction tests passed successfully.")


def test_predictor_if_model_exists():
    if os.path.exists("models/phishguard_best_model.joblib"):
        print("[*] Testing PhishGuardPredictor inference...")
        from src.model import PhishGuardPredictor
        predictor = PhishGuardPredictor(models_dir="models")

        res_legit = predictor.predict("https://www.wikipedia.org")
        assert res_legit["valid"] == True
        print(f"  Legitimate URL pred: {res_legit['prediction']} (Prob: {res_legit['phishing_probability']})")

        res_phish = predictor.predict("http://192.168.0.1/signin.paypal.com/verify-account.php")
        assert res_phish["valid"] == True
        print(f"  Phishing URL pred: {res_phish['prediction']} (Prob: {res_phish['phishing_probability']})")

        res_invalid = predictor.predict("not a url at all")
        assert res_invalid["valid"] == False
        print("  [+] Predictor tests passed successfully.")
    else:
        print("[!] Model file not yet created; skipping predictor inference test.")


if __name__ == "__main__":
    print("=" * 60)
    print("Running PhishGuard AI Verification Suite")
    print("=" * 60)
    test_utils()
    test_feature_extractor()
    test_predictor_if_model_exists()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
