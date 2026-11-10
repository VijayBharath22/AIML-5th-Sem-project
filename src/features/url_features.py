"""Deterministic URL feature extraction for PhreshPhish.

Features use only the URL string. Labels, dates, and identifiers are not inputs.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import parse_qs, urlparse

import tldextract

# Structural characters are not counted as "special". '@' is intentionally special.
_STRUCTURAL = set(".:/-_?=&%+")
_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

FEATURE_COLUMNS = [
    "url_length",
    "hostname_length",
    "path_length",
    "n_dots",
    "n_hyphens",
    "n_digits",
    "n_special_chars",
    "n_subdirectories",
    "n_query_params",
    "is_https",
    "is_ip_hostname",
    "n_suspicious_keywords",
    "n_subdomains",
    "has_at_symbol",
    "is_shortened",
]


def registrable_domain(value: str) -> str:
    """Return eTLD+1 for a URL or hostname. Empty string if it cannot be derived."""
    if not isinstance(value, str) or not value.strip():
        return ""
    text = value.strip()
    host = text
    if "://" in text or text.startswith("//") or "/" in text:
        try:
            parsed = urlparse(text if "://" in text else f"//{text}" if text.startswith("//") else f"http://{text}")
            host = parsed.hostname or ""
        except (ValueError, Exception):
            return ""
    host = host.strip().lower()
    if not host:
        return ""
    if _is_ip(host):
        return host
    try:
        extracted = _EXTRACTOR(host)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}".lower()
        return (extracted.domain or host).lower()
    except Exception:
        return host


def _is_ip(hostname: str) -> bool:
    candidate = hostname.strip("[]")
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        return False


def url_is_usable(url: str) -> bool:
    """True when the URL has an http(s) scheme and a hostname."""
    if not isinstance(url, str):
        return False
    raw = url.strip()
    if not raw or len(raw) > 8000:
        return False
    try:
        parsed = urlparse(raw)
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)
    except (ValueError, Exception):
        return False


def extract_url_features(url: str, keywords: list[str], shorteners: list[str]) -> dict | None:
    """Return features, or None if the URL is missing a scheme or hostname."""
    if not isinstance(url, str):
        return None
    raw = url.strip()
    if not raw or len(raw) > 8000:
        return None
    try:
        parsed = urlparse(raw)
    except (ValueError, Exception):
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    hostname = parsed.hostname.lower()
    path = parsed.path or ""
    query = parsed.query or ""
    lowered = raw.lower()
    special = sum(1 for char in raw if not char.isalnum() and char not in _STRUCTURAL)
    try:
        subdomain_text = _EXTRACTOR(hostname).subdomain
        n_subdomains = len([part for part in subdomain_text.split(".") if part]) if subdomain_text else 0
    except Exception:
        n_subdomains = 0
    host_for_short = hostname[4:] if hostname.startswith("www.") else hostname
    shortened = int(any(host_for_short == item or host_for_short.endswith("." + item) for item in shorteners))
    keyword_hits = sum(1 for word in keywords if word in lowered)
    directories = [part for part in path.split("/") if part]
    try:
        n_params = len(parse_qs(query, keep_blank_values=True))
    except ValueError:
        n_params = 0
    return {
        "url_length": len(raw),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "n_dots": raw.count("."),
        "n_hyphens": raw.count("-"),
        "n_digits": sum(char.isdigit() for char in raw),
        "n_special_chars": special,
        "n_subdirectories": len(directories),
        "n_query_params": n_params,
        "is_https": int(parsed.scheme.lower() == "https"),
        "is_ip_hostname": int(_is_ip(hostname)),
        "n_suspicious_keywords": keyword_hits,
        "n_subdomains": 0 if _is_ip(hostname) else n_subdomains,
        "has_at_symbol": int("@" in raw),
        "is_shortened": shortened,
        "registrable_domain": registrable_domain(raw),
    }


def feature_definitions() -> dict[str, str]:
    return {
        "url_length": "Character length of the raw URL.",
        "hostname_length": "Character length of the parsed hostname.",
        "path_length": "Character length of the URL path.",
        "n_dots": "Count of '.' characters in the raw URL.",
        "n_hyphens": "Count of '-' characters in the raw URL.",
        "n_digits": "Count of digit characters in the raw URL.",
        "n_special_chars": "Count of characters that are not alphanumeric and not in .:/-_?=&%+.",
        "n_subdirectories": "Count of non-empty path segments.",
        "n_query_params": "Count of query-string keys.",
        "is_https": "1 if the scheme is https, else 0.",
        "is_ip_hostname": "1 if the hostname is an IP address, else 0.",
        "n_suspicious_keywords": "Count of distinct configured keywords present as substrings.",
        "n_subdomains": "Number of subdomain labels from the public-suffix parse.",
        "has_at_symbol": "1 if '@' appears in the URL, else 0.",
        "is_shortened": "1 if the hostname is a configured shortening service, else 0.",
    }
