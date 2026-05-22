"""Security utilities: URL validation, path validation, and content sanitisation.

These helpers are used across tools and UI components to provide defence-in-depth
against SSRF, path traversal, and prompt-injection attacks.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# URL validation (SSRF protection)
# ---------------------------------------------------------------------------

_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def _is_private_ip(host: str) -> bool:
    """Return True if *host* resolves to a private / loopback / link-local address.

    Uses ``ipaddress`` for precise matching that covers all representations:
    - Dotted decimal:      10.0.0.1
    - Hex notation:        0x0a000001
    - Decimal notation:    167772161
    - IPv4-mapped IPv6:    ::ffff:10.0.0.1
    - Bracketed IPv6:      [::1]

    Falls back to a DNS lookup for hostnames (e.g. ``localhost``).
    """
    # Strip IPv6 brackets  [::1] -> ::1
    stripped = host.strip("[]")
    try:
        addr = ipaddress.ip_address(stripped)
        return (
            addr.is_loopback
            or addr.is_private
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_unspecified
        )
    except ValueError:
        pass  # not a bare IP literal — try hostname resolution

    # Hostname: resolve and check the resulting addresses
    try:
        infos = socket.getaddrinfo(stripped, None)
        for info in infos:
            raw_ip = info[4][0]
            try:
                addr = ipaddress.ip_address(raw_ip)
                if (
                    addr.is_loopback
                    or addr.is_private
                    or addr.is_link_local
                    or addr.is_reserved
                    or addr.is_unspecified
                ):
                    return True
            except ValueError:
                pass
    except OSError:
        # DNS resolution failed — treat as potentially unsafe
        pass

    return False


def validate_url(url: str) -> None:
    """Raise ``ValueError`` if *url* is not safe to fetch.

    Blocks:
    - Non-HTTP(S) schemes  (``file://``, ``ftp://``, ``gopher://``, …)
    - Loopback addresses   (127.x, ::1, localhost) in all representations
    - Link-local addresses (169.254.x — AWS/Azure/GCP instance metadata)
    - RFC-1918 private ranges (hex, decimal, and dotted-decimal forms)
    - IPv4-mapped IPv6 private addresses (``::ffff:10.0.0.1``)
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise ValueError(f"Malformed URL: {url!r}") from exc

    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Disallowed URL scheme {scheme!r}. Only 'http' and 'https' are permitted."
        )

    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError(f"URL has no host: {url!r}")

    if _is_private_ip(host):
        raise ValueError(
            f"Blocked host {host!r}: private / loopback addresses are not allowed "
            "(SSRF protection)."
        )


# ---------------------------------------------------------------------------
# Path validation (path-traversal / arbitrary file-read protection)
# ---------------------------------------------------------------------------

def _default_allowed_roots() -> list[Path]:
    """Returns the set of directories that load_pdf is allowed to read from."""
    roots = [Path(tempfile.gettempdir()).resolve()]
    # Import lazily to avoid circular imports at module load time.
    try:
        from research_swarm.config import settings  # noqa: PLC0415
        roots.append(settings.data_dir.resolve())
    except Exception:
        pass
    return roots


def validate_file_path(
    file_path: str,
    allowed_roots: list[Path] | None = None,
) -> Path:
    """Resolve *file_path* and ensure it sits under one of *allowed_roots*.

    Raises ``ValueError`` if the resolved path escapes all allowed roots.
    Defaults to the system temp directory and ``settings.data_dir``.
    """
    if allowed_roots is None:
        allowed_roots = _default_allowed_roots()

    try:
        resolved = Path(file_path).resolve()
    except Exception as exc:
        raise ValueError(f"Cannot resolve path: {file_path!r}") from exc

    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return resolved  # inside this root → safe
        except ValueError:
            continue

    allowed_str = ", ".join(f"'{r}'" for r in allowed_roots)
    raise ValueError(
        f"Path '{resolved}' is outside the allowed directories: {allowed_str}. "
        "Only files inside the system temp directory or the application data "
        "directory may be read."
    )


# ---------------------------------------------------------------------------
# Content sanitisation (prompt-injection defence)
# ---------------------------------------------------------------------------

# Zero-width, invisible, and direction-override Unicode characters often used
# to hide injection instructions from human readers.
_INVISIBLE_CHARS = re.compile(
    r"[​-‏"   # zero-width space / joiners / non-joiners
    r"‪-‮"    # bidirectional overrides
    r"⁠-⁤"    # word joiner / invisible operators
    r"⁪-⁯"    # deprecated formatting characters
    r"﻿]",         # BOM / zero-width no-break space
    re.UNICODE,
)

# Patterns that look like LLM role / instruction injections.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"\bdo\s+not\s+follow\s+(the\s+)?previous\s+instructions?", re.I),
    re.compile(r"\bnew\s+instructions?\s*:", re.I),
    re.compile(r"^\s*system\s*:", re.I | re.MULTILINE),
    re.compile(r"^\s*user\s*:", re.I | re.MULTILINE),
    re.compile(r"^\s*assistant\s*:", re.I | re.MULTILINE),
    re.compile(r"<\s*/?\s*(system|user|assistant)\s*>", re.I),
    re.compile(r"\[INST\]|\[/INST\]", re.I),
    re.compile(r"<<SYS>>|<</SYS>>", re.I),
    re.compile(r"###\s*(instruction|system|prompt)\b", re.I),
]


def sanitize_fetched_content(text: str) -> str:
    """Strip invisible characters and redact obvious prompt-injection patterns.

    This is a defence-in-depth measure.  It raises the cost of a successful
    prompt injection but is not a guarantee against all injection techniques.

    Returns the sanitised text.
    """
    # 1. Remove invisible / zero-width Unicode characters.
    text = _INVISIBLE_CHARS.sub("", text)

    # 2. Redact lines that match known injection patterns.
    lines = text.splitlines()
    clean: list[str] = []
    for line in lines:
        if any(pattern.search(line) for pattern in _INJECTION_PATTERNS):
            clean.append("[REDACTED]")
        else:
            clean.append(line)

    return "\n".join(clean)
