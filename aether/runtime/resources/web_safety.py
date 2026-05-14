"""Web tool safety helpers.

Shared by ``WebFetchTool``, ``WebSearchTool``, and
``WebBrowserTool``.

Provides:
* :func:`is_url_safe` — SSRF guard.  Rejects loopback, private IPs,
  link-local, multicast and unsupported schemes (``file:`` / ``ftp:`` /
  ``gopher:``).  Resolves DNS so a public-looking host that points back
  to ``127.0.0.1`` is also rejected.
* :func:`is_preapproved` — fast-path for high-confidence read-only
  documentation hosts (docs.python.org, github.com, ...).  Reserved for
  a future per-domain permission prompter.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


__all__ = ["PREAPPROVED_HOSTS", "is_url_safe", "is_preapproved"]


PREAPPROVED_HOSTS: frozenset[str] = frozenset(
    {
        "docs.python.org",
        "docs.rs",
        "doc.rust-lang.org",
        "developer.mozilla.org",
        "github.com",
        "raw.githubusercontent.com",
        "stackoverflow.com",
        "pypi.org",
        "npmjs.com",
        "registry.npmjs.org",
        "crates.io",
        "go.dev",
        "pkg.go.dev",
    }
)


_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def is_url_safe(url: str) -> tuple[bool, str]:
    """Return ``(safe, reason)``. ``reason`` is empty when ``safe``.

    The function runs three gates:

    1. URL parse + scheme check (``http`` / ``https`` only).
    2. Host present.
    3. DNS resolution + per-address rejection of loopback, private,
       link-local and multicast IPs.

    DNS failures are treated as unsafe — there is nothing useful we can
    do without resolving, and silently allowing them would let the
    network layer raise much later with a far less specific error.
    """

    if not isinstance(url, str) or not url.strip():
        return False, "URL is empty"

    try:
        parsed = urlparse(url.strip())
    except ValueError as exc:
        return False, f"invalid URL: {exc}"

    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        return False, f"unsupported scheme: {scheme!r} (only http/https allowed)"

    host = parsed.hostname
    if not host:
        return False, "URL has no hostname"

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        return False, f"DNS lookup failed: {exc}"
    except OSError as exc:
        return False, f"DNS lookup failed: {exc}"

    if not infos:
        return False, f"DNS lookup returned no addresses for {host!r}"

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_unspecified
            or ip.is_reserved
        ):
            return False, f"refusing to access internal IP: {addr}"

    return True, ""


def is_preapproved(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in PREAPPROVED_HOSTS
