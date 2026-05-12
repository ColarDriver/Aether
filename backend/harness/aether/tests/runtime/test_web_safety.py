"""Unit tests for ``aether.runtime.resources.web_safety``.

Sprint 3.5 / PR-2 (PR 3.5.5).
"""

from __future__ import annotations

import socket
import unittest
from unittest.mock import patch

from aether.runtime.resources.web_safety import (
    PREAPPROVED_HOSTS,
    is_preapproved,
    is_url_safe,
)


def _addrinfo(addr: str) -> list:
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (addr, 0))]


class IsUrlSafeTests(unittest.TestCase):
    def test_a1_loopback_localhost_rejected(self) -> None:
        with patch("socket.getaddrinfo", return_value=_addrinfo("127.0.0.1")):
            ok, reason = is_url_safe("http://localhost/")
        self.assertFalse(ok)
        self.assertIn("internal IP", reason)

    def test_a2_explicit_loopback_ip_rejected(self) -> None:
        ok, reason = is_url_safe("http://127.0.0.1/foo")
        self.assertFalse(ok)
        self.assertIn("internal IP", reason)

    def test_a3_private_192_rejected(self) -> None:
        with patch("socket.getaddrinfo", return_value=_addrinfo("192.168.1.5")):
            ok, reason = is_url_safe("http://internal.example.test/")
        self.assertFalse(ok)
        self.assertIn("192.168.1.5", reason)

    def test_a4_private_10_rejected(self) -> None:
        ok, _ = is_url_safe("http://10.0.0.1/")
        self.assertFalse(ok)

    def test_a5_link_local_aws_metadata_rejected(self) -> None:
        ok, reason = is_url_safe("http://169.254.169.254/latest/meta-data/")
        self.assertFalse(ok)
        self.assertIn("169.254.169.254", reason)

    def test_a6_file_scheme_rejected(self) -> None:
        ok, reason = is_url_safe("file:///etc/passwd")
        self.assertFalse(ok)
        self.assertIn("scheme", reason)

    def test_a6b_ftp_scheme_rejected(self) -> None:
        ok, _ = is_url_safe("ftp://example.com/")
        self.assertFalse(ok)

    def test_a6c_gopher_scheme_rejected(self) -> None:
        ok, _ = is_url_safe("gopher://example.com/")
        self.assertFalse(ok)

    def test_a7_public_https_accepted(self) -> None:
        with patch("socket.getaddrinfo", return_value=_addrinfo("140.82.114.4")):
            ok, reason = is_url_safe("https://github.com/python/cpython")
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "")

    def test_a8_unparseable_url_rejected(self) -> None:
        ok, _ = is_url_safe("not-a-url")
        self.assertFalse(ok)

    def test_a9_dns_rebind_resolving_to_loopback_rejected(self) -> None:
        # Public-looking host that DNS resolves to 127.0.0.1 — this is
        # exactly the rebind attack the SSRF guard exists to defeat.
        with patch("socket.getaddrinfo", return_value=_addrinfo("127.0.0.1")):
            ok, reason = is_url_safe("https://evil.example.com/")
        self.assertFalse(ok)
        self.assertIn("internal IP", reason)

    def test_a10_empty_string_rejected(self) -> None:
        ok, reason = is_url_safe("")
        self.assertFalse(ok)
        self.assertIn("empty", reason.lower())

    def test_a11_dns_failure_rejected(self) -> None:
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("nope")):
            ok, reason = is_url_safe("https://does-not-resolve.example.test/")
        self.assertFalse(ok)
        self.assertIn("DNS lookup failed", reason)

    def test_a12_no_hostname_rejected(self) -> None:
        ok, reason = is_url_safe("https:///foo")
        self.assertFalse(ok)
        self.assertIn("hostname", reason)

    def test_a13_unspecified_address_rejected(self) -> None:
        with patch("socket.getaddrinfo", return_value=_addrinfo("0.0.0.0")):
            ok, _ = is_url_safe("http://wildcard.example/")
        self.assertFalse(ok)

    def test_a14_multicast_rejected(self) -> None:
        with patch("socket.getaddrinfo", return_value=_addrinfo("224.0.0.1")):
            ok, reason = is_url_safe("http://multicast.example/")
        self.assertFalse(ok)
        self.assertIn("internal IP", reason)


class IsPreapprovedTests(unittest.TestCase):
    def test_a15_known_host_preapproved(self) -> None:
        for host in ("github.com", "docs.python.org", "stackoverflow.com"):
            with self.subTest(host=host):
                self.assertTrue(is_preapproved(f"https://{host}/anything"))

    def test_a16_unknown_host_not_preapproved(self) -> None:
        self.assertFalse(is_preapproved("https://example.org/"))

    def test_a17_preapproved_set_contains_expected_entries(self) -> None:
        for host in ("docs.python.org", "github.com", "pypi.org"):
            self.assertIn(host, PREAPPROVED_HOSTS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
