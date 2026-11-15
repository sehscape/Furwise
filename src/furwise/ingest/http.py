"""A small, polite HTTP client with a disk cache.

Two promises to the sites we read from:
  * every URL is downloaded once and then served from data/raw/;
  * requests to a host are spaced out by at least its robots.txt Crawl-delay
    (ASPCA asks for 10 s), and disallowed paths are never fetched.
"""

from __future__ import annotations

import logging
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests

from furwise.settings import MIN_INTERVAL, USER_AGENT

log = logging.getLogger(__name__)


class PoliteClient:
    def __init__(self, user_agent: str = USER_AGENT, timeout: float = 30.0, retries: int = 4):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.timeout = timeout
        self.retries = retries
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last_hit: dict[str, float] = {}

    def _rules(self, host: str, scheme: str) -> urllib.robotparser.RobotFileParser:
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                resp = self.session.get(f"{scheme}://{host}/robots.txt", timeout=self.timeout)
                rp.parse(resp.text.splitlines() if resp.ok else [])
            except requests.RequestException:
                rp.parse([])
            self._robots[host] = rp
        return self._robots[host]

    def _wait_turn(self, host: str, rules: urllib.robotparser.RobotFileParser) -> None:
        delay = max(MIN_INTERVAL.get(host, 1.0), float(rules.crawl_delay("*") or 0))
        elapsed = time.monotonic() - self._last_hit.get(host, 0.0)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_hit[host] = time.monotonic()

    def get(self, url: str, params: dict | None = None) -> bytes:
        parts = urlparse(url)
        rules = self._rules(parts.netloc, parts.scheme)
        if not rules.can_fetch(self.session.headers["User-Agent"], url):
            raise PermissionError(f"robots.txt disallows {url}")
        for attempt in range(self.retries + 1):
            self._wait_turn(parts.netloc, rules)
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == self.retries:
                    raise
                log.warning("%s failed (%s), retrying", url, exc)
                time.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                return resp.content
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                time.sleep(float(resp.headers.get("Retry-After") or 2 ** (attempt + 2)))
                continue
            resp.raise_for_status()
        raise RuntimeError(f"giving up on {url}")

    def fetch(self, url: str, dest: Path, params: dict | None = None) -> Path:
        """Download to `dest` unless it is already cached there."""
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        content = self.get(url, params)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(content)
        tmp.replace(dest)
        return dest
