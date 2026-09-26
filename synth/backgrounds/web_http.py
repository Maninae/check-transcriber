"""Polite keyless HTTP for the web-background fetcher (stdlib `urllib` only).

- Every request names the project in its User-Agent (plus `CHECK_SYNTH_FETCH_CONTACT` if set).
- Per-host minimum spacing between requests (Openverse allows ~20 anonymous requests a minute).
- 429 and 5xx are retried with exponential backoff, honouring `Retry-After`.
- `HttpRangeFile` is a seekable read-only file over HTTP Range requests, so `zipfile` can pull a
  single member out of a large remote zip without downloading the archive.
"""

import io
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

# Wikimedia asks for contact details in the User-Agent; they come from the environment, never the repo.
FETCH_CONTACT = os.environ.get("CHECK_SYNTH_FETCH_CONTACT", "")
USER_AGENT = ("check-transcriber-synth/0.1 (background-texture fetcher for synthetic training data; "
              f"CC0/public-domain only{'; contact ' + FETCH_CONTACT if FETCH_CONTACT else ''})")
DEFAULT_MIN_SECONDS_BETWEEN_REQUESTS = 0.5
HOST_MIN_SECONDS_BETWEEN_REQUESTS = {
    "api.openverse.org": 3.5,        # 20/min anonymous burst limit
    "commons.wikimedia.org": 1.0,
    "upload.wikimedia.org": 1.0,
}
MAX_ATTEMPTS = 6
FIRST_BACKOFF_SECONDS = 5.0
REQUEST_TIMEOUT_SECONDS = 60
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class PoliteHttpClient:
    """urllib wrapper with a project User-Agent, per-host spacing and retry/backoff."""

    def __init__(self):
        self.last_request_time_by_host: dict[str, float] = {}

    def wait_for_host(self, url: str) -> None:
        """Sleep until this host's minimum spacing since our last request has passed."""
        host = urllib.parse.urlparse(url).hostname or ""
        spacing = HOST_MIN_SECONDS_BETWEEN_REQUESTS.get(host, DEFAULT_MIN_SECONDS_BETWEEN_REQUESTS)
        elapsed = time.monotonic() - self.last_request_time_by_host.get(host, 0.0)
        if elapsed < spacing:
            time.sleep(spacing - elapsed)
        self.last_request_time_by_host[host] = time.monotonic()

    def open(self, url: str, extra_headers: dict[str, str] | None = None):
        """Open `url` and return the response; retries 429/5xx and connection errors, raises on anything else."""
        backoff_seconds = FIRST_BACKOFF_SECONDS
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.wait_for_host(url)
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(extra_headers or {})})
            try:
                return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS)
            except urllib.error.HTTPError as http_error:
                if http_error.code not in RETRYABLE_STATUS_CODES or attempt == MAX_ATTEMPTS:
                    raise
                retry_after = http_error.headers.get("Retry-After", "")
                wait_seconds = float(retry_after) if retry_after.isdigit() else backoff_seconds
                logger.warning("HTTP %d from %s; retrying in %.0f s", http_error.code, url, wait_seconds)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as network_error:
                if attempt == MAX_ATTEMPTS:
                    raise
                wait_seconds = backoff_seconds
                logger.warning("network error %s on %s; retrying in %.0f s", network_error, url, wait_seconds)
            time.sleep(wait_seconds)
            backoff_seconds *= 2
        raise RuntimeError("unreachable")

    def get_bytes(self, url: str, extra_headers: dict[str, str] | None = None) -> bytes:
        """Body of a GET."""
        with self.open(url, extra_headers) as response:
            return response.read()

    def get_json(self, url: str, query: dict[str, str | int] | None = None) -> dict | list:
        """Parsed JSON of a GET with an optional query string."""
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        return json.loads(self.get_bytes(url))


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, fetching only the byte ranges actually read.

    Wrap in `io.BufferedReader` so zipfile's many small reads coalesce into few requests.
    """

    def __init__(self, client: PoliteHttpClient, url: str):
        super().__init__()
        self.client = client
        with client.open(url, {"Range": "bytes=0-0"}) as response:
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or "/" not in content_range:
                raise ValueError(f"server ignored Range for {url} (status {response.status})")
            self.url = response.geturl()  # follow the redirect once; later reads hit the CDN directly
            self.size = int(content_range.rsplit("/", 1)[1])
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def readinto(self, buffer) -> int:
        if self.position >= self.size or len(buffer) == 0:
            return 0
        last_byte = min(self.size, self.position + len(buffer)) - 1
        data = self.client.get_bytes(self.url, {"Range": f"bytes={self.position}-{last_byte}"})
        buffer[:len(data)] = data
        self.position += len(data)
        return len(data)
