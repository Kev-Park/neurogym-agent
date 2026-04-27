from __future__ import annotations

import os
import socket
import threading
import time
import urllib.request

import psutil
from playwright.sync_api import sync_playwright


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_cdp(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/json/version", timeout=1)
            return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError(f"Chrome CDP not ready on port {port} after {timeout}s")


class BrowserManager:
    """Owns the single shared Chrome browser process.

    Playwright Python's sync API is thread-affine (greenlet-based) so a Browser
    object cannot be shared across threads. Instead Chrome is launched with
    --remote-debugging-port and each worker connects via connect_over_cdp() with
    its own sync_playwright() instance — same Chrome process, separate proxies.

    On crash: clears _ready, kills Chrome, relaunches, sets _ready.
    Workers call wait_ready() before reconnecting so they block transparently.
    """

    def __init__(self, headless: bool = True, extra_args: list[str] | None = None):
        self._headless = headless
        self._extra_args = extra_args or []
        self._ready = threading.Event()
        self._cdp_url: str | None = None
        self._pw = None
        self._browser = None   # launched browser (lives in its own thread)
        self._monitor = None   # connected proxy used only for crash detection
        self._launch()

    # ------------------------------------------------------------------

    def _launch(self) -> None:
        # Always create a fresh playwright in the calling thread.
        if self._pw is not None:
            try:
                self._pw.stop()
            except Exception:
                pass
        self._pw = sync_playwright().start()
        port = _find_free_port()
        self._cdp_url = f"http://localhost:{port}"
        self._browser = self._pw.chromium.launch(
            headless=self._headless,
            args=self._extra_args + [f"--remote-debugging-port={port}"],
        )
        _wait_for_cdp(port)
        self._monitor = self._pw.chromium.connect_over_cdp(self._cdp_url)
        self._monitor.on("disconnected", self._on_disconnect)
        self._ready.set()
        print(f"[BrowserManager] Chrome launched at {self._cdp_url}", flush=True)

    def _on_disconnect(self) -> None:
        """Playwright fires this on the event-loop thread — defer blocking work."""
        print("[BrowserManager] browser disconnected — killing Chrome and relaunching", flush=True)
        self._ready.clear()
        threading.Thread(target=self._do_restart, daemon=True).start()

    def _do_restart(self) -> None:
        self._kill_all_chrome()
        time.sleep(3)
        self._launch()
        print("[BrowserManager] browser relaunched", flush=True)

    def _kill_all_chrome(self) -> None:
        try:
            for proc in psutil.Process(os.getpid()).children(recursive=True):
                try:
                    if "chrome" in proc.name().lower():
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass

    # ------------------------------------------------------------------

    @property
    def cdp_url(self) -> str:
        """CDP endpoint URL of the current Chrome process. Does NOT block."""
        return self._cdp_url

    def wait_ready(self, timeout: float = 120.0) -> None:
        """Block until Chrome is alive. Call before using cdp_url."""
        self._ready.wait(timeout=timeout)

    def chrome_rss_mb(self) -> float:
        """Total RSS in MB of all Chrome child processes."""
        try:
            total = 0
            for child in psutil.Process(os.getpid()).children(recursive=True):
                try:
                    if "chrome" in child.name().lower():
                        total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total / (1024 * 1024)
        except Exception:
            return 0.0

    def close(self) -> None:
        """Call once from the main thread after all worker threads have exited."""
        try:
            self._monitor.close()
        except Exception:
            pass
        try:
            self._browser.close()
        except Exception:
            pass
        try:
            self._pw.stop()
        except Exception:
            pass
