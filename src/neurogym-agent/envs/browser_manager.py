from __future__ import annotations

import os
import threading
import time

import psutil
from playwright.sync_api import sync_playwright


class BrowserManager:
    """Owns the single shared Chrome browser process.

    Workers must NOT share a Browser object — Playwright's sync API is thread-affine
    (greenlet-based). Instead each worker connects to the shared Chrome via the
    WebSocket endpoint exposed by BrowserServer, creating its own lightweight proxy.

    On crash: clears _ready, kills Chrome, relaunches, sets _ready.
    Workers call wait_ready() before reconnecting so they block transparently.
    """

    def __init__(self, headless: bool = True, extra_args: list[str] | None = None):
        self._headless = headless
        self._extra_args = extra_args or []
        self._ready = threading.Event()
        self._ws_endpoint: str | None = None
        self._pw = sync_playwright().start()
        self._server = None
        self._monitor = None  # connected browser used only for crash detection
        self._launch()

    # ------------------------------------------------------------------

    def _launch(self) -> None:
        self._server = self._pw.chromium.launch_server(
            headless=self._headless,
            args=self._extra_args,
        )
        self._ws_endpoint = self._server.ws_endpoint
        # Connect a lightweight monitor browser solely to receive "disconnected" events.
        self._monitor = self._pw.chromium.connect(self._ws_endpoint)
        self._monitor.on("disconnected", self._on_disconnect)
        self._ready.set()
        print(f"[BrowserManager] Chrome launched (ws={self._ws_endpoint})", flush=True)

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
    def ws_endpoint(self) -> str:
        """WebSocket endpoint of the current Chrome process. Does NOT block."""
        return self._ws_endpoint

    def wait_ready(self, timeout: float = 120.0) -> None:
        """Block until Chrome is alive. Call before using ws_endpoint."""
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
            self._server.close()
        except Exception:
            pass
        try:
            self._pw.stop()
        except Exception:
            pass
