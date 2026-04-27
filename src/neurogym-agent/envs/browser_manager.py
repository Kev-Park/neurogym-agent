from __future__ import annotations

import os
import threading
import time

import psutil
from playwright.sync_api import Browser, sync_playwright


class BrowserManager:
    """Owns the single shared Chrome browser process.

    Workers (threads) share the Browser object directly — no CDP connections needed.
    On GPU/browser crash Playwright fires 'disconnected'; BrowserManager clears _ready,
    kills any surviving Chrome children, relaunches, then sets _ready again.
    Workers call wait_ready() before using the browser, so they block transparently
    during restarts without any per-worker coordination logic.
    """

    def __init__(self, headless: bool = True, extra_args: list[str] | None = None):
        self._headless = headless
        self._extra_args = extra_args or []
        self._ready = threading.Event()
        self._browser: Browser | None = None
        self._pw = sync_playwright().start()
        self._launch()

    # ------------------------------------------------------------------

    def _launch(self) -> None:
        self._browser = self._pw.chromium.launch(
            headless=self._headless,
            args=self._extra_args,
        )
        self._browser.on("disconnected", self._on_disconnect)
        self._ready.set()

    def _on_disconnect(self) -> None:
        """Playwright fires this on the event-loop thread — defer blocking work."""
        self._ready.clear()
        threading.Thread(target=self._do_restart, daemon=True).start()

    def _do_restart(self) -> None:
        print("[BrowserManager] browser disconnected — killing Chrome and relaunching", flush=True)
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
    def browser(self) -> Browser:
        return self._browser

    def wait_ready(self, timeout: float = 120.0) -> None:
        """Block until the browser is alive. Safe to call from any worker thread."""
        self._ready.wait(timeout=timeout)

    def chrome_rss_mb(self) -> float:
        """Total RSS in MB of all Chrome child processes (all workers combined)."""
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
        """Call once from the main process after all worker threads have exited."""
        try:
            self._browser.close()
        except Exception:
            pass
        try:
            self._pw.stop()
        except Exception:
            pass
