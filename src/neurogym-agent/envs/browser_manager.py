from __future__ import annotations

import os

import psutil


class BrowserManager:
    """Holds Chrome launch configuration and monitors aggregate RSS.

    With Playwright Python's thread-affine sync API, each NGLGymEnv worker
    manages its own Chrome process. This class exists to:
      1. Pass headless/extra_args config to workers without duplicating it everywhere.
      2. Provide aggregate chrome_rss_mb() across all worker Chromes.
      3. Keep train.py / eval.py / rollout.py interfaces stable.
    """

    def __init__(self, headless: bool = True, extra_args: list[str] | None = None):
        self._headless = headless
        self._extra_args = extra_args or []

    @property
    def headless(self) -> bool:
        return self._headless

    @property
    def extra_args(self) -> list[str]:
        return self._extra_args

    def wait_ready(self, timeout: float = 120.0) -> None:
        """No-op — each worker launches and manages its own Chrome."""

    def chrome_rss_mb(self) -> float:
        """Total RSS in MB across all Chrome child processes (all workers combined)."""
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
        """No-op — workers close their own Chrome on env.close()."""
