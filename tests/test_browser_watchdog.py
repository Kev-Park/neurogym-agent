from __future__ import annotations

import time

import pytest

ngllib_env = pytest.importorskip("ngllib.environment")
_BrowserWatchdog = ngllib_env._BrowserWatchdog


def test_fires_after_timeout():
    fired = []
    wd = _BrowserWatchdog(0.1, lambda: fired.append(1))
    time.sleep(0.3)
    assert wd.fired is True and fired == [1]
    wd.cancel()  # cancel after fire is a no-op


def test_cancel_prevents_fire():
    fired = []
    wd = _BrowserWatchdog(0.3, lambda: fired.append(1))
    wd.cancel()
    time.sleep(0.5)
    assert wd.fired is False and fired == []


def test_disabled_when_timeout_none():
    wd = _BrowserWatchdog(None, lambda: (_ for _ in ()).throw(AssertionError))
    time.sleep(0.1)
    assert wd.fired is False
    wd.cancel()


def test_env_watchdog_noop_without_chrome_pid():
    # Environment._watchdog must disable itself when no browser pid is known
    # (pre-launch, or pid discovery failed) — otherwise the timer would "kill"
    # nothing and still flag restarts.
    from ngllib import Environment

    env = Environment.__new__(Environment)  # no browser, no __init__ side effects
    env._chrome_pid = None
    wd = env._watchdog(0.05)
    time.sleep(0.2)
    assert wd.fired is False
