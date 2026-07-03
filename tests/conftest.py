"""Shared test setup.

Must run before ANY test module imports `ray` (ngllib_agent.policies pulls it
in transitively): under `uv run`, ray captures the uv-run runtime_env setting at
import time, and auto-shipping the 1.2GB repo CWD exceeds ray's 512MB cap.
"""

import os

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
