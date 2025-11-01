"""System inspection utilities for ollamabench."""

from __future__ import annotations

import platform
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import psutil

_DEFAULT_TIMEOUT = 5
_SYSTEM_PROFILER_LIMIT = 2048


def _run_command(cmd: list[str], limit: Optional[int] = None) -> Optional[str]:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=_DEFAULT_TIMEOUT,
        )
    except FileNotFoundError:
        return None
    except subprocess.SubprocessError:
        return None

    if result.returncode != 0:
        return None

    output = (result.stdout or result.stderr).strip()
    if not output:
        return None
    if limit is not None:
        return output[:limit]
    return output


def _collect_nvidia_info() -> Optional[str]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    return _run_command(cmd)


def _collect_system_profiler() -> Optional[str]:
    if platform.system() != "Darwin":
        return None
    return _run_command(["system_profiler", "SPDisplaysDataType"], limit=_SYSTEM_PROFILER_LIMIT)


def _collect_tegrastats() -> Optional[str]:
    return _run_command(["tegrastats", "--interval", "1000", "--count", "1"])


def collect_system_info() -> Dict[str, Any]:
    """Gather portable system metadata for benchmark provenance."""
    virtual_mem = psutil.virtual_memory()

    info: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gib": round(virtual_mem.total / (1024**3), 2),
    }

    profiler = _collect_system_profiler()
    if profiler:
        info["system_profiler"] = profiler

    nvidia = _collect_nvidia_info()
    if nvidia:
        info["nvidia_smi"] = nvidia

    tegra = _collect_tegrastats()
    if tegra:
        info["tegrastats"] = tegra

    powermetrics_path = shutil.which("powermetrics")
    if powermetrics_path:
        info["powermetrics_hint"] = (
            f"powermetrics detected at {powermetrics_path}. "
            "Consider running `sudo powermetrics -f ascii -i 1000` in parallel for power data."
        )

    return info


__all__ = ["collect_system_info"]
