"""Hardware context collector for run_hardware_metrics.

Attempts nvidia-smi first, then rocm-smi. Falls back to platform / /proc
for CPU/RAM. All fields are best-effort — any exception sets that field to
None; collect_hardware_context() never raises.
"""

from __future__ import annotations

import contextlib
import platform
import re
import subprocess
import sys
from typing import Any


def _run(*args: str, timeout: int = 5) -> str:
    """Run a command and return stdout. Returns '' on any error."""
    try:
        result = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _nvidia_info() -> dict[str, Any]:
    """Query nvidia-smi for GPU model, VRAM, power limit, and driver version."""
    out = _run(
        "nvidia-smi",
        "--query-gpu=name,memory.total,power.limit,driver_version",
        "--format=csv,noheader,nounits",
    )
    if not out:
        return {}
    first_line = out.splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    result: dict[str, Any] = {}
    if len(parts) >= 1 and parts[0]:
        result["gpu_model"] = parts[0]
    if len(parts) >= 2:
        with contextlib.suppress(ValueError, TypeError):
            result["vram_mb"] = int(float(parts[1]))
    if len(parts) >= 3:
        with contextlib.suppress(ValueError, TypeError):
            result["power_limit_w"] = float(parts[2])
    if len(parts) >= 4 and parts[3]:
        result["driver_version"] = parts[3]
    return result


def _rocm_info() -> dict[str, Any]:
    """Query rocm-smi for GPU model, VRAM, and power limit."""
    result: dict[str, Any] = {}

    name_out = _run("rocm-smi", "--showproductname")
    if name_out:
        for line in name_out.splitlines():
            if "Card series" in line or "GPU" in line:
                m = re.search(r":\s*(.+)$", line)
                if m:
                    result["gpu_model"] = m.group(1).strip()
                    break

    mem_out = _run("rocm-smi", "--showmeminfo", "vram")
    if mem_out:
        for line in mem_out.splitlines():
            if "Total Memory" in line or "vram" in line.lower():
                m = re.search(r"(\d+)", line)
                if m:
                    with contextlib.suppress(ValueError, TypeError):
                        # rocm-smi reports in bytes; convert to MB
                        result["vram_mb"] = int(m.group(1)) // (1024 * 1024)
                    break

    power_out = _run("rocm-smi", "--showpower")
    if power_out:
        for line in power_out.splitlines():
            if "Power Cap" in line or "power" in line.lower():
                m = re.search(r"([\d.]+)", line)
                if m:
                    with contextlib.suppress(ValueError, TypeError):
                        result["power_limit_w"] = float(m.group(1))
                    break

    return result


def _cpu_info() -> dict[str, Any]:
    """Read CPU model + core/thread counts from /proc/cpuinfo (Linux) or platform fallback."""
    result: dict[str, Any] = {}
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                content = f.read()
            model_match = re.search(r"^model name\s*:\s*(.+)$", content, re.MULTILINE)
            if model_match:
                result["cpu_model"] = model_match.group(1).strip()
            physical_ids = set(re.findall(r"^physical id\s*:\s*(\d+)", content, re.MULTILINE))
            core_ids = re.findall(r"^cpu cores\s*:\s*(\d+)", content, re.MULTILINE)
            processor_ids = re.findall(r"^processor\s*:\s*(\d+)", content, re.MULTILINE)
            if core_ids:
                try:
                    cores_per_socket = int(core_ids[0])
                    n_sockets = max(len(physical_ids), 1)
                    result["cpu_cores"] = cores_per_socket * n_sockets
                except (ValueError, TypeError):
                    pass
            if processor_ids:
                result["cpu_threads"] = len(processor_ids)
        except Exception:
            pass

    if "cpu_model" not in result:
        proc = platform.processor()
        if proc:
            result["cpu_model"] = proc

    return result


def _ram_gb() -> float | None:
    """Return total RAM in GB via psutil; None if psutil is unavailable."""
    try:
        import psutil  # type: ignore[import]

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        return None


def _os_info() -> dict[str, Any]:
    """Return os_name and os_version."""
    result: dict[str, Any] = {}
    try:
        result["os_name"] = platform.system()
        result["os_version"] = platform.version()
    except Exception:
        pass
    return result


def collect_hardware_context() -> dict[str, Any]:
    """Collect hardware/OS context for run_hardware_metrics.

    Returns a dict with keys:
        gpu_model, vram_mb, power_limit_w,
        cpu_model, cpu_cores, ram_gb,
        os_name, os_version, driver_version

    All fields are best-effort — missing/failed fields are None.
    Never raises.
    """
    ctx: dict[str, Any] = {
        "gpu_model": None,
        "vram_mb": None,
        "power_limit_w": None,
        "cpu_model": None,
        "cpu_cores": None,
        "ram_gb": None,
        "os_name": None,
        "os_version": None,
        "driver_version": None,
    }

    try:
        gpu = _nvidia_info()
        if not gpu:
            gpu = _rocm_info()
        ctx.update({k: v for k, v in gpu.items() if v is not None})
    except Exception:
        pass

    try:
        cpu = _cpu_info()
        ctx.update({k: v for k, v in cpu.items() if v is not None})
    except Exception:
        pass

    try:
        ram = _ram_gb()
        if ram is not None:
            ctx["ram_gb"] = ram
    except Exception:
        pass

    try:
        os_data = _os_info()
        ctx.update({k: v for k, v in os_data.items() if v is not None})
    except Exception:
        pass

    return ctx
