"""Portable process-level resource measurements for CP05 experiments.

``PEAK_PROCESS_WORKING_SET`` is the operating-system-reported maximum resident
working set of the current process since process start.  It includes resident
pages owned by Python and native dependencies such as NumPy/SciPy, excludes
child processes, and is reported in bytes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
from ctypes import wintypes
import math
import sys
from typing import Any


MEMORY_METRIC = "PEAK_PROCESS_WORKING_SET"
MEMORY_UNIT = "bytes"


class MemoryProbeError(RuntimeError):
    """Raised when the native memory backend cannot return a valid metric."""

    def __init__(self, operation: str, *, error_code: int | None = None) -> None:
        self.operation = operation
        self.error_code = error_code
        detail = f"{operation} failed"
        if error_code is not None:
            formatter = getattr(ctypes, "FormatError", None)
            error_text = formatter(error_code).strip() if formatter is not None else "message unavailable"
            detail += f" with OS error {error_code}: {error_text}"
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class PeakProcessMemory:
    metric: str
    unit: str
    value: int
    value_mib: float
    backend: str
    scope: str
    includes_native_memory: bool
    includes_child_processes: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_bytes(value: int | float) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MemoryProbeError("normalizing peak process memory")
    if not math.isfinite(float(value)) or value <= 0 or int(value) != value:
        raise MemoryProbeError("normalizing peak process memory")
    return int(value)


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _WindowsMemoryApi:
    """Typed PSAPI binding.  GetCurrentProcess returns a borrowed pseudo-handle."""

    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self._get_current_process = kernel32.GetCurrentProcess
        self._get_current_process.argtypes = []
        self._get_current_process.restype = wintypes.HANDLE
        self._get_process_memory_info = psapi.GetProcessMemoryInfo
        self._get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        self._get_process_memory_info.restype = wintypes.BOOL

    def peak_working_set_bytes(self) -> int:
        handle = self._get_current_process()
        if not handle:
            error_code = ctypes.get_last_error()
            raise MemoryProbeError("GetCurrentProcess", error_code=error_code)
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.set_last_error(0)
        success = self._get_process_memory_info(
            handle, ctypes.byref(counters), counters.cb
        )
        if not success:
            raise MemoryProbeError(
                "GetProcessMemoryInfo", error_code=ctypes.get_last_error()
            )
        # The pseudo-handle returned by GetCurrentProcess is borrowed and must
        # not be passed to CloseHandle.  No owned process handle is created.
        return _positive_bytes(counters.PeakWorkingSetSize)


_WINDOWS_API: _WindowsMemoryApi | None = None


def _get_windows_api() -> _WindowsMemoryApi:
    global _WINDOWS_API
    if _WINDOWS_API is None:
        _WINDOWS_API = _WindowsMemoryApi()
    return _WINDOWS_API


def _posix_peak_rss_bytes(*, platform_name: str) -> int:
    try:
        import resource
    except ImportError as exc:
        raise MemoryProbeError("importing POSIX resource backend") from exc
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the BSDs report KiB.
    multiplier = 1 if platform_name == "darwin" else 1024
    return _positive_bytes(raw * multiplier)


def measure_peak_process_memory(*, platform_name: str | None = None) -> PeakProcessMemory:
    """Measure peak resident process memory and fail closed on backend errors."""

    selected_platform = sys.platform if platform_name is None else platform_name
    if selected_platform == "win32":
        value = _get_windows_api().peak_working_set_bytes()
        backend = "Windows PSAPI GetProcessMemoryInfo.PeakWorkingSetSize"
    elif selected_platform.startswith(("linux", "freebsd", "openbsd", "netbsd")) or selected_platform == "darwin":
        value = _posix_peak_rss_bytes(platform_name=selected_platform)
        backend = "POSIX getrusage(RUSAGE_SELF).ru_maxrss"
    else:
        raise MemoryProbeError(f"unsupported platform {selected_platform!r}")
    value = _positive_bytes(value)
    return PeakProcessMemory(
        metric=MEMORY_METRIC,
        unit=MEMORY_UNIT,
        value=value,
        value_mib=value / (1024**2),
        backend=backend,
        scope="current process since process start; child processes excluded",
        includes_native_memory=True,
        includes_child_processes=False,
    )
