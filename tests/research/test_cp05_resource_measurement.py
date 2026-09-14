from __future__ import annotations

import ctypes
from ctypes import wintypes
import math
import sys

import pytest

from experiments.distribution_gof import resource_measurement
from experiments.distribution_gof.resource_measurement import (
    MEMORY_METRIC,
    MEMORY_UNIT,
    MemoryProbeError,
    _positive_bytes,
    measure_peak_process_memory,
)


def test_peak_process_memory_is_finite_positive_integer_bytes():
    measurement = measure_peak_process_memory()

    assert measurement.metric == MEMORY_METRIC == "PEAK_PROCESS_WORKING_SET"
    assert measurement.unit == MEMORY_UNIT == "bytes"
    assert type(measurement.value) is int
    assert measurement.value > 0
    assert math.isfinite(measurement.value_mib)
    assert measurement.value_mib == measurement.value / (1024**2)
    assert measurement.includes_native_memory is True
    assert measurement.includes_child_processes is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows handle accounting")
def test_repeated_windows_measurement_does_not_leak_process_handles():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = wintypes.HANDLE
    get_process_handle_count = kernel32.GetProcessHandleCount
    get_process_handle_count.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    get_process_handle_count.restype = wintypes.BOOL

    def handle_count() -> int:
        count = wintypes.DWORD()
        ctypes.set_last_error(0)
        if not get_process_handle_count(get_current_process(), ctypes.byref(count)):
            raise OSError(ctypes.get_last_error())
        return int(count.value)

    measure_peak_process_memory()
    before = handle_count()
    for _ in range(100):
        measure_peak_process_memory()
    after = handle_count()

    assert after == before


def test_native_backend_failure_is_explicit_not_zero_or_null(monkeypatch):
    class FailingWindowsApi:
        def peak_working_set_bytes(self):
            raise MemoryProbeError("GetProcessMemoryInfo", error_code=6)

    monkeypatch.setattr(resource_measurement, "_get_windows_api", lambda: FailingWindowsApi())
    with pytest.raises(MemoryProbeError, match="OS error 6"):
        measure_peak_process_memory(platform_name="win32")


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), None, True, 1.5])
def test_invalid_native_values_fail_closed(value):
    with pytest.raises(MemoryProbeError):
        _positive_bytes(value)


def test_platform_specific_path_is_isolated(monkeypatch):
    class WindowsApi:
        def peak_working_set_bytes(self):
            return 4096

    monkeypatch.setattr(resource_measurement, "_get_windows_api", lambda: WindowsApi())
    measurement = measure_peak_process_memory(platform_name="win32")

    assert measurement.value == 4096
    assert measurement.backend == "Windows PSAPI GetProcessMemoryInfo.PeakWorkingSetSize"
