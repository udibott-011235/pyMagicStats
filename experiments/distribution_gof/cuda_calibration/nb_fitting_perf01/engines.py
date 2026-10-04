"""FIT adapters and synchronous accounting; imports never initialize CUDA."""
from __future__ import annotations

from contextlib import AbstractContextManager
import threading
import time
from pathlib import Path

import numpy as np

from ..r11_reference_workload.codec import require
from .fast_cpu import failed


def canonical_cpu():
    """Load the unmodified production solver, including canonical pair checks."""
    from pyMagicStat.distributions.families import (
        NegativeBinomialFamily, FitIdentifiabilityError, NoFiniteMLEError,
        FitNumericalError, _fitting,
    )
    require(Path(_fitting.__file__).resolve() == Path(__file__).resolve().parents[4] /
            "pyMagicStat/distributions/families/_fitting.py", "foreign canonical fitting module")
    family = NegativeBinomialFamily()

    def fit(sample):
        label = "FAILED"
        try:
            result = _fitting.fit_negative_binomial(family, sample)
            label = "ELIGIBLE"
            require(result.metadata["estimator_id"] ==
                    "pymagicstats-negative-binomial-profile-mle-v1" and
                    result.metadata["solver_id"] == "scipy.optimize.brentq",
                    "noncanonical CPU estimator/solver")
            params = result.fitted_distribution.parameters
            return {"classification": label, "converged": bool(result.converged),
                    "parameters": {"r": float(params.r), "p": float(params.p)},
                    "log_likelihood": float(result.log_likelihood), "failure_reason": None,
                    "metadata": dict(result.metadata)}
        except Exception as exc:
            # Derive A's classification solely from the production solver's
            # result/error contract, independently of B's classifier.
            if isinstance(exc, FitIdentifiabilityError):
                label = "ALL_ZERO_NON_IDENTIFYING"
            elif isinstance(exc, NoFiniteMLEError):
                label = "VARIANCE_NOT_GREATER_THAN_MEAN"
            elif isinstance(exc, FitNumericalError):
                label = "ELIGIBLE"
            return failed(type(exc).__name__ + ": " + str(exc), label)
    return fit


class RSSPeak(AbstractContextManager):
    """Per-region RSS sampled every 2 ms; includes baseline, not lifetime HWM."""
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = 0
        self.stop = threading.Event()

    def sample(self):
        self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.sample()
        def monitor():
            while not self.stop.wait(0.002):
                self.sample()
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()


def measure(operation, *, synchronize=None, resources=None,
            wall_clock=time.perf_counter, cpu_clock=time.process_time):
    """One host region; warm-up/gates excluded. Synchronize even on failure."""
    resources = resources if resources is not None else RSSPeak()
    value, error = None, None
    with resources:
        if synchronize is not None:
            synchronize()
        wall_start, cpu_start = wall_clock(), cpu_clock()
        try:
            value = operation()
        except Exception as exc:
            error = exc
        finally:
            if synchronize is not None:
                try:
                    synchronize()
                except Exception as exc:
                    error = RuntimeError((str(error) + "; " if error else "") +
                                         "final CUDA synchronization failed: " + str(exc))
            cpu_end, wall_end = cpu_clock(), wall_clock()
    return value, {"wall_seconds": wall_end - wall_start,
                   "cpu_process_seconds": cpu_end - cpu_start,
                   "peak_rss_bytes": resources.peak,
                   "peak_rss_method": "per-region RSS samples every 2ms including baseline"}, error


def compatible_batches(records, limit):
    """Contiguous compatible runs only. No sort, padding, truncation or retries."""
    batch = []
    for record in records:
        key = (record.payload.dtype_str, record.payload.shape)
        if batch and (key != (batch[0].payload.dtype_str, batch[0].payload.shape)
                      or (isinstance(limit, int) and len(batch) == limit)):
            yield tuple(batch)
            batch = []
        batch.append(record)
    if batch:
        yield tuple(batch)


class CudaEngine:
    """Only this explicitly constructed adapter imports/requests CUDA."""
    def __init__(self):
        from .. import cuda_candidate
        self.candidate = cuda_candidate
        self.cp = cuda_candidate.require_cuda()
        require(Path(cuda_candidate.__file__).resolve() ==
                Path(__file__).resolve().parents[1] / "cuda_candidate.py",
                "foreign CUDA candidate module")
        self.memory_peak = self.cp.get_default_memory_pool().total_bytes()

    def synchronize(self):
        self.cp.cuda.runtime.deviceSynchronize()

    def reset_memory_accounting(self):
        self.synchronize()
        self.cp.get_default_memory_pool().free_all_blocks()
        self.memory_peak = self.cp.get_default_memory_pool().total_bytes()

    def _stage(self, operation):
        self.synchronize()
        start, end = self.cp.cuda.Event(), self.cp.cuda.Event()
        wall_start = time.perf_counter()
        start.record()
        try:
            result = operation()
        finally:
            end.record()
            self.synchronize()
        wall = time.perf_counter() - wall_start
        device = float(self.cp.cuda.get_elapsed_time(start, end)) / 1000
        self.memory_peak = max(self.memory_peak, self.cp.get_default_memory_pool().total_bytes())
        return result, wall, device

    def fit_batch(self, records):
        # Stacking and float64 conversion are part of total host wall and H2D
        # wall. Raw read-only payloads are never changed or reconstructed by RNG.
        device, h2d, event_h2d = self._stage(lambda: self.cp.asarray(
            np.stack([r.sample for r in records]), dtype=self.cp.float64))
        raw, compute, event_compute = self._stage(
            lambda: self.candidate.fit_negative_binomial(device))
        def download():
            return {key: self.cp.asnumpy(value) if isinstance(value, self.cp.ndarray) else value
                    for key, value in raw.items()}
        host, d2h, event_d2h = self._stage(download)
        results = []
        for index in range(len(records)):
            converged = bool(np.asarray(host["converged"]).reshape(-1)[index])
            label = host["classification"][index]
            checks = {key: bool(np.asarray(host[key]).reshape(-1)[index]) for key in
                      ("bracket_found", "root_inside_bracket", "root_sign_check",
                       "root_precision_check", "root_residual_check", "objective_valid")}
            results.append({"classification": label, "converged": converged,
                            "parameters": {key: float(np.asarray(host[key]).reshape(-1)[index])
                                           for key in ("r", "p")},
                            "log_likelihood": float(np.asarray(host["log_likelihood"]).reshape(-1)[index]),
                            "checks": checks, "iterations": host["iterations"],
                            "failure_reason": None if converged else
                                "CUDA NB validation/classification failed: " +
                                ",".join(key for key, ok in checks.items() if not ok)})
        return results, {"host_to_device": h2d, "device_compute": compute,
                         "device_to_host": d2h,
                         "gpu_device_seconds": event_h2d + event_compute + event_d2h,
                         "gpu_memory_peak_bytes": self.memory_peak}

    def environment(self):
        runtime = self.cp.cuda.runtime
        device = self.cp.cuda.Device()
        properties = runtime.getDeviceProperties(device.id)
        name = properties["name"]
        return {"cupy": self.cp.__version__, "device_id": device.id,
                "device_name": name.decode() if isinstance(name, bytes) else str(name),
                "cuda_runtime_version": runtime.runtimeGetVersion(),
                "cuda_driver_version": runtime.driverGetVersion(),
                "device_memory_bytes": int(properties["totalGlobalMem"])}
