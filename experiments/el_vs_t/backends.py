"""Optional hybrid CPU/GPU sample-generation backend.

Only random generation and descriptive moments are eligible for GPU work.
Student t and the validated scalar-root empirical-likelihood routines remain
on the CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import platform
from typing import Any, Sequence

import numpy as np

from experiments.adversarial_robustness_calibration import Scenario
from .seeds import SeedMaterial, numpy_rng


AUTO_GPU_MIN_ELEMENTS = 250_000


@dataclass(frozen=True)
class BackendInfo:
    requested: str
    resolved: str
    gpu_available: bool
    gpu_reason: str | None
    python_version: str
    numpy_version: str
    cupy_version: str | None
    gpu_device: str | None
    auto_gpu_min_elements: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NativeBatch:
    values: Any
    engine: str


def _load_cupy() -> tuple[Any | None, str | None, str | None]:
    try:
        cupy = importlib.import_module("cupy")
        count = int(cupy.cuda.runtime.getDeviceCount())
        if count < 1:
            return None, "CuPy reports no CUDA device", None
        properties = cupy.cuda.runtime.getDeviceProperties(0)
        raw_name = properties.get("name", b"unknown")
        device = raw_name.decode(errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
        return cupy, None, device
    except (ImportError, ModuleNotFoundError) as error:
        return None, f"CuPy unavailable: {error}", None
    except Exception as error:  # CUDA runtime errors vary by CuPy release.
        return None, f"CUDA unavailable: {type(error).__name__}: {error}", None


class SampleBackend:
    """Generate bounded batches and calculate backend-neutral diagnostics."""

    def __init__(self, requested: str) -> None:
        normalized = requested.casefold()
        if normalized not in {"auto", "cpu", "gpu"}:
            raise ValueError("backend must be one of: auto, cpu, gpu")
        self.requested = normalized
        self._cupy, gpu_reason, gpu_device = _load_cupy()
        if normalized == "gpu" and self._cupy is None:
            raise RuntimeError(gpu_reason or "GPU backend is unavailable")
        if normalized == "cpu":
            resolved = "cpu"
        elif normalized == "gpu":
            resolved = "gpu-hybrid"
        elif self._cupy is None:
            resolved = "cpu-fallback"
        else:
            resolved = "auto-hybrid"
        self.info = BackendInfo(
            requested=normalized,
            resolved=resolved,
            gpu_available=self._cupy is not None,
            gpu_reason=gpu_reason,
            python_version=platform.python_version(),
            numpy_version=np.__version__,
            cupy_version=getattr(self._cupy, "__version__", None),
            gpu_device=gpu_device,
            auto_gpu_min_elements=AUTO_GPU_MIN_ELEMENTS,
        )

    def generate_native(
        self,
        scenario: Scenario,
        n: int,
        seeds: Sequence[SeedMaterial],
    ) -> NativeBatch:
        use_gpu = self.requested == "gpu" or (
            self.requested == "auto"
            and self._cupy is not None
            and int(n) * len(seeds) >= AUTO_GPU_MIN_ELEMENTS
        )
        if use_gpu:
            return NativeBatch(self._generate_gpu(scenario, n, seeds), "gpu")
        return NativeBatch(self._generate_cpu(scenario, n, seeds), "cpu")

    @staticmethod
    def _generate_cpu(
        scenario: Scenario,
        n: int,
        seeds: Sequence[SeedMaterial],
    ) -> np.ndarray:
        rows = [scenario.draw(numpy_rng(seed), int(n)) for seed in seeds]
        samples = np.stack(rows, axis=0)
        if samples.shape != (len(seeds), int(n)) or not np.all(np.isfinite(samples)):
            raise FloatingPointError("CPU generator produced invalid simulation inputs")
        return samples

    def _generate_gpu(
        self,
        scenario: Scenario,
        n: int,
        seeds: Sequence[SeedMaterial],
    ) -> Any:
        cp = self._cupy
        if cp is None:  # Defensive; explicit GPU was checked in __init__.
            raise RuntimeError("GPU backend is unavailable")
        rows = cp.empty((len(seeds), int(n)), dtype=cp.float64)
        for index, seed in enumerate(seeds):
            rng = cp.random.default_rng(int(seed.uint64))
            rows[index] = _draw_gpu(scenario, rng, int(n), cp)
        if not bool(cp.all(cp.isfinite(rows)).item()):
            raise FloatingPointError("GPU generator produced invalid simulation inputs")
        return rows

    def diagnostics(self, batch: NativeBatch) -> np.ndarray:
        xp = np if batch.engine == "cpu" else self._cupy
        if xp is None:
            raise RuntimeError("GPU diagnostics requested without CuPy")
        values = batch.values
        count = int(values.shape[1])
        means = xp.mean(values, axis=1)
        centered = values - means[:, None]
        m2 = xp.mean(centered**2, axis=1)
        variance = xp.sum(centered**2, axis=1) / (count - 1)
        with np.errstate(divide="ignore", invalid="ignore") if xp is np else _nullcontext():
            g1 = xp.mean(centered**3, axis=1) / xp.power(m2, 1.5)
            g2 = xp.mean(centered**4, axis=1) / (m2**2) - 3.0
        if count > 2:
            skewness = xp.sqrt(count * (count - 1)) / (count - 2) * g1
        else:
            skewness = xp.full_like(means, xp.nan)
        if count > 3:
            kurtosis = (count - 1) / ((count - 2) * (count - 3)) * (
                (count + 1) * g2 + 6.0
            )
        else:
            kurtosis = xp.full_like(means, xp.nan)
        diagnostics = xp.stack((means, variance, skewness, kurtosis), axis=1)
        if batch.engine == "gpu":
            return np.asarray(self._cupy.asnumpy(diagnostics), dtype=float)
        return np.asarray(diagnostics, dtype=float)

    def to_cpu(self, batch: NativeBatch) -> np.ndarray:
        if batch.engine == "gpu":
            if self._cupy is None:
                raise RuntimeError("GPU transfer requested without CuPy")
            return np.asarray(self._cupy.asnumpy(batch.values), dtype=float)
        return np.asarray(batch.values, dtype=float)


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


def _draw_gpu(scenario: Scenario, rng: Any, n: int, cp: Any) -> Any:
    """GPU equivalent of the canonical generator, parameterized by its metadata."""

    family = scenario.family
    parameters = scenario.parameters
    if family == "normal":
        return _gpu_normal(
            rng,
            float(parameters["mean"]),
            float(parameters["sd"]),
            n,
        )
    if family == "student_t":
        df = float(parameters["df"])
        numerator = rng.standard_normal(size=n)
        chi_squared = 2.0 * rng.gamma(df / 2.0, 1.0, size=n)
        student_t = numerator / cp.sqrt(chi_squared / df)
        return student_t * np.sqrt((df - 2.0) / df)
    if family == "lognormal":
        sigma = float(parameters["sigma"])
        mean = np.exp(sigma * sigma / 2.0)
        variance = (np.exp(sigma * sigma) - 1.0) * np.exp(sigma * sigma)
        raw = cp.exp(sigma * rng.standard_normal(size=n))
        return (raw - mean) / np.sqrt(variance)
    if family == "gamma":
        shape = float(parameters["shape"])
        scale = float(parameters["scale"])
        raw = rng.gamma(shape, scale, size=n)
        return (raw - shape * scale) / (np.sqrt(shape) * scale)
    if family in {
        "bimodal",
        "normal_mixture",
        "normal_contamination_symmetric",
        "normal_contamination_asymmetric",
    }:
        if "weights" in parameters:
            probability = float(parameters["weights"][1])
            base_mean, component_mean = (float(value) for value in parameters["means"])
            base_sd, component_sd = (float(value) for value in parameters["sds"])
        else:
            probability = float(parameters["epsilon"])
            base_mean, base_sd = _parse_normal(parameters["base"])
            component_mean, component_sd = _parse_normal(parameters["contaminant"])
        indicator = rng.random(n) < probability
        base = _gpu_normal(rng, base_mean, base_sd, n)
        component = _gpu_normal(rng, component_mean, component_sd, n)
        mixture_mean = (1.0 - probability) * base_mean + probability * component_mean
        second = (1.0 - probability) * (base_sd**2 + base_mean**2) + probability * (
            component_sd**2 + component_mean**2
        )
        mixture_sd = np.sqrt(second - mixture_mean**2)
        return (cp.where(indicator, component, base) - mixture_mean) / mixture_sd
    raise ValueError(f"no GPU generator for canonical family {family!r}")


def _gpu_normal(rng: Any, mean: float, sd: float, size: int) -> Any:
    """Draw a normal array using the CuPy 13.6 Generator primitive API."""

    return mean + sd * rng.standard_normal(size=size)


def _parse_normal(value: object) -> tuple[float, float]:
    match = __import__("re").fullmatch(
        r"N\((-?[0-9.]+),([0-9.]+)\)", str(value).replace(" ", "")
    )
    if match is None:
        raise ValueError(f"unrecognized canonical normal specification: {value!r}")
    return float(match.group(1)), float(match.group(2))


def resolve_backend(requested: str) -> SampleBackend:
    return SampleBackend(requested)
