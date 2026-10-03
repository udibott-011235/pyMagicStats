"""DEC-029 operational toy oracle. No scientific modules, samples or host repair."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import uuid

REQUIRED_GATES = (
    "CUDA_DEVICE_COUNT", "CUDA_RUNTIME_CALLABLE", "LIBNVRTC_SO_13_LOAD", "NVRTC_VERSION",
    "FLOAT64_ARRAY", "FLOAT64_ELEMENTWISE", "FLOAT64_REDUCTION",
    "RAWKERNEL_BACKEND", "RAWKERNEL_FRESH_COMPILE", "RAWKERNEL_LAUNCH",
    "RAWKERNEL_NUMERICAL_RESULT", "CUPYX_SCIPY_GAMMALN", "CUPYX_SCIPY_DIGAMMA",
    "CUPYX_SCIPY_POLYGAMMA", "CUPYX_SCIPY_GAMMAINC", "CUPYX_SCIPY_GAMMAINCC",
    "CUPYX_SCIPY_BETAINC", "CUPY_SORT", "CUPY_NEXTAFTER",
)
_INVOCATIONS = set()


class ReadinessError(ValueError):
    """Fail closed; no repair, retry, fallback or authorization consumption."""
    consumed = False


class ReadinessFailed(ReadinessError):
    def __init__(self, evidence):
        super().__init__("CUDA execution-readiness oracle failed")
        self.evidence = evidence
        self.consumed = False


def _require(condition, message):
    if not condition:
        raise ReadinessError(message)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def validate_target(output, repository, scientific_output=None):
    """Validate before any GPU operation or parent-directory creation."""
    _require(output is not None, "--readiness-output mandatory")
    original = Path(output)
    target = original.resolve()
    repository = Path(repository).resolve()
    _require(not target.is_relative_to(repository), "readiness evidence must be outside repository")
    if scientific_output is not None:
        scientific = Path(scientific_output).resolve()
        _require(not target.is_relative_to(scientific),
                 "readiness evidence must be separate from scientific output")
    _require(not original.is_symlink() and not target.exists(),
             "readiness output exists; overwrite/resume prohibited")
    return target


def _initial(identity):
    for name in ("R11_HARNESS_SHA", "R11_HARNESS_TREE"):
        value = identity.get(name)
        _require(type(value) is str and len(value) == 40
                 and all(c in "0123456789abcdef" for c in value), "invalid harness identity")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "platform": platform.platform(),
        "python": platform.python_version(), "cupy": None,
        "cuda_driver_version": None, "cuda_runtime_version": None,
        "device_count": None, "device_identity": None,
        "effective_CUDA_PATH": os.environ.get("CUDA_PATH"),
        "effective_LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        "LIBNVRTC_SO_13_LOAD": "NOT_RUN", "NVRTC_VERSION": None,
        "RAWKERNEL_BACKEND": None, "fresh_jit_proof": None,
        "gates": {name: "NOT_RUN" for name in REQUIRED_GATES},
        "toy_results": {}, "CUDA_READINESS_ORACLE_PASS": False,
        "R11_HARNESS_SHA": identity["R11_HARNESS_SHA"],
        "R11_HARNESS_TREE": identity["R11_HARNESS_TREE"],
        "SCIENTIFIC_EXECUTION_STARTED": False, "SCIENTIFIC_RECORDS_EVALUATED": 0,
        "EXECUTION_AUTHORIZATION_CONSUMED": False,
        "SCIENTIFIC_OUTPUT_DIRECTORY_CREATED": False,
        "REAL_R11_SAMPLES_EVALUATED": False,
        "CURAND_REQUIRED_BY_CURRENT_R11": False,
        "CUSOLVER_REQUIRED_BY_CURRENT_R11": False, "failure": None,
        "AUTO_REPAIR": False, "AUTO_RERUN": False, "AUTO_RESUME": False,
    }


def _fresh_source():
    # A distinct symbol changes the actual compiled source, not only a comment.
    nonce = uuid.uuid4().hex
    _require(nonce not in _INVOCATIONS, "fresh-JIT invocation identity reused")
    _INVOCATIONS.add(nonce)
    name = "r11_readiness_" + nonce
    source = ('extern "C" __global__ void ' + name +
              '(const double* x, double* y) { int i = blockDim.x * blockIdx.x + threadIdx.x; '
              'if (i < 3) y[i] = x[i] + 4.0; }')
    return name, source, {"invocation_id": nonce, "kernel_name": name,
        "source_sha256": _digest(source.encode("utf-8")), "backend": "nvrtc",
        "cache_strategy": "unique-per-invocation-source-and-symbol",
        "compile_called": False, "compile_synchronized": False}


def _probe(evidence):
    """Only this lazy entry point calls real infrastructure in a future authorized invocation."""
    evidence["_stage"] = "cupy_import"
    cp = importlib.import_module("cupy")
    evidence["cupy"] = cp.__version__
    api = cp.cuda.runtime

    def gate(name, action, check=lambda value: True, *, synchronize=False):
        evidence["_stage"] = name
        value = action()
        if synchronize:
            api.deviceSynchronize()
        _require(check(value), name + " failed")
        evidence["gates"][name] = "PASS"
        return value

    count = gate("CUDA_DEVICE_COUNT", api.getDeviceCount,
                 lambda value: type(value) is int and value >= 1)
    evidence["device_count"] = count

    def runtime_context():
        evidence["cuda_driver_version"] = api.driverGetVersion()
        evidence["cuda_runtime_version"] = api.runtimeGetVersion()
        index = api.getDevice()
        properties = api.getDeviceProperties(index)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        evidence["device_identity"] = {"index": index, "name": str(name)}
        api.deviceSynchronize()
        return all(type(evidence[k]) is int and evidence[k] > 0
                   for k in ("cuda_driver_version", "cuda_runtime_version"))

    gate("CUDA_RUNTIME_CALLABLE", runtime_context, lambda value: value is True)
    library = gate("LIBNVRTC_SO_13_LOAD", lambda: ctypes.CDLL("libnvrtc.so.13"))
    evidence["LIBNVRTC_SO_13_LOAD"] = "PASS"

    def version():
        major, minor = ctypes.c_int(), ctypes.c_int()
        function = library.nvrtcVersion
        function.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        function.restype = ctypes.c_int
        _require(function(ctypes.byref(major), ctypes.byref(minor)) == 0,
                 "nvrtcVersion failed")
        loaded = (major.value, minor.value)
        evidence["NVRTC_VERSION"] = list(loaded)
        return loaded, tuple(cp.cuda.nvrtc.getVersion())

    gate("NVRTC_VERSION", version, lambda value: value == ((13, 0), (13, 0)))

    def numerical(name, producer, expected, *, approximate=False):
        def operation():
            value = producer()
            api.deviceSynchronize()
            host = cp.asnumpy(value)
            api.deviceSynchronize()
            _require(str(host.dtype) == "float64", name + " dtype mismatch")
            observed = [float(v) for v in host.reshape(-1).tolist()]
            _require(len(observed) == len(expected) and all(math.isfinite(v) for v in observed),
                     name + " invalid numerical result")
            _require(all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)
                         if approximate else a == b for a, b in zip(observed, expected)),
                     name + " numerical mismatch")
            evidence["toy_results"][name] = {"observed": observed, "expected": expected,
                                            "synchronized": True}
            return value
        return gate(name, operation)

    x = numerical("FLOAT64_ARRAY", lambda: cp.asarray([1., 2., 3.], dtype=cp.float64),
                  [1., 2., 3.])
    numerical("FLOAT64_ELEMENTWISE", lambda: x * 2. + 1., [3., 5., 7.])
    numerical("FLOAT64_REDUCTION", lambda: cp.sum(x, dtype=cp.float64), [6.])
    evidence["_stage"] = "RAWKERNEL_FRESH_COMPILE"
    name, source, proof = _fresh_source()
    evidence["fresh_jit_proof"] = proof
    kernel = gate("RAWKERNEL_BACKEND",
                  lambda: cp.RawKernel(source, name, backend="nvrtc"),
                  lambda value: value.backend == "nvrtc"
                  and value.code == source and value.name == name)
    evidence["RAWKERNEL_BACKEND"] = "NVRTC"

    def compile_fresh():
        kernel.compile()  # Mandatory; no cached-result-only shortcut.
        proof["compile_called"] = True
    gate("RAWKERNEL_FRESH_COMPILE", compile_fresh, synchronize=True)
    proof["compile_synchronized"] = True

    def launch():
        y = cp.empty_like(x)
        kernel((1,), (32,), (x, y))
        return y
    y = gate("RAWKERNEL_LAUNCH", launch, synchronize=True)
    numerical("RAWKERNEL_NUMERICAL_RESULT", lambda: y, [5., 6., 7.])

    evidence["_stage"] = "cupyx_special_import"
    special = importlib.import_module("cupyx.scipy.special")
    tests = (
        ("GAMMALN", lambda: special.gammaln(cp.asarray([3.], dtype=cp.float64)), [math.log(2.)]),
        ("DIGAMMA", lambda: special.digamma(cp.asarray([2.], dtype=cp.float64)),
         [1. - 0.5772156649015329]),
        ("POLYGAMMA", lambda: special.polygamma(1, cp.asarray([2.], dtype=cp.float64)),
         [math.pi ** 2 / 6. - 1.]),
        ("GAMMAINC", lambda: special.gammainc(1., cp.asarray([1.], dtype=cp.float64)),
         [1. - math.exp(-1.)]),
        ("GAMMAINCC", lambda: special.gammaincc(1., cp.asarray([1.], dtype=cp.float64)),
         [math.exp(-1.)]),
        ("BETAINC", lambda: special.betainc(1., 1., cp.asarray([.25], dtype=cp.float64)), [.25]),
    )
    for name, producer, expected in tests:
        numerical("CUPYX_SCIPY_" + name, producer, expected, approximate=True)
    numerical("CUPY_SORT", lambda: cp.sort(cp.asarray([3., 1., 2.], dtype=cp.float64)),
              [1., 2., 3.])
    numerical("CUPY_NEXTAFTER",
              lambda: cp.nextafter(cp.asarray([1.], dtype=cp.float64),
                                   cp.asarray([2.], dtype=cp.float64)),
              [math.nextafter(1., 2.)])


def _complete(evidence):
    gates = evidence.get("gates", {})
    proof = evidence.get("fresh_jit_proof") or {}
    nonce, digest = proof.get("invocation_id"), proof.get("source_sha256")
    return (set(gates) == set(REQUIRED_GATES) and all(gates[n] == "PASS" for n in REQUIRED_GATES)
            and evidence.get("NVRTC_VERSION") == [13, 0]
            and evidence.get("LIBNVRTC_SO_13_LOAD") == "PASS"
            and evidence.get("RAWKERNEL_BACKEND") == "NVRTC"
            and proof.get("backend") == "nvrtc" and proof.get("compile_called") is True
            and proof.get("compile_synchronized") is True
            and type(digest) is str and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest)
            and type(nonce) is str and len(nonce) == 32
            and all(c in "0123456789abcdef" for c in nonce)
            and proof.get("kernel_name") == "r11_readiness_" + nonce
            and proof.get("cache_strategy") == "unique-per-invocation-source-and-symbol"
            and evidence.get("failure") is None)


@dataclass(frozen=True)
class ReadinessReceipt:
    path: Path
    data: bytes
    sha256: str

    def evidence(self):
        _require(_digest(self.data) == self.sha256, "readiness receipt digest mismatch")
        return json.loads(self.data.decode("utf-8"))


def run_readiness(output, repository, identity, scientific_output=None):
    target = validate_target(output, repository, scientific_output)
    evidence = _initial(identity)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Reserve exclusively before touching a device, then persist either outcome.
    with target.open("xb") as stream:
        try:
            _probe(evidence)
            _require(_complete(evidence), "incomplete mandatory readiness surface")
            evidence["CUDA_READINESS_ORACLE_PASS"] = True
        except BaseException as exc:
            stage = evidence.get("_stage", "readiness")
            if stage in evidence["gates"]:
                evidence["gates"][stage] = "FAIL"
            if stage == "LIBNVRTC_SO_13_LOAD":
                evidence["LIBNVRTC_SO_13_LOAD"] = "FAIL"
            evidence["failure"] = {"stage": stage, "error_type": type(exc).__name__,
                                   "message": str(exc)}
        evidence.pop("_stage", None)
        data = _encode(evidence)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return ReadinessReceipt(target, data, _digest(data))


def require_pass(receipt):
    evidence = receipt.evidence()
    _require(receipt.path.read_bytes() == receipt.data, "persisted readiness evidence changed")
    if evidence.get("CUDA_READINESS_ORACLE_PASS") is not True or not _complete(evidence):
        raise ReadinessFailed(evidence)
    return evidence
