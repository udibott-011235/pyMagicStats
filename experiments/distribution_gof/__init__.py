"""Private CP05 research harness; not a production API.

Imports stay lazy so standalone experimental submodules can be inspected on a
CUDA host before the optional CPU/SciPy reference stack is installed.
"""

__all__ = ["AssessmentStatus", "ExperimentManifest", "ReasonCode", "run_manifest"]


def __getattr__(name):
    if name in {"AssessmentStatus", "ReasonCode"}:
        from .accounting import AssessmentStatus, ReasonCode

        return {"AssessmentStatus": AssessmentStatus, "ReasonCode": ReasonCode}[name]
    if name == "ExperimentManifest":
        from .manifest import ExperimentManifest

        return ExperimentManifest
    if name == "run_manifest":
        from .runner import run_manifest

        return run_manifest
    raise AttributeError(name)
