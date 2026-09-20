"""Read-back, digest and atomic publication validation for C2C artifacts."""
from __future__ import annotations
import hashlib,json,shutil
from pathlib import Path
from .artifact_writers import FIT_COLUMNS,STAT_COLUMNS,CLASS_COLUMNS
from .equivalence_preregistration import REQUIRED_ARTIFACTS
class ArtifactValidationError(RuntimeError): pass
def validate_parquet(path, columns):
    import pandas as pd
    frame=pd.read_parquet(path)
    if tuple(frame.columns)!=tuple(columns): raise ArtifactValidationError("parquet schema violation")
    if "identity" in frame and frame["identity"].duplicated().any(): raise ArtifactValidationError("duplicate identity")
    return len(frame)
def validate_bundle(directory):
    names={p.name for p in directory.iterdir()}
    if names!=set(REQUIRED_ARTIFACTS): raise ArtifactValidationError("required artifact set mismatch")
    validate_parquet(directory/"fit_comparison.parquet",FIT_COLUMNS); validate_parquet(directory/"statistic_comparison.parquet",STAT_COLUMNS); validate_parquet(directory/"classification_comparison.parquet",CLASS_COLUMNS)
    summary=json.loads((directory/"summary.json").read_text())
    if summary.get("calibration_claim") is not False: raise ArtifactValidationError("calibration claim")
    digests=json.loads((directory/"digests.json").read_text())
    if set(digests)!=set(REQUIRED_ARTIFACTS)-{"digests.json"}: raise ArtifactValidationError("digest set mismatch")
    for name,digest in digests.items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest: raise ArtifactValidationError("digest mismatch")
    return True
def publish_atomic(temp:Path, output:Path):
    if output.exists(): raise ArtifactValidationError("preexisting output protected")
    validate_bundle(temp); temp.replace(output)
