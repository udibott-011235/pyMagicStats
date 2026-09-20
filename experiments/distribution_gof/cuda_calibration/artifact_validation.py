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

def validate_bootstrap_attempts(attempts):
    grouped={}
    for row in attempts: grouped.setdefault((row["cell_id"],row["raw_outer_index"]),[]).append(row)
    for rows in grouped.values():
        raw=[x["raw_inner_index"] for x in rows]; eligible=[x["eligible_index_or_null"] for x in rows if x.get("eligible_index_or_null") is not None]
        if len(raw)!=len(set(raw)) or len(eligible)!=15 or sorted(eligible)!=list(range(15)): raise ArtifactValidationError("bootstrap accounting")
    return True
def generate_digests(directory):
    values={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.name!="digests.json"}
    if set(values)!=set(REQUIRED_ARTIFACTS)-{"digests.json"}: raise ArtifactValidationError("digest generation set")
    (directory/"digests.json").write_text(json.dumps(values,sort_keys=True),encoding="utf-8"); return values
