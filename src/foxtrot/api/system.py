"""Setup check (V8): is this installation and workspace ready to train?

Reads only. Used by the Setup page and ``foxtrot --workspace W --check``. Package
availability is checked without importing the heavy libraries.
"""

import importlib.metadata
import importlib.util
from pathlib import Path
import re
import shutil
import sys

MIN_FREE_BYTES = 5 * 1024 ** 3  # generated datasets, checkpoints and caches need room
HEAVY_PACKAGES = {"torch": "torch", "ultralytics": "ultralytics", "mlflow": "mlflow", "optuna": "optuna",
                  "dvc": "dvc", "cv2": "opencv-python"}


# One check result: name, passed or not, and a plain explanation.
def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


# The pinned versions foxtrot-local declares, e.g. {"torch": "2.14.1"}.
def _declared_pins() -> dict[str, str]:
    pins = {}
    try:
        for requirement in importlib.metadata.requires("foxtrot-local") or []:
            match = re.match(r"^([A-Za-z0-9_.\-]+)==([^;\s]+)", requirement)
            if match:
                pins[match.group(1).lower()] = match.group(2)
    except importlib.metadata.PackageNotFoundError:
        pass
    return pins


# Run every check for one workspace; never changes anything.
def system_check(workspace: Path, checkout: Path, static_directory: Path | None = None) -> dict:
    from foxtrot.persistence.database import DATABASE_SCHEMA_VERSION
    from foxtrot.persistence.models import list_weights
    from foxtrot.pipeline.contracts import PipelineFailure
    from foxtrot.pipeline.foxtrot_generator import generator_files

    checks = [_check("Python", sys.version_info[:2] == (3, 14),
                     f"{sys.version.split()[0]} (supported: 3.14)")]
    missing = [name for name in HEAVY_PACKAGES if importlib.util.find_spec(name) is None]
    checks.append(_check("Libraries", not missing, "all present" if not missing else f"missing: {', '.join(missing)}"))
    pins, drift = _declared_pins(), []
    for distribution, pinned in sorted(pins.items()):
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        if installed != pinned:
            drift.append(f"{distribution} {installed} (expected {pinned})")
    checks.append(_check("Pinned versions", not drift,
                         f"{len(pins)} pins match" if not drift else "; ".join(drift[:5])))
    try:
        generator_files()
        checks.append(_check("Dataset generator", True, "found next to the package (source checkout)"))
    except PipelineFailure as failure:
        checks.append(_check("Dataset generator", False, failure.message))
    root = Path(workspace).resolve()
    free = shutil.disk_usage(root if root.exists() else root.parent).free
    checks.append(_check("Free disk space", free >= MIN_FREE_BYTES, f"{free / 1024 ** 3:.1f} GB free (want ≥ 5 GB)"))
    checks.append(_check("Workspace database", True, f"schema version {DATABASE_SCHEMA_VERSION}"))
    weights = [w["model_id"] for w in list_weights(workspace, checkout) if w["status"] == "installed"]
    checks.append(_check("Pretrained weights", bool(weights), ", ".join(weights) if weights
                         else "none installed (only needed for pretrained starts)"))
    if static_directory is not None:
        checks.append(_check("Browser interface", (static_directory / "index.html").is_file(),
                             "built into the package" if (static_directory / "index.html").is_file()
                             else "missing: run npm run build in frontend/"))
    from .gpu import detect_gpus
    gpus = detect_gpus()
    checks.append(_check("GPU", True, gpus["message"]))  # informational: CPU training always works
    required = {"Python", "Libraries", "Workspace database"}
    return {"ok": all(c["ok"] for c in checks if c["name"] in required), "checks": checks}
