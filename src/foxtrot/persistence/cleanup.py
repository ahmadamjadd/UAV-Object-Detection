"""Safe storage cleanup (V8).

Only clearly disposable files are offered, and the plan is recomputed right
before deleting, so nothing that became important in the meantime is removed:

* finished generation/preview/DVC/prediction run folders (not the running one);
* hidden leftovers of interrupted builds older than six hours (``.generate-*``,
  ``.import-*``, ``.dvc-*``, ``.letters-*``, ``.validation-*``);
* intermediate training files of jobs that **finished and are fully uploaded**
  (registered artifacts such as best.pt, data.yaml and results are kept, so
  every job stays verifiable). Interrupted/failed/cancelled jobs keep last.pt
  for resuming; active, held or not-yet-uploaded jobs are never touched.

Datasets are never deleted here: jobs and studies refer to them by identity.
"""

from pathlib import Path
import shutil
import time

from .database import connect_database
from .jobs import list_job_records
from .workspace import initialize_workspace

STALE_SECONDS = 3600            # finished-run folders
STAGING_SECONDS = 6 * 3600      # hidden build folders: long builds must never look abandoned
HIDDEN_PREFIXES = (".generate-", ".import-", ".dvc-", ".letters-", ".validation-")


# Total bytes under a folder (files only, no symbolic links followed).
def folder_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


# Disk use per workspace area, for the Setup page.
def storage_usage(workspace_root, source_checkout) -> dict[str, int]:
    root = initialize_workspace(workspace_root, source_checkout)
    return {area: folder_size(root / area) for area in
            ("jobs", "datasets", "studies", "generation", "models", "tracking-outbox") if (root / area).is_dir()}


# Paths (relative to the job folder) registered as a job's artifacts; these are always kept.
def _registered(root: Path, checkout, job_id: str) -> set[str]:
    with connect_database(root, checkout) as connection:
        rows = connection.execute("SELECT relative_path FROM artifacts WHERE job_id = ?", (job_id,)).fetchall()
    return {row[0] for row in rows}


# Every cleanup candidate with its size and why it is safe; `active_run` is the running generation run.
def plan_cleanup(workspace_root, source_checkout, *, active_run: str | None = None) -> list[dict]:
    root = initialize_workspace(workspace_root, source_checkout)
    now = time.time()
    found = []
    runs = root / "generation" / "runs"
    for folder in sorted(runs.iterdir()) if runs.is_dir() else ():
        status = folder / "status.json"
        if folder.name == active_run or folder.is_symlink() or not status.is_file():
            continue
        if '"state": "finished"' in (text := status.read_text()) or '"state": "failed"' in text \
                or '"state": "cancelled"' in text or now - status.stat().st_mtime > STALE_SECONDS:
            found.append({"id": f"run:{folder.name}", "path": str(folder.relative_to(root)),
                          "bytes": folder_size(folder), "reason": "finished background run (previews, logs)"})
    for parent in (root / "datasets", root / "generation" / "letters", *sorted((root / "studies").glob("*"))):
        for folder in sorted(parent.glob(".*")) if parent.is_dir() else ():
            if folder.name.startswith(HIDDEN_PREFIXES) and folder.is_dir() and not folder.is_symlink() \
                    and now - folder.stat().st_mtime > STAGING_SECONDS:
                found.append({"id": f"staging:{folder.relative_to(root)}", "path": str(folder.relative_to(root)),
                              "bytes": folder_size(folder), "reason": "left over from an interrupted build"})
    for record in list_job_records(root, source_checkout, limit=500):  # the newest 500 jobs
        if (record.execution_state != "finished" or record.tracking_state != "synced" or record.recovery_required):
            continue
        directory = root / "jobs" / record.job_id
        keep = _registered(root, source_checkout, record.job_id)
        extras = [p for p in (directory / "output").rglob("*")
                  if p.is_file() and not p.is_symlink() and p.relative_to(directory).as_posix() not in keep]
        size = sum(p.stat().st_size for p in extras)
        if extras:
            found.append({"id": f"extras:{record.job_id}", "path": f"jobs/{record.job_id}/output",
                          "bytes": size, "reason": f"{len(extras)} intermediate files of a finished, uploaded job "
                                                   "(best.pt and results kept)"})
    return found


# Delete the chosen candidates, but only those still offered by a fresh plan.
def apply_cleanup(workspace_root, source_checkout, ids: list[str], *, active_run: str | None = None) -> dict:
    root = initialize_workspace(workspace_root, source_checkout)
    current = {item["id"]: item for item in plan_cleanup(root, source_checkout, active_run=active_run)}
    removed, freed, skipped = [], 0, []
    for item_id in ids:
        item = current.get(item_id)
        if item is None:
            skipped.append(item_id)
            continue
        if item_id.startswith("extras:"):
            job_id = item_id.split(":", 1)[1]
            directory = root / "jobs" / job_id
            keep = _registered(root, source_checkout, job_id)
            for path in sorted((directory / "output").rglob("*"), reverse=True):
                if path.is_file() and not path.is_symlink() and path.relative_to(directory).as_posix() not in keep:
                    path.unlink()
                elif path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
        else:
            shutil.rmtree(root / item["path"])
        removed.append(item_id)
        freed += item["bytes"]
    return {"removed": removed, "skipped": skipped, "freed_bytes": freed}
