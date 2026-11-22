"""Human names for datasets and runs, chosen by the member (e.g. "airfield noisy v2").

IDs and fingerprints stay the real identity (they never change); names are only labels
to recognise things later, so they can be set or changed at any time. Stored in
``settings/labels.json`` as {"datasets": {id: name}, "jobs": {id: name}}.
"""

import json
import os
from pathlib import Path
import threading
from uuid import uuid4

from .workspace import initialize_workspace

KINDS = ("datasets", "jobs")
MAX_LENGTH = 80
_lock = threading.Lock()


# Raised for an empty, too long or unprintable name.
class LabelError(ValueError):
    pass


# The labels file of a workspace.
def _path(workspace_root, source_checkout) -> Path:
    return initialize_workspace(workspace_root, source_checkout) / "settings" / "labels.json"


# All names: {"datasets": {...}, "jobs": {...}} (empty when none were set).
def read_labels(workspace_root, source_checkout) -> dict[str, dict[str, str]]:
    try:
        value = json.loads(_path(workspace_root, source_checkout).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        value = {}
    return {kind: dict(value.get(kind) or {}) for kind in KINDS}


# A cleaned name, or None to remove it.
def clean_name(name: str | None) -> str | None:
    if name is None or not name.strip():
        return None
    name = " ".join(name.split())
    if len(name) > MAX_LENGTH or not name.isprintable():
        raise LabelError(f"Names are up to {MAX_LENGTH} printable characters")
    return name


# Set (or with None/blank, remove) the name of one dataset or job; returns the stored name.
def set_label(workspace_root, source_checkout, kind: str, item_id: str, name: str | None) -> str | None:
    if kind not in KINDS:
        raise LabelError("Unknown label kind")
    name = clean_name(name)
    path = _path(workspace_root, source_checkout)
    with _lock:
        labels = read_labels(workspace_root, source_checkout)
        if name is None:
            labels[kind].pop(item_id, None)
        else:
            labels[kind][item_id] = name
        temporary = path.with_name(f".labels.{uuid4().hex}")
        temporary.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    return name
