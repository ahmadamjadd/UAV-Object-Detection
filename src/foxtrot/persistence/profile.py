"""This installation's member profile (V8): who is training, for shared MLflow.

Stored in ``settings/profile.json``. The member name is added to every run as the
``foxtrot.member`` tag, so results from several team members' installations can be
told apart in one shared MLflow experiment. Nothing secret is stored here.
"""

import json
from pathlib import Path
import re

from .workspace import initialize_workspace

MEMBER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")


# The saved profile ({"member": None} until set).
def read_profile(workspace_root, source_checkout) -> dict:
    path = initialize_workspace(workspace_root, source_checkout) / "settings" / "profile.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return {"member": value.get("member")}
    except (OSError, ValueError):
        return {"member": None}


# Save the member name (letters, digits, space, dot, dash, underscore; up to 64 characters).
def save_profile(workspace_root, source_checkout, member: str) -> dict:
    member = member.strip()
    if not MEMBER.fullmatch(member):
        raise ValueError("Member name: 1-64 letters, digits, spaces, dots, dashes or underscores")
    path = initialize_workspace(workspace_root, source_checkout) / "settings" / "profile.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"member": member}, indent=2), encoding="utf-8")
    temporary.replace(path)
    return {"member": member}
