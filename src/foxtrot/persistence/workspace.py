"""Validate workspace locations before creating application files."""

from pathlib import Path


def resolve_workspace_root(
    workspace_root: str | Path,
    source_checkout: str | Path,
) -> Path:
    """Return an absolute workspace path separate from the checkout."""

    checkout = Path(source_checkout).expanduser()

    if not checkout.is_absolute():
        raise ValueError("Source checkout must be an absolute path")

    checkout = checkout.resolve(strict=True)

    if not checkout.is_dir():
        raise NotADirectoryError("Source checkout must be a directory")

    workspace = Path(workspace_root).expanduser()

    if not workspace.is_absolute():
        raise ValueError("Workspace root must be an absolute path")

    workspace = workspace.resolve()

    if (
        workspace == checkout
        or workspace.is_relative_to(checkout)
        or checkout.is_relative_to(workspace)
    ):
        raise ValueError(
            "Workspace and source checkout must not contain one another"
        )

    if workspace.exists() and not workspace.is_dir():
        raise NotADirectoryError("Workspace root must be a directory")

    return workspace


WORKSPACE_MARKER = ".foxtrot-workspace"
WORKSPACE_FORMAT = "foxtrot-workspace-v1\n"


def initialize_workspace(
    workspace_root: str | Path,
    source_checkout: str | Path,
) -> Path:
    """Initialize a new workspace or reopen a recognized one."""

    root = resolve_workspace_root(workspace_root, source_checkout)
    root.mkdir(parents=True, exist_ok=True)

    marker = root / WORKSPACE_MARKER

    if marker.is_symlink():
        raise ValueError("Workspace marker must not be a symbolic link")

    if marker.exists():
        if (
            not marker.is_file()
            or marker.read_text(encoding="utf-8") != WORKSPACE_FORMAT
        ):
            raise ValueError("Unrecognized workspace marker")
    else:
        if any(root.iterdir()):
            raise ValueError(
                "Refusing a nonempty folder without a Foxtrot marker"
            )

        with marker.open("x", encoding="utf-8") as stream:
            stream.write(WORKSPACE_FORMAT)

    managed = ("jobs", "settings", "models", "datasets", "tracking-outbox")
    for name in managed:
        directory = root / name
        if directory.is_symlink():
            raise ValueError(f"Workspace {name} folder must not be a symbolic link")
        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(f"Workspace {name} path must be a directory")
    for name in managed:
        (root / name).mkdir(exist_ok=True)

    return root
