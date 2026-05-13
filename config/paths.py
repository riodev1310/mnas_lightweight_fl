from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parent


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path from cwd, MNAS project root, or repository root."""

    p = Path(path)
    if p.is_absolute():
        return p
    candidates = [Path.cwd() / p, PACKAGE_ROOT / p, PROJECT_ROOT / p, REPO_ROOT / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / p).resolve()
