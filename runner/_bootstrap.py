from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_parent_on_path() -> None:
    """Allow `python -m runner.run_experiment` from inside the package folder.

    The active layout treats this directory as the runnable project root.
    Adding the parent path also keeps older package-style entrypoints usable
    without requiring notebook-era imports in the implementation.
    """

    package_root = Path(__file__).resolve().parents[1]
    parent = package_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
