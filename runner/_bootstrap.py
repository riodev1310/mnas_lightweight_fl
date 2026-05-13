from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_parent_on_path() -> None:
    """Allow `python -m runner.run_experiment` from inside the package folder.

    When the current working directory is `mnas_project/`, Python can import
    top-level siblings such as `runner`, but not the package name
    `mnas_project` itself. Adding the parent directory keeps the normal
    `mnas_project.*` imports working without changing the rest of the package.
    """

    package_root = Path(__file__).resolve().parents[1]
    parent = package_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
