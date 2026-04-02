#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_paths() -> None:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]
    package_root = repo_root / "src" / "onemap_semantic_mapper"
    ovo_root = repo_root / "reference" / "OVO"

    for path in [package_root, ovo_root]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def main() -> None:
    _bootstrap_paths()
    from onemap_semantic_mapper.observer_online import main as observer_main

    observer_main()


if __name__ == "__main__":
    main()
