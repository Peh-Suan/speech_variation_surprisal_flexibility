"""
Where a drive is mounted, versus what the dataset records.

Dataset csv files store absolute paths -- `/Volumes/Transcend/...`. Keeping
those fixed is what makes the files portable and comparable between runs. But
the drive is not always readable at that path: it can remount under a
different name, live somewhere else on another machine, or be mounted by a
sandbox under its own root.

SVSF_MOUNTS bridges the two without rewriting any data:

    export SVSF_MOUNTS="/Volumes/Transcend=/mnt/backup/Transcend"

Several mappings separate with the platform path separator. Everything that
opens a recording goes through `local_path`, so a dataset built on one machine
can be trained from on another with one environment variable and no edits.
"""

from __future__ import annotations

import os

MOUNTS_ENV = "SVSF_MOUNTS"


def aliases():
    out = []
    for part in os.environ.get(MOUNTS_ENV, "").split(os.pathsep):
        if "=" in part:
            canonical_root, actual_root = part.split("=", 1)
            out.append((canonical_root.rstrip("/"), actual_root.rstrip("/")))
    return out


def _swap(path, pairs):
    path = str(path)
    for src, dst in pairs:
        if path == src or path.startswith(src + "/"):
            return dst + path[len(src):]
    return path


def local_path(path):
    """Canonical path -> where it is actually readable right now."""
    pairs = aliases()
    return _swap(path, pairs) if pairs else str(path)


def canonical_path(path):
    """Where it is readable right now -> the canonical path to record."""
    pairs = aliases()
    return _swap(path, [(b, a) for a, b in pairs]) if pairs else str(path)
