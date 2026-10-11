"""
Shared argument handling for the stage runners.

Every stage names its work the same way: a unit ("phone" or "syllable"), one or
more corpora, and a window ("base", "prop25", ...). Putting that here means a
stage script is a thin wrapper over utils/, and that the same three words
appear in the filename, the log line and the manuscript.
"""

from __future__ import annotations

import argparse
import os
import sys

# Apple's MPS backend has no kernel for aten::grid_sampler_2d_backward, which
# the spatial transformer in VideoAutoencoder needs on every backward pass. The
# forward exists, so training starts and then dies at the first .backward().
# This sends only the missing ops to the CPU and leaves everything else on the
# GPU. It has to be set before torch is imported anywhere, which is why it sits
# here, above the utils imports, rather than in a function.
#
# Every successful video run so far was already relying on this -- it just came
# from an `export` in the terminal, so a fresh shell silently lost it. Setting
# it here makes the behaviour a property of the pipeline instead of a property
# of whichever window the command was typed into.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.corpora import CORPORA            # noqa: E402
from utils.paths import RunPaths             # noqa: E402


def parser(description, *, window=True, device=False):
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--unit", choices=("phone", "syllable"), required=True,
                   help="linguistic unit, which also picks the directory tree")
    p.add_argument("--corpus", nargs="+", default=list(CORPORA),
                   choices=list(CORPORA),
                   help="one or more corpora; defaults to all of them")
    if window:
        p.add_argument("--window", default="base",
                       help="token window: base, prop25, prop10, prop50, "
                            "frame1, fixed50 (see utils/windows.py)")
    if device:
        p.add_argument("--device", default=None,
                       help="torch device; default picks mps/cuda/cpu")
    p.add_argument("--root", default=REPO, help="repository root")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute even when the output already exists")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N rows, for a quick check")
    return p


def jobs(args):
    """Yields (spec, paths) for each requested corpus."""
    for name in args.corpus:
        spec = CORPORA[name]
        paths = RunPaths(args.unit, name, getattr(args, "window", "base"),
                         root=args.root).ensure_dirs()
        yield spec, paths


def announce(spec, paths, stage):
    print(f"\n=== {stage}: {spec.name} | {paths.unit} | "
          f"window={paths.window.name} ({paths.window.describe()}) ===")


def require(path, produced_by, args, corpus):
    """
    Stop with a readable message when an earlier stage has not run.

    Without this a missing input surfaces as a FileNotFoundError from inside
    numpy or pandas, which says where the read failed but not which stage was
    skipped -- and stages get skipped easily, since pasting several commands at
    once runs them all whether or not the first succeeded.
    """
    if os.path.exists(path):
        return
    window = getattr(args, "window", "base")
    cmd = (f"python runs/{produced_by} --unit {args.unit} --corpus {corpus}"
           + (f" --window {window}" if window != "base" else ""))
    raise SystemExit(
        f"\n  Missing: {os.path.relpath(path, args.root)}\n"
        f"  This stage reads it; it is produced by runs/{produced_by}\n"
        f"  Run this first:\n      {cmd}\n")


def skip_existing(path, overwrite):
    if os.path.exists(path) and not overwrite:
        print(f"  exists, skipping: {os.path.relpath(path)}  (--overwrite to redo)")
        return True
    return False
