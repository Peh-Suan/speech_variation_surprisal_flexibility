#!/usr/bin/env python3
"""
Stage 2: bigram surprisal, positional entropy and unit frequency.

Lexical only -- reads the processed dataset and the pronunciation counts,
touches no recordings, and is therefore independent of the token window. Runs
in minutes on a laptop.

    python runs/compute_surprisal.py --unit syllable
    python runs/compute_surprisal.py --unit phone --corpus aemri frmri
"""

from _common import announce, jobs, parser, skip_existing

from utils.lexicon import surprisal_table
from utils.pipeline import load_dataset


def main():
    args = parser(__doc__, window=False).parse_args()
    for spec, paths in jobs(args):
        announce(spec, paths, "surprisal")
        if skip_existing(paths.surprisal, args.overwrite):
            continue
        df = load_dataset(spec, paths.dataset, unit=args.unit)
        if args.limit:
            df = df.head(args.limit)
        table = surprisal_table(spec, df, root=args.root, unit=args.unit)
        table.to_csv(paths.surprisal, index=False)
        print(f"  wrote {paths.surprisal} ({len(table):,} rows)")


if __name__ == "__main__":
    main()
