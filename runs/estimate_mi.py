#!/usr/bin/env python3
"""
Stage 7: per-unit mutual information between articulation and acoustics,
and the flexibility measure derived from it.

Thin wrapper over utils/mi.py, reading the cached latents. Folds are grouped by
recording, which is why extract_latents stores the recording alongside each
token.

    python runs/estimate_mi.py --unit syllable --corpus aemri
"""

import numpy as np

from _common import announce, jobs, parser, require, skip_existing


def main():
    p = parser(__doc__, device=True)
    p.add_argument("--min-tokens", type=int, default=30,
                   help="units with fewer tokens get no estimate. 30 is right "
                        "for phones; syllables are far sparser, so check the "
                        "token counts before assuming it leaves you any levels")
    args = p.parse_args()

    from utils import mi

    for spec, paths in jobs(args):
        announce(spec, paths, "mutual information")
        if skip_existing(paths.mi, args.overwrite):
            continue

        require(paths.latents, "extract_latents.py", args, spec.name)
        from utils.pipeline import paired
        latents = np.load(paths.latents, allow_pickle=True)
        art, ac, labels, groups, keep = paired(latents)
        art = art.astype(np.float32); ac = ac.astype(np.float32)
        labels, groups = list(labels), list(groups)
        print(f"  {len(art):,} tokens with both streams of {len(keep):,} extracted")

        res = mi.compute_all(art, ac, labels, groups)
        table = mi.per_phoneme(res, min_tokens=args.min_tokens)
        table.to_csv(paths.mi, index=False)
        print(f"  {len(table)} units -> {paths.mi}")


if __name__ == "__main__":
    main()
