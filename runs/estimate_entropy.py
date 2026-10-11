#!/usr/bin/env python3
"""
Stage 6: conditional entropies H(art|ac) and H(ac|art), per unit.

Cross-validated MDN density estimation on the cached latents. Both directions
are written, as the combine stage and the flexibility measure expect.

    python runs/estimate_entropy.py --unit syllable --corpus aemri
"""

import pickle

import numpy as np

from _common import announce, jobs, parser, require, skip_existing

from utils.pipeline import best_device, conditional_entropy, paired


def main():
    p = parser(__doc__, device=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--min-tokens", type=int, default=30,
                   help="units with fewer tokens get no entropy estimate")
    args = p.parse_args()
    device = args.device or best_device()

    for spec, paths in jobs(args):
        announce(spec, paths, "conditional entropy")
        if skip_existing(paths.entropy("art_given_ac"), args.overwrite):
            continue

        require(paths.latents, "extract_latents.py", args, spec.name)
        latents = np.load(paths.latents, allow_pickle=True)
        art, ac, labels, _, keep = paired(latents)
        print(f"  {len(art):,} tokens with both streams of {len(keep):,} extracted"
              f" ({(~keep).sum():,} have only one), "
              f"{len(set(labels.tolist())):,} distinct units")

        for direction, (X, Y) in (("art_given_ac", (ac, art)),
                                  ("ac_given_art", (art, ac))):
            print(f"  {direction} ...")
            entropies, _ = conditional_entropy(
                X, Y, labels, k=args.folds, min_tokens=args.min_tokens,
                device=device)
            with open(paths.entropy(direction), "wb") as f:
                pickle.dump(entropies, f)
            print(f"    {len(entropies)} units estimated -> "
                  f"{paths.entropy(direction)}")


if __name__ == "__main__":
    main()
