#!/usr/bin/env python3
"""
Stage 5: latent distance between the two members of each adjacent unit pair.

Reads the cached latents rather than the recordings, so it is fast. The output
is row-aligned with the processed dataset, as the combine stage expects.

A pair whose members did not both encode -- a token the extractor rejected --
gets NaN, exactly as the notebooks produced None.

    python runs/compute_distances.py --unit syllable --corpus aemri
"""

import numpy as np
import pandas as pd

from _common import announce, jobs, parser, require, skip_existing

from utils.pipeline import latent_lookup, load_dataset, stream_masks


def main():
    args = parser(__doc__).parse_args()

    for spec, paths in jobs(args):
        announce(spec, paths, "pair distances")
        if skip_existing(paths.distances, args.overwrite):
            continue

        require(paths.latents, "extract_latents.py", args, spec.name)
        require(paths.dataset, "build_dataset.py", args, spec.name)
        latents = dict(np.load(paths.latents, allow_pickle=True))
        index, key = latent_lookup(latents)
        art, ac = latents["art"], latents["ac"]
        has_art, has_ac = stream_masks(latents)

        df = load_dataset(spec, paths.dataset, unit=args.unit)
        if args.limit:
            df = df.head(args.limit)

        u = args.unit
        out = {"art_latent_dist": [], "ac_latent_dist": []}
        for _, r in df.iterrows():
            i = index.get(key(r["art_f"], r[f"{u}1_start"], r[f"{u}1_end"]))
            j = index.get(key(r["art_f"], r[f"{u}2_start"], r[f"{u}2_end"]))
            # The two modalities are measured independently, matching the
            # original per-corpus notebooks: a pair whose MFCCs failed still
            # contributes an articulatory distance, and vice versa.
            for name, vecs, mask in (("art_latent_dist", art, has_art),
                                     ("ac_latent_dist", ac, has_ac)):
                if i is None or j is None or not (mask[i] and mask[j]):
                    out[name].append(np.nan)
                else:
                    out[name].append(float(np.linalg.norm(vecs[i] - vecs[j])))

        table = pd.DataFrame(out)
        table.to_csv(paths.distances, index=False)
        n_art = table["art_latent_dist"].notna().sum()
        n_ac = table["ac_latent_dist"].notna().sum()
        both = table.notna().all(axis=1).sum()
        print(f"  of {len(table):,} pairs: {n_art:,} articulatory, {n_ac:,} acoustic, "
              f"{both:,} with both")
        print(f"  wrote {paths.distances}")


if __name__ == "__main__":
    main()
