#!/usr/bin/env python3
"""
Stage 8: one long-format table per corpus, ready for the mixed models.

Joins the processed dataset with the surprisal columns, the pair distances and
the per-unit flexibility / entropy values, adds the cross-syllable-boundary
flag, and melts the two distance columns into (modality, data_space, dist), the
shape the analysis notebooks read.

    python runs/combine.py --unit syllable --corpus aemri
"""

import numpy as np
import pandas as pd

from _common import announce, jobs, parser, require, skip_existing

from utils.pipeline import load_dataset


def main():
    args = parser(__doc__).parse_args()

    for spec, paths in jobs(args):
        announce(spec, paths, "combine")
        if skip_existing(paths.combined, args.overwrite):
            continue

        for path, stage in ((paths.distances, "compute_distances.py"),
                            (paths.surprisal, "compute_surprisal.py"),
                            (paths.mi, "estimate_mi.py")):
            require(path, stage, args, spec.name)

        u = args.unit
        df = load_dataset(spec, paths.dataset, unit=u)

        # The boundary flag rewrites word_id in place, so it goes first and the
        # rest of the join is positional, exactly as before.
        if u == "phone":
            from utils.data_processing import add_boundary_column
            df = add_boundary_column(df, lang=spec.lang)

        for source in (paths.distances, paths.surprisal):
            extra = pd.read_csv(source)
            if len(extra) != len(df):
                raise ValueError(
                    f"{source} has {len(extra):,} rows but the dataset has "
                    f"{len(df):,}; these are joined positionally, so they must "
                    "come from the same dataset csv")
            for col in extra.columns:
                df[col] = extra[col].values

        flex = pd.read_csv(paths.mi)
        flexibility = dict(zip(flex["phone"], flex["flexibility"]))
        entropy = dict(zip(flex["phone"], flex["H_art_given_ac_phone"]))
        for i in ("1", "2"):
            df[f"{u}{i}_flexibility"] = df[f"{u}{i}"].map(flexibility)
            df[f"{u}{i}_H_art_given_ac"] = df[f"{u}{i}"].map(entropy)

        # Pairs the instrument cannot see properly. The analysis notebooks
        # filter on this, and in R a missing column silently yields zero rows,
        # so it is written for every corpus -- False throughout for the MRI
        # ones rather than absent.
        df["is_limited"] = [
            int(spec.is_limited(a, b, unit=u))
            for a, b in zip(df[f"{u}1"], df[f"{u}2"])
        ]

        df["token_id"] = f"{paths.tag}_" + df.index.astype(str)
        df["unit_type"] = u
        df["window"] = paths.window.name

        dist_cols = ["ac_latent_dist", "art_latent_dist"]
        long = df.melt(
            id_vars=[c for c in df.columns if c not in dist_cols],
            value_vars=dist_cols, var_name="dist_type", value_name="dist")
        long[["modality", "data_space"]] = (
            long["dist_type"].str.extract(r"^(ac|art)_(latent|raw)_dist$"))
        long = long.drop(columns="dist_type")
        long.to_csv(paths.combined, index=False)

        usable = long.dropna(subset=["dist", "bigram_surprisal"])
        lim = long["is_limited"].mean()
        print(f"  {len(long):,} rows ({len(usable):,} with both a distance and "
              f"a surprisal value) -> {paths.combined}")
        print(f"  is_limited: {lim:.1%} of rows flagged"
              + ("" if spec.limited_phones else " (no blind spots for this instrument)"))


if __name__ == "__main__":
    main()
