#!/usr/bin/env python3
"""
Stage 1: build the processed dataset -- one row per adjacent unit pair.

Reads each corpus's alignments, reconciles them against the lexical
pronunciation, and writes processed_<corpus>.csv. Needs the recordings (for the
paths it records) and the alignment files; no torch, no GPU.

The pronunciation dictionary comes from the corpus registry. That is the fix
for the BE ultrasound branch, which previously loaded the FRENCH dictionary in
both the phone and syllable trees: BEUS words were looked up in a French
lexicon, so English entries arrived tagged "(en)...(fr)" -- which normalize_ipa
turns into stray e, n, f, r phones -- and forms that also exist in French got
their French pronunciation. Measured on the word forms that survived the old
build, the correct dictionary raises the alignment rate from 36% to 67%.

Rebuilding a dataset invalidates everything downstream of it, so this refuses
to overwrite unless you pass --overwrite.

    python runs/build_dataset.py --unit phone    --corpus beus --overwrite
    python runs/build_dataset.py --unit syllable --corpus beus --overwrite
"""

import os

import pandas as pd
import tqdm

from _common import announce, jobs, parser, skip_existing

from utils import data_processing, sources

UNIT_ARG = {"phone": "phoneme", "syllable": "syllable"}


def main():
    args = parser(__doc__, window=False).parse_args()

    for spec, paths in jobs(args):
        announce(spec, paths, "build dataset")
        if skip_existing(paths.dataset, args.overwrite):
            continue

        lexicon = sources.lexicon_kwargs(spec, args.root)
        print("  lexicon:", ", ".join(
            f"{k}={os.path.basename(v) if isinstance(v, str) else type(v).__name__ + f'({len(v):,} entries)'}"
            for k, v in lexicon.items()))

        found = sources.find(spec, args.root)
        if not found:
            print(f"  no recordings found under {spec.dataset_path} -- is the "
                  "drive attached? skipping.")
            continue
        if args.limit:
            found = found[:args.limit]
        print(f"  {len(found):,} recordings")

        frames = []
        for source, art_f, ac_f in tqdm.tqdm(found, desc=f"{spec.name} alignments"):
            part = data_processing.alignment_to_dataframe(
                source, lang=spec.lang, unit_type=UNIT_ARG[args.unit], **lexicon)
            part["art_f"] = art_f
            part["ac_f"] = ac_f
            frames.append(part)

        df = pd.concat(frames, ignore_index=True)
        os.makedirs(os.path.dirname(paths.dataset), exist_ok=True)
        df.to_csv(paths.dataset, index=False)
        print(f"  {len(df):,} unit pairs, {df['word'].nunique():,} word types "
              f"-> {paths.dataset}")
        print("  NOTE: everything downstream of this dataset is now stale -- "
              "rerun surprisal, encoders, latents, distances, entropy, MI, combine.")


if __name__ == "__main__":
    main()
