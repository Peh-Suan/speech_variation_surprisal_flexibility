#!/usr/bin/env python3
"""
Stage 4: encode every token once and cache the latent vectors.

Distinctiveness, conditional entropy and mutual information all need the same
encoder output. Previously each of them decoded the recordings from scratch,
which is three passes over the video for one set of numbers. This does the pass
once and writes an .npz that the other three read.

    python runs/extract_latents.py --unit syllable --corpus aemri
    python runs/extract_latents.py --unit phone --corpus aemri --window prop25
"""

import pickle

import numpy as np

from _common import announce, jobs, parser, require, skip_existing

from utils.featurecache import FeatureCache
from utils.pipeline import (RecordingStats, best_device, check_subjects, extract_latents,
                            load_encoders, load_dataset, token_frame)


def main():
    p = parser(__doc__, device=True)
    p.add_argument("--no-cache", action="store_true",
                   help="ignore the feature cache and decode per token")
    p.add_argument("--batch-size", type=int, default=32,
                   help="tokens encoded per forward pass. Padding is masked by "
                        "`lengths`, so this does not change the latents; 1 "
                        "reproduces the old one-at-a-time behaviour")
    args = p.parse_args()
    device = args.device or best_device()

    for spec, paths in jobs(args):
        announce(spec, paths, "extract latents")
        if skip_existing(paths.latents, args.overwrite):
            continue

        require(paths.art_ckpt, "train_encoders.py", args, spec.name)
        require(paths.ac_ckpt, "train_encoders.py", args, spec.name)

        window = spec.window(paths.window.name)
        df = load_dataset(spec, paths.dataset, unit=args.unit)
        tokens = token_frame(df, unit=args.unit)
        if args.limit:
            tokens = tokens.head(args.limit)

        stats = RecordingStats(spec, paths.video_stats) if spec.rec_stats else None
        art_scalers = ac_scalers = None
        if spec.art_scaler_mode:
            with open(paths.art_scalers, "rb") as f:
                art_scalers = pickle.load(f)
        with open(paths.ac_scalers, "rb") as f:
            ac_scalers = pickle.load(f)

        cache = None
        if not args.no_cache:
            candidate = FeatureCache(paths.feature_cache)
            if candidate.exists():
                cache = candidate.open()
                print(f"  feature cache: {len(cache):,} tokens")

        check_subjects(tokens, ac_scalers, "the acoustic scalers")
        if art_scalers is not None:
            check_subjects(tokens, art_scalers, "the articulatory scalers")

        art_model, ac_model = load_encoders(spec, paths.art_ckpt, paths.ac_ckpt,
                                            device=device)
        out = extract_latents(spec, tokens, art_model, ac_model, unit=args.unit,
                              art_window=window, stats=stats,
                              art_scalers=art_scalers, ac_scalers=ac_scalers,
                              device=device, cache=cache,
                              batch_size=args.batch_size)
        np.savez_compressed(paths.latents, unit=args.unit,
                            window=paths.window.name, **out)
        ha, hc = out["has_art"], out["has_ac"]
        print(f"  {len(ha):,} of {len(tokens):,} tokens yielded at least one stream")
        print(f"    articulatory {ha.sum():,} ({out['art'].shape[1]}-d)   "
              f"acoustic {hc.sum():,} ({out['ac'].shape[1]}-d)   "
              f"both {(ha & hc).sum():,}")
        if (ha & ~hc).sum() or (~ha & hc).sum():
            print(f"    single-stream: {(ha & ~hc).sum():,} articulatory-only, "
                  f"{(~ha & hc).sum():,} acoustic-only "
                  f"(kept for distances, excluded from entropy and MI)")
        print(f"  wrote {paths.latents}")


if __name__ == "__main__":
    main()
