#!/usr/bin/env python3
"""
Stage 3a: decode every recording once and cache the token features.

Optional, but it is the difference between a training epoch that re-seeks into
compressed video for every token and one that slices a memmap. Measured on
USC-TIMIT MRI over an external drive: ~19 ms/token seeking against ~1.5 ms
sequential. Three epochs pay that three times, and stage 4 pays it again.

Build it once per (unit, corpus, window); `train_encoders.py` and
`extract_latents.py` pick it up automatically if it is there.

    python runs/build_cache.py --unit syllable --corpus aemri
    python runs/build_cache.py --unit syllable --corpus aemri --window prop25

Sizes are substantial -- roughly 1.6 GB for aemri syllables, 34 GB for BEUS
phones -- so it lives under .cache/ and SVSF_CACHE can put it on another disk.
"""

import os

from _common import announce, jobs, parser, skip_existing

from utils.featurecache import FeatureCache, compare_against_direct
from utils.pipeline import RecordingStats, load_dataset, token_frame


def main():
    p = parser(__doc__)
    p.add_argument("--force", action="store_true",
                   help="build even for a corpus whose spec says a cache does "
                        "not pay for itself")
    p.add_argument("--verify", type=int, default=25,
                   help="tokens to compare against per-token extraction "
                        "afterwards; 0 to skip")
    args = p.parse_args()

    for spec, paths in jobs(args):
        announce(spec, paths, "build feature cache")
        if not spec.cache_recommended and not args.force:
            print(f"  skipping: {spec.name} reads well at random, so a cache "
                  f"buys little here (--force to build anyway)")
            continue

        cache = FeatureCache(paths.feature_cache)
        if cache.exists() and not args.overwrite:
            print(f"  exists, skipping: {paths.feature_cache}  (--overwrite to redo)")
            continue

        window = spec.window(paths.window.name)
        df = load_dataset(spec, paths.dataset, unit=args.unit)
        tokens = token_frame(df, unit=args.unit)
        if args.limit:
            tokens = tokens.head(args.limit)

        stats = None
        if spec.rec_stats:
            stats = RecordingStats(spec, paths.video_stats)
            stats.build(tokens["art_f"].unique())
            stats.save()

        print(f"  {len(tokens):,} tokens across "
              f"{tokens['art_f'].nunique():,} recordings")
        report = FeatureCache.build(spec, tokens, paths.feature_cache,
                                    window=window, stats=stats)
        print(f"  cached {report['cached']:,}/{report['tokens']:,} tokens "
              f"({report['dropped']:,} the extractor rejected)")
        print(f"  {report['frames']:,} frames of {report['frame_shape']}, "
              f"{report['bytes']/1e9:.2f} GB -> {paths.feature_cache}")
        odd = report.get("wrong_shape_recordings") or {}
        if odd:
            print(f"  {len(odd)} recording(s) skipped for not matching "
                  f"{spec.expected_frame_shape}, as the original notebooks did:")
            for rec, shape in list(odd.items())[:5]:
                print(f"    {os.path.basename(rec)}: {tuple(shape)}")

        if args.verify:
            check = compare_against_direct(spec, cache, tokens, stats=stats,
                                           window=window, n=args.verify)
            print(f"  against per-token extraction: {check['identical']} identical, "
                  f"{check['differing']} differing, {check['unavailable']} unavailable")
            if check["max_abs_diff"] is not None:
                print(f"    max |diff| {check['max_abs_diff']:.4g}, "
                      f"median {check['median_abs_diff']:.4g}")
            if check["differing"]:
                print("    Differences are expected: seeking into compressed video "
                      "is approximate, a sequential read is not. The cached "
                      "features are the better-aligned ones.")


if __name__ == "__main__":
    main()
