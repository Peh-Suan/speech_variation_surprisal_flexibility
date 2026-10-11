#!/usr/bin/env python3
"""
Stage 3: train the articulatory and acoustic autoencoders.

The expensive stage -- it decodes every token's video or EMA record once per
epoch -- and the only one that must run where the recordings are. Checkpoints
every 50 batches and resumes from where it stopped, so it survives being
interrupted.

The acoustic encoder is window-independent (only the ARTICULATORY window is
widened), so it is keyed by corpus and trained once; a later window variant
reuses it. Pass --skip-ac to make that explicit, or --only ac to train just it.

    python runs/train_encoders.py --unit syllable --corpus aemri
    python runs/train_encoders.py --unit phone --corpus aemri --window prop25 --only art
"""

import os

from _common import announce, jobs, parser

from utils.featurecache import FeatureCache
from utils.pipeline import RecordingStats, best_device, load_dataset, token_frame, window_report
from utils.training import fit_ac_scalers, fit_art_scalers, train_autoencoder


def main():
    p = parser(__doc__, device=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="static vs velocity weight in the reconstruction loss")
    p.add_argument("--only", choices=("art", "ac"), default=None)
    p.add_argument("--no-cache", action="store_true",
                   help="ignore the feature cache and decode per token")
    args = p.parse_args()
    device = args.device or best_device()

    for spec, paths in jobs(args):
        announce(spec, paths, "train encoders")
        window = spec.window(paths.window.name)

        df = load_dataset(spec, paths.dataset, unit=args.unit)
        tokens = token_frame(df, unit=args.unit)
        if args.limit:
            tokens = tokens.head(args.limit)
        print(f"  {len(tokens):,} distinct tokens")

        if not window.is_identity:
            print("  window: " + ", ".join(
                f"{k}={v}" for k, v in
                window_report(tokens, window, sample=300).items()))

        cache = None
        if not args.no_cache:
            candidate = FeatureCache(paths.feature_cache)
            if candidate.exists():
                cache = candidate.open()
                print(f"  feature cache: {len(cache):,} tokens "
                      f"({os.path.relpath(paths.feature_cache, args.root)})")
            else:
                print("  no feature cache; decoding per token. Building one with "
                      "runs/build_cache.py is usually much faster.")

        if args.only != "ac":
            stats = None
            if spec.rec_stats:
                stats = RecordingStats(spec, paths.video_stats)
                stats.build(tokens["art_f"].unique())
                stats.save()
                print(f"  intensity statistics for {len(stats)} recordings")

            art_scalers = fit_art_scalers(spec, tokens, window, paths.art_scalers,
                                          overwrite=args.overwrite, cache=cache)
            os.makedirs(paths.ckpt_dir, exist_ok=True)
            train_autoencoder(
                spec, tokens, spec.build_art_model(device), stream="art",
                ckpt_path=paths.art_ckpt, log_path=paths.art_log,
                window=window, stats=stats, scalers=art_scalers, device=device,
                n_epochs=args.epochs, batch_size=args.batch_size,
                alpha=args.alpha, cache=cache)

        if args.only != "art":
            if os.path.exists(paths.ac_ckpt) and not args.overwrite:
                print(f"  acoustic encoder already trained: "
                      f"{os.path.relpath(paths.ac_ckpt)} (--overwrite to redo)")
            else:
                ac_scalers = fit_ac_scalers(tokens, paths.ac_scalers,
                                            overwrite=args.overwrite)
                os.makedirs(os.path.dirname(paths.ac_ckpt), exist_ok=True)
                train_autoencoder(
                    spec, tokens, spec.build_ac_model(device), stream="ac",
                    ckpt_path=paths.ac_ckpt, log_path=paths.ac_log,
                    scalers=ac_scalers, device=device, n_epochs=args.epochs,
                    batch_size=args.batch_size, alpha=args.alpha)


if __name__ == "__main__":
    main()
