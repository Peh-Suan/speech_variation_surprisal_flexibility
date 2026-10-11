"""
The autoencoder training loop, with resume.

One loop, used for both streams and every corpus. It reproduces what the
per-corpus notebooks did -- same shuffle seed, same loss scales pass, same
checkpoint cadence -- with the corpus-specific parts coming from a CorpusSpec
and the token window from a Window.

Resume is by (epoch, batch index): the shuffle is seeded with 42 + epoch, so
re-running an interrupted epoch reproduces the same order and skipping the
completed batches lands on exactly the samples that were still pending. That
matters here because a single epoch over the MRI corpora is hours of video
decoding, and the runs do get interrupted.
"""

from __future__ import annotations

import os
import pickle

import pandas as pd
import tqdm

from utils.pipeline import ac_batches, art_batches


def _save(path, log_path, model, optimizer, epoch, batch_idx, log):
    """Written to a temp file and renamed, so an interrupt cannot truncate it."""
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"epoch": epoch, "batch_idx": batch_idx,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, path + ".tmp")
    os.replace(path + ".tmp", path)
    pd.DataFrame(log).to_csv(log_path, index=False)


def _resume(path, model, optimizer, device):
    import torch
    if not os.path.exists(path):
        return 1, 0
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f"  resuming at epoch {ckpt['epoch']}, after batch {ckpt['batch_idx']}")
        return ckpt["epoch"], ckpt["batch_idx"]
    model.load_state_dict(ckpt)
    print("  loaded an old-style checkpoint: no optimizer state (expect a brief "
          "loss bump) and no position, so training restarts from epoch 1")
    return 1, 0


def fit_art_scalers(spec, token_df, window, path, overwrite=False, verbose=True,
                    cache=None):
    """
    Per-speaker articulatory scalers, for the corpora whose features are raw
    coordinates rather than pixel intensities.

    They are fitted on the SAME window the model will be trained on: a widened
    window includes different frames, so reusing a base-window scaler would
    standardise the widened features by the wrong statistics.
    """
    from utils.scaling import describe, fit_scalers

    if spec.art_scaler_mode is None:
        return None
    if os.path.exists(path) and not overwrite:
        with open(path, "rb") as f:
            return pickle.load(f)

    from utils.pipeline import art_features

    def raw(row):
        return art_features(spec, row, window=window, cache=cache)

    scalers = fit_scalers(token_df, raw, mode=spec.art_scaler_mode, verbose=verbose)
    if verbose:
        describe(scalers, name=f"{spec.name} art ")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scalers, f)
    return scalers


def fit_ac_scalers(token_df, path, overwrite=False, verbose=True, **mfcc_kwargs):
    """Per-speaker MFCC scalers. Window-independent: acoustics are not widened."""
    from utils.pipeline import ac_features
    from utils.scaling import describe, fit_scalers

    if os.path.exists(path) and not overwrite:
        with open(path, "rb") as f:
            return pickle.load(f)

    scalers = fit_scalers(token_df, lambda r: ac_features(r, **mfcc_kwargs),
                          mode="per_feature", verbose=verbose)
    if verbose:
        describe(scalers, name="ac ")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scalers, f)
    return scalers


def train_autoencoder(spec, token_df, model, *, stream, ckpt_path, log_path,
                      window=None, stats=None, scalers=None, device="cpu",
                      n_epochs=3, batch_size=32, alpha=0.5, lr=1e-3,
                      save_every=50, min_frames=1, scale_batches=2000,
                      duration_mode="off", cache=None, **mfcc_kwargs):
    """
    Train one autoencoder to completion, resuming if a checkpoint is present.

    `stream` is "art" or "ac" and decides which batch generator is used. The
    loss scales are estimated once, up front, so `alpha` means the same thing
    from the first batch to the last.
    """
    import torch
    from utils.train import estimate_loss_scales, train_step

    def batches(df, progress=True, skip=0):
        if stream == "art":
            return art_batches(spec, df, window=window, stats=stats,
                               scalers=scalers, batch_size=batch_size,
                               min_frames=min_frames, progress=progress,
                               skip=skip, cache=cache)
        return ac_batches(df, scalers, batch_size=batch_size,
                          min_frames=min_frames, progress=progress, skip=skip,
                          **mfcc_kwargs)

    time_dim = spec.time_dim if stream == "art" else 1

    head = token_df.sample(frac=1, random_state=42).reset_index(drop=True)
    if scale_batches:
        head = head.head(scale_batches)
    scales = estimate_loss_scales(((x, L) for x, L, _ in batches(head)),
                                  time_dim=time_dim, max_batches=50)
    print(f"  loss scales: var(x)={scales[0]:.5f} var(dx)={scales[1]:.5f} "
          f"(ratio {scales[0] / max(scales[1], 1e-12):.1f}x)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch, start_batch = _resume(ckpt_path, model, optimizer, device)

    log_keys = ("total", "static", "delta", "duration")
    log = {"epoch": [], "batch_idx": [], **{k: [] for k in log_keys}}
    if os.path.exists(log_path):
        try:
            log = pd.read_csv(log_path).to_dict("list")
        except Exception:
            pass

    model.train()
    for epoch in range(start_epoch, n_epochs + 1):
        shuffled = token_df.sample(frac=1, random_state=42 + epoch).reset_index(drop=True)
        skip = start_batch if epoch == start_epoch else 0
        bar = tqdm.tqdm(desc=f"{spec.name} {stream} e{epoch}")

        for batch_idx, (x, lengths, durations) in enumerate(
                batches(shuffled, progress=False, skip=skip), start=skip + 1):
            x, lengths = x.to(device), lengths.to(device)
            durations = durations.to(device)

            step = train_step(model, optimizer, x, lengths, durations,
                              alpha=alpha, scales=scales,
                              duration_mode=duration_mode)

            log["epoch"].append(epoch)
            log["batch_idx"].append(batch_idx)
            for k in log_keys:
                log[k].append(step[k])
            bar.update(1)
            bar.set_description(
                f"{spec.name} {stream} e{epoch} b{batch_idx}: "
                f"{step['total']:.4f} (s {step['static']:.3f} d {step['delta']:.3f})")

            if batch_idx % save_every == 0:
                _save(ckpt_path, log_path, model, optimizer, epoch, batch_idx, log)

        bar.close()
        _save(ckpt_path, log_path, model, optimizer, epoch + 1, 0, log)

    _save(ckpt_path, log_path, model, optimizer, n_epochs + 1, 0, log)
    return model
