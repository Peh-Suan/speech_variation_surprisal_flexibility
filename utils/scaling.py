"""
Per-subject feature scaling, shared by the acoustic and articulatory pipelines.

The accumulator is modality-agnostic — it is just a streaming per-subject mean
and variance over the trailing (feature) axes — so the only thing that had to
change to make the MFCC version general is that it now takes the feature
extractor as an argument instead of calling get_mfccs directly.

What is NOT modality-agnostic is the choice of `mode`.

    "per_feature"  z-score every feature dimension independently.
                   Correct for MFCCs: cepstral coefficients are not spatial
                   axes, they have wildly different natural ranges (c0 carried
                   ~35% of total variance unscaled), and nothing about their
                   relative magnitude is meaningful.

                   WRONG for coordinates. It divides x, y and z by different
                   numbers, so it warps the space: an equal physical
                   displacement becomes 8x larger in the lateral direction than
                   the vertical one, purely because the tongue moves less
                   laterally. Your `dist` variable is a Euclidean distance, so
                   this silently rewrites what it measures.

    "isotropic"    subtract a per-sensor, per-axis mean; divide everything by
                   ONE scale per subject (RMS displacement in mm). Removes
                   anatomy, palate position and sensor placement, and removes
                   overall vocal-tract size, while leaving every direction and
                   every sensor's relative amplitude intact. The default for
                   coordinates.

                   This is the EMA analogue of what the sequence-level STN does
                   for the video modalities: centre and scale away the nuisance,
                   do not rotate or stretch the articulation.

    "per_sensor"   as isotropic, but one scale per sensor. Equalises the
                   articulators, so tongue tip and jaw contribute equally even
                   though the tip moves much more. A defensible choice, but a
                   choice — say so if you use it.
"""

import numpy as np
import tqdm


def fit_scalers(df, extract, subject_col="subject", mode="per_feature", verbose=True):
    """
    One pass over the TRAINING rows -> {subject: (mean, sd)}, broadcastable
    against a (T, ...) feature array.

    `extract`: callable(row) -> (T, ...) array of UNSCALED features, or None.

        Write it as a def, not a lambda, and check for None BEFORE you touch the
        result. This crashes:

            raw = lambda row: get_aeema_articulatory_features(...).reshape(-1, 6, 3)

        because .reshape runs before fit_scalers ever sees the None. This works:

            def raw(row):
                m = get_aeema_articulatory_features(
                        row["art_f"], row["start"], row["end"], scalers=None)
                if m is None or m.shape[0] < MIN_FRAMES:
                    return None
                return m.reshape(-1, 6, 3)

        Define it once and use the same function for the scaler pass and the
        training pass, so both drop exactly the same tokens.

        Note the chicken-and-egg for EMA: if your extractor applies the scaler
        internally (as get_aeema_articulatory_features does), pass a version
        with scaling turned off, or you will fit a scaler on already-scaled
        features.

    Fit on train only and reuse on every split. Refitting per split puts the
    splits in different spaces and makes their latents incomparable.
    """
    if mode not in ("per_feature", "isotropic", "per_sensor"):
        raise ValueError("mode must be 'per_feature', 'isotropic' or 'per_sensor'")

    acc, used, skipped = {}, 0, 0
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="extracting features"):
        m = extract(row)
        if m is None:
            skipped += 1
            continue
        used += 1
        m = np.asarray(m, dtype=np.float64)          # accumulate in float64
        s = row[subject_col]
        n, sm, sq = acc.get(s, (0, 0.0, 0.0))
        acc[s] = (n + m.shape[0], sm + m.sum(0), sq + (m ** 2).sum(0))

    if verbose:
        # surface the drops: these tokens will be missing from your latents too,
        # and the acoustic and articulatory extractors do not drop the same ones
        print(f"fit_scalers: used {used} rows, skipped {skipped} "
              f"({100 * skipped / max(used + skipped, 1):.1f}%), "
              f"{len(acc)} subjects")
    if not acc:
        raise ValueError("extract returned None for every row — check the extractor")

    out = {}
    for s, (n, sm, sq) in tqdm.tqdm(acc.items(), total=len(acc), desc="fitting scalers"):
        mean = sm / n
        var = np.maximum(sq / n - mean ** 2, 1e-12)

        if mode == "per_feature":
            sd = np.sqrt(var)
        else:
            if var.ndim < 2:
                raise ValueError(
                    f"mode={mode!r} needs features shaped (T, sensors, coords); "
                    f"got a flat feature of shape {var.shape}. Use 'per_feature'.")
            if mode == "isotropic":
                # RMS 3-D displacement per sensor: one number for the subject
                sd = np.sqrt(var.sum() / np.prod(var.shape[:-1]))
            else:                                     # per_sensor
                sd = np.sqrt(var.sum(axis=-1, keepdims=True))
        out[s] = (mean.astype(np.float32), np.asarray(sd, dtype=np.float32))
    return out


def apply_scaler(m, scaler):
    """
    (m - mean) / sd, broadcasting for all three modes.
 
    Check your extractor's return value for None BEFORE calling this. numpy
    turns None into nan rather than raising — np.asarray(None, np.float32) is
    array(nan) — so scaling a dropped token silently yields an all-NaN array
    with the SHAPE OF THE SCALER, which then sails past an `if m is None or
    m.shape[0] < MIN_FRAMES` test and poisons the batch. Hence the explicit
    guard here.
    """
    if m is None:
        raise TypeError(
            "apply_scaler received None. Check the extractor's return value "
            "before scaling — numpy would otherwise convert it to NaN silently.")
    mean, sd = scaler
    m = np.asarray(m, dtype=np.float32)
    mean = np.asarray(mean)
    if m.ndim - 1 != mean.ndim or m.shape[1:] != mean.shape:
        raise ValueError(
            f"feature shape {m.shape[1:]} does not match the scaler's {mean.shape}. "
            f"Most often this is an ordering problem: reshape to (T, sensors, coords) "
            f"BEFORE fitting and before applying, not in between.")
    return (m - mean) / sd


def describe(scalers, name=""):
    """Print what was fitted — worth eyeballing once."""
    for s, (mean, sd) in scalers.items():
        sd_arr = np.atleast_1d(np.asarray(sd).ravel())
        print(f"  {name}{s}: mean shape {np.shape(mean)}, sd shape {np.shape(sd)}, "
              f"sd range {sd_arr.min():.3f}..{sd_arr.max():.3f}")