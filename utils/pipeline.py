"""
The stages every analysis shares.

Feature extraction, encoder loading, latent extraction, the cross-validated MDN
conditional entropies and the pairwise distances were previously written out
once per corpus per stage. They are identical apart from the facts now held in
utils.corpora, so they live here and take a CorpusSpec.

The one substantive addition is `window`: a utils.windows.Window that decides
how a token's [start, end] becomes the interval features are read from. The
articulatory and acoustic streams take SEPARATE windows, because the LabPhon
question is specifically whether the ARTICULATORY token should be wider than
its acoustic segmentation. The default is `art_window` widened, `ac_window`
left at the acoustic boundaries.

Nothing here writes files. Stages hand back arrays and frames; the notebooks
decide where they land, using utils.paths.RunPaths.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from utils.windows import get_window


# --------------------------------------------------------------------------- #
# Devices
# --------------------------------------------------------------------------- #

def best_device(prefer_mps=True, verbose=True):
    """
    The fastest device available, preferring Apple's GPU on a Mac.

    `prefer_mps=False` is almost never what you want here: on Apple Silicon it
    means CPU, because there is no CUDA to fall back to. Some ops in the STN
    and the GRU have historically been missing on MPS -- set
    PYTORCH_ENABLE_MPS_FALLBACK=1 and those run on CPU while everything else
    stays on the GPU, rather than raising.
    """
    import torch
    if prefer_mps and torch.backends.mps.is_available():
        if verbose:
            # Say which mode the run is in either way, so a log records it.
            # The video corpora CANNOT train without the fallback: MPS has no
            # aten::grid_sampler_2d_backward for the STN.
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
                print("  device: mps (CPU fallback enabled for unimplemented "
                      "ops, e.g. the STN's grid_sample backward)")
            else:
                print("  device: mps, WITHOUT the CPU fallback. Video corpora "
                      "will fail at the first backward pass -- re-run with "
                      "PYTORCH_ENABLE_MPS_FALLBACK=1")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if verbose:
        try:
            if torch.backends.mps.is_available():
                print("  device: cpu -- but MPS IS available. This will be "
                      "roughly an order of magnitude slower than it needs to be.")
        except Exception:
            pass
    return "cpu"


# --------------------------------------------------------------------------- #
# Per-recording intensity statistics
# --------------------------------------------------------------------------- #

class RecordingStats:
    """
    Pixel mean and sd over a WHOLE recording, cached to disk.

    Whole-recording rather than per-token: normalising each token to mean 0 /
    sd 1 would erase the between-token intensity variation that partly
    distinguishes the tokens. These statistics are a property of the recording,
    so they do not depend on how token windows are drawn and are shared across
    every window variant of a corpus.
    """

    def __init__(self, spec, path, n_sample=400):
        self.spec = spec
        self.path = path
        self.n_sample = n_sample
        self.cache = {}
        if path and os.path.isfile(path):
            with open(path, "rb") as f:
                self.cache = pickle.load(f)

    def __len__(self):
        return len(self.cache)

    def get(self, recording):
        if self.spec.rec_stats is None:
            return None
        if recording in self.cache:
            return self.cache[recording]
        value = (self._beus(recording) if self.spec.rec_stats == "beus"
                 else self._mri(recording))
        if value is not None:
            self.cache[recording] = value
        return value

    def _mri(self, recording):
        import cv2
        cap = cv2.VideoCapture(recording)
        if not cap.isOpened():
            cap.release()
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None
        idxs = np.unique(np.linspace(0, total - 1, min(self.n_sample, total)).astype(int))
        vals = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                vals.append(fr[:, :, 0].astype(np.float32))
        cap.release()
        if not vals:
            return None
        v = np.stack(vals)
        return float(v.mean()), float(v.std()) + 1e-5

    def _beus(self, recording):
        from utils.articulation_processing import _beus_streams
        size = self.spec.extra_getter_kwargs.get("spatial_size", 112)
        out = _beus_streams(recording, size)
        if out is None:
            return None
        us, lp, mask = out
        return ((float(us[:, mask].mean()), float(us[:, mask].std()) + 1e-5),
                (float(lp.mean()), float(lp.std()) + 1e-5))

    def build(self, recordings, progress=True):
        it = recordings
        if progress:
            import tqdm
            it = tqdm.tqdm(list(recordings), desc=f"{self.spec.name} recording stats")
        for r in it:
            self.get(r)
        return self

    def save(self, path=None):
        path = path or self.path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.cache, f)
        return path


# --------------------------------------------------------------------------- #
# Datasets and token frames
# --------------------------------------------------------------------------- #

def load_dataset(spec, csv_path, unit="phone"):
    """The pair-level frame: one row per adjacent unit pair, with `subject`."""
    df = pd.read_csv(csv_path)
    df["subject"] = df["art_f"].apply(spec.subject_from_art_f)
    return df


TOKEN_ORDER = ["art_f", "ac_f", "subject", "start", "end"]


def token_frame(df, unit="phone", sort=True):
    """
    The token-level frame: one row per distinct (start, end, unit, recording).

    Deduplication happens on the ORIGINAL segmentation, before any window is
    applied, so that a widened window never turns one token into two.

    `sort` fixes a reproducibility bug worth knowing about.
    `dataset_df_to_token_df` deduplicates through a Python set and then sorts
    only by (art_f, ac_f, subject), which leaves the order WITHIN a recording
    to set iteration -- and since Python randomises string hashing per process,
    that order differs every run. Three consequences:

      * two runs over the same data encode tokens in different orders, so
        latent files cannot be compared row by row;
      * training is not reproducible run to run;
      * and most seriously, resume is silently wrong. Training shuffles with
        `sample(random_state=42 + epoch)` and resumes by batch index, on the
        stated assumption that the same seed reproduces the same order. It does
        not, if the frame it shuffles arrived in a different order -- so
        resuming a multi-hour job after an interruption continues on a
        different subset than the one it left off in.

    Sorting on the full token identity makes the order a function of the data.
    `sort=False` reproduces the old behaviour if you ever need to match an
    earlier run exactly.
    """
    from utils.data_processing import dataset_df_to_token_df
    tokens = dataset_df_to_token_df(df, unit_type=unit)
    if sort:
        tokens = (tokens.sort_values(TOKEN_ORDER, kind="mergesort")
                        .reset_index(drop=True))
    return tokens


def check_subjects(token_df, scalers, label="scalers"):
    """
    Every speaker in the data must have a scaler, checked before the loop.

    Scalers are keyed by whatever `CorpusSpec.subject_from_art_f` pulls out of
    the filename, so a wrong field index does not fail until the first token is
    scaled -- which on the ultrasound corpora is a KeyError a quarter of a
    million tokens into a run. This turns it into one message naming both sides.
    """
    have = set(scalers)
    want = set(token_df["subject"].unique())
    missing = sorted(want - have)
    if not missing:
        return
    raise KeyError(
        f"{len(missing)} speaker(s) in the data have no entry in {label}: "
        f"{missing[:6]}{' ...' if len(missing) > 6 else ''}\n"
        f"  {label} has {len(have)}: {sorted(have)[:6]}{' ...' if len(have) > 6 else ''}\n"
        f"  This is almost always subject_from_art_f in utils/corpora.py taking "
        f"the wrong underscore-separated field of the filename.")


def window_report(token_df, window, path_col="art_f", sample=None, seed=0):
    """
    What a window actually did, in milliseconds.

    Worth printing before committing to a long run. A nominal 25% pad is a
    silent no-op if the tokens sit at the edges of short recordings and clip
    back to nothing, and at ~23 fps a pad below ~22 ms often buys no extra
    frame at all -- `mean_frames_gained` makes that visible. `sample` limits
    the check to a random subset, since the duration lookups touch the drive.
    """
    window = get_window(window)
    df = token_df
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=seed)

    nominal, left, right, dur = [], [], [], []
    for _, r in df.iterrows():
        s, e = float(r["start"]), float(r["end"])
        l, rt = window.realised_pad(s, e, path=r[path_col])
        nominal.append(window.pad_for(s, e))
        left.append(l)
        right.append(rt)
        dur.append(e - s)

    nominal = np.asarray(nominal)
    left, right, dur = np.asarray(left), np.asarray(right), np.asarray(dur)
    clipped = (left < nominal - 1e-9) | (right < nominal - 1e-9)

    return {
        "window": window.name,
        "rule": window.describe(),
        "n_tokens_checked": int(len(df)),
        "token_duration_ms_median": float(np.median(dur) * 1000),
        "nominal_pad_ms_median": float(np.median(nominal) * 1000),
        "left_pad_ms_median": float(np.median(left) * 1000),
        "right_pad_ms_median": float(np.median(right) * 1000),
        "clipped_frac": float(clipped.mean()),
        "duration_gain_mean": float(((left + right) / np.maximum(dur, 1e-9)).mean()),
    }


def frames_gained(window, durations, fps):
    """
    Expected extra frames per token at a given frame rate -- the sanity check
    that a proportional pad is doing anything at all for a video corpus.
    """
    window = get_window(window)
    durations = np.asarray(durations, dtype=float)
    pads = np.array([window.pad_for(0.0, d) for d in durations])
    return {
        "fps": fps,
        "frame_ms": 1000.0 / fps,
        "mean_frames_gained": float((2 * pads * fps).mean()),
        "tokens_gaining_no_frame_frac": float((2 * pads * fps < 1).mean()),
    }


# --------------------------------------------------------------------------- #
# Token features
# --------------------------------------------------------------------------- #

def art_features(spec, row, window=None, stats=None, scalers=None, cache=None):
    """
    Articulatory features for one token, windowed, reshaped and scaled.
    Returns a float32 array or None.

    `cache` is a utils.featurecache.FeatureCache holding features already
    decoded and windowed in one sequential pass per recording. A hit skips the
    seek-and-decode entirely; a miss falls back to reading the recording, which
    costs nothing extra for the handful of tokens the extractor rejects.
    """
    window = get_window(window)
    art_f = row["art_f"]

    out = cache.get(row) if cache is not None else None
    if out is None:
        start, end = window.apply(row["start"], row["end"], path=art_f)
        out = spec.art_features(art_f, start, end, stats=stats)
    if out is None:
        return None
    # Recordings whose frames are not the shape the encoder was built for are
    # dropped, exactly as the original phone notebooks did. This has to live
    # here rather than only in the feature cache: the cache marks such tokens
    # empty, and a miss falls back to reading the recording directly, which
    # would otherwise put the odd shape straight back into the collate.
    expected = spec.expected_frame_shape
    if expected is not None and tuple(np.shape(out)[-len(expected):]) != expected:
        return None
    if scalers is not None:
        from utils.scaling import apply_scaler
        out = apply_scaler(out, scalers[row["subject"]])
    return np.asarray(out, dtype=np.float32)


def ac_features(row, window=None, scalers=None, **mfcc_kwargs):
    """
    MFCCs for one token. `window` defaults to the acoustic segmentation, which
    is the point: the manuscript's claim is about the ARTICULATORY window.
    """
    from utils.acoustics_processing import get_mfccs
    from utils.mounts import local_path
    window = get_window(window)
    ac_f = local_path(row["ac_f"])
    start, end = window.apply(row["start"], row["end"], path=ac_f)
    out = get_mfccs(ac_f, start, end, **mfcc_kwargs)
    if out is None:
        return None
    if scalers is not None:
        from utils.scaling import apply_scaler
        out = apply_scaler(out, scalers[row["subject"]])
    return np.asarray(out, dtype=np.float32)


def art_batches(spec, token_df, window=None, stats=None, scalers=None,
                batch_size=32, min_frames=1, progress=True, skip=0, cache=None):
    """
    Yields (x, lengths, durations) for the articulatory autoencoder, padded to
    each batch's own maximum length. `skip` drops leading batches, for resuming
    mid-epoch. Duration is the ORIGINAL acoustic duration even when the window
    is widened, so the duration head keeps predicting the linguistic quantity
    rather than the padding rule.
    """
    import torch
    collate = spec.collate()
    rows = token_df.iterrows()
    if progress:
        import tqdm
        total = (len(token_df) + batch_size - 1) // batch_size
        bar = tqdm.tqdm(total=total, desc=f"{spec.name} art batches")
    buf, emitted = [], 0
    for _, row in rows:
        feat = art_features(spec, row, window=window,
                            stats=stats.get(row["art_f"]) if stats is not None else None,
                            scalers=scalers, cache=cache)
        if feat is None or feat.shape[0] < min_frames:
            continue
        buf.append((torch.tensor(feat, dtype=torch.float32),
                    float(row["end"] - row["start"])))
        if len(buf) == batch_size:
            emitted += 1
            if progress:
                bar.update(1)
            if emitted > skip:
                yield collate(buf)
            buf = []
    if buf:
        if progress:
            bar.update(1)
        emitted += 1
        if emitted > skip:
            yield collate(buf)
    if progress:
        bar.close()


def ac_batches(token_df, scalers, batch_size=32, min_frames=1, progress=True,
               window=None, skip=0, **mfcc_kwargs):
    """The acoustic counterpart of art_batches."""
    import torch
    from utils.train import collate_sequence
    if progress:
        import tqdm
        total = (len(token_df) + batch_size - 1) // batch_size
        bar = tqdm.tqdm(total=total, desc="ac batches")
    buf, emitted = [], 0
    for _, row in token_df.iterrows():
        feat = ac_features(row, window=window, scalers=scalers, **mfcc_kwargs)
        if feat is None or feat.shape[0] < min_frames:
            continue
        buf.append((torch.tensor(feat, dtype=torch.float32),
                    float(row["end"] - row["start"])))
        if len(buf) == batch_size:
            emitted += 1
            if progress:
                bar.update(1)
            if emitted > skip:
                yield collate_sequence(buf)
            buf = []
    if buf:
        emitted += 1
        if progress:
            bar.update(1)
        if emitted > skip:
            yield collate_sequence(buf)
    if progress:
        bar.close()


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #

def load_encoders(spec, art_ckpt, ac_ckpt, device="cpu", ac_feature_shape=13):
    """
    Both trained autoencoders in eval mode.

    Checkpoints come in two shapes in this repository: a dict with "model" and
    "optimizer" (the current format) and a bare state_dict (older runs). Both
    load.
    """
    import torch

    def _state(path):
        obj = torch.load(path, map_location=device)
        return obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    art_model = spec.build_art_model(device=device)
    art_model.load_state_dict(_state(art_ckpt))
    art_model.eval()

    ac_model = spec.build_ac_model(device=device, feature_shape=ac_feature_shape)
    ac_model.load_state_dict(_state(ac_ckpt))
    ac_model.eval()

    return art_model, ac_model


def encode_stream(spec, arrays, model, device="cpu", stream="art"):
    """
    Encode a batch of feature arrays through one encoder.

    Collation goes through the SAME functions training uses. That matters for
    the articulatory stream, where the two video layouts differ -- (T, H, W)
    for rtMRI against (T, V, H, W) for ultrasound-plus-lips -- and
    `collate_video` is what inserts or moves the channel axis.

    Batching is exact: padded positions are masked by `lengths` through the CNN,
    the GRU and the pooling. Checked against one-at-a-time encoding on tokens of
    mixed length -- max latent difference 1.2e-07 against a latent scale of
    0.035.
    """
    import torch
    from utils.train import collate_sequence

    collate = spec.collate() if stream == "art" else collate_sequence
    x, lengths, _ = collate([(torch.from_numpy(a).float(), 0.0) for a in arrays])
    with torch.no_grad():
        _, z = model(x.to(device), lengths.to(device), return_latent=True)[:2]
    return z.detach().cpu().numpy().reshape(len(arrays), -1)


def extract_latents(spec, token_df, art_model, ac_model, *, unit="phone",
                    art_window=None, ac_window=None, stats=None,
                    art_scalers=None, ac_scalers=None, device="cpu",
                    progress=True, cache=None, batch_size=32, **mfcc_kwargs):
    """
    Latent vectors for every token that yields AT LEAST ONE stream.

    The two streams are encoded INDEPENDENTLY, and `has_art` / `has_ac` record
    which of them each token produced. This matches the original per-corpus
    notebooks, where a token whose MFCCs failed still contributed an
    articulatory distance: requiring both streams silently cost 4,001
    articulatory observations on BEUS phone.

    Entropy and mutual information still need paired observations and take the
    `has_art & has_ac` subset; the distinctiveness stage uses each modality's
    own mask. Rows without a given stream carry NaN in that stream's array, so
    a caller that ignores the masks gets NaN rather than a wrong number.
    """
    rows = token_df.iterrows()
    if progress:
        import tqdm
        rows = tqdm.tqdm(rows, total=len(token_df), desc=f"{spec.name} latents")

    meta = []
    art_idx, art_vecs, art_pending, art_pending_idx = [], [], [], []
    ac_idx, ac_vecs, ac_pending, ac_pending_idx = [], [], [], []

    def flush(which):
        if which == "art" and art_pending:
            art_vecs.append(encode_stream(spec, art_pending, art_model, device, "art"))
            art_idx.extend(art_pending_idx)
            art_pending.clear(); art_pending_idx.clear()
        elif which == "ac" and ac_pending:
            ac_vecs.append(encode_stream(spec, ac_pending, ac_model, device, "ac"))
            ac_idx.extend(ac_pending_idx)
            ac_pending.clear(); ac_pending_idx.clear()

    for idx, row in rows:
        art = art_features(
            spec, row, window=art_window,
            stats=stats.get(row["art_f"]) if stats is not None else None,
            scalers=art_scalers, cache=cache)
        ac = ac_features(row, window=ac_window, scalers=ac_scalers, **mfcc_kwargs)
        if art is None and ac is None:
            continue

        here = len(meta)
        meta.append((idx, row))
        if art is not None:
            art_pending.append(art); art_pending_idx.append(here)
            if len(art_pending) >= batch_size:
                flush("art")
        if ac is not None:
            ac_pending.append(ac); ac_pending_idx.append(here)
            if len(ac_pending) >= batch_size:
                flush("ac")
    flush("art"); flush("ac")

    if not meta:
        raise RuntimeError(
            f"no token of {spec.name} yielded either stream -- check that the "
            "recordings are reachable at the paths in the dataset csv")

    n = len(meta)

    def assemble(vec_batches, indices):
        if not vec_batches:
            return np.full((n, 0), np.nan, dtype=np.float32), np.zeros(n, bool)
        stacked = np.concatenate(vec_batches)
        out = np.full((n, stacked.shape[1]), np.nan, dtype=np.float32)
        out[np.asarray(indices)] = stacked
        mask = np.zeros(n, bool)
        mask[np.asarray(indices)] = True
        return out, mask

    art_arr, has_art = assemble(art_vecs, art_idx)
    ac_arr, has_ac = assemble(ac_vecs, ac_idx)

    return {
        "art": art_arr,
        "ac": ac_arr,
        "has_art": has_art,
        "has_ac": has_ac,
        "labels": np.array([r[unit] for _, r in meta]),
        "subjects": np.array([r["subject"] for _, r in meta]),
        "groups": np.array([r["art_f"] for _, r in meta]),
        "index": np.array([i for i, _ in meta]),
        "start": np.array([float(r["start"]) for _, r in meta]),
        "end": np.array([float(r["end"]) for _, r in meta]),
        "art_f": np.array([r["art_f"] for _, r in meta]),
        "ac_f": np.array([r["ac_f"] for _, r in meta]),
    }


def stream_masks(latents):
    """
    (has_art, has_ac) for a latent file, defaulting to all-True.

    Extractions written before the streams were separated contain only tokens
    that produced both, so treating a missing mask as all-True is correct for
    those files.
    """
    n = len(latents["art"])
    has_art = np.asarray(latents["has_art"]) if "has_art" in latents else np.ones(n, bool)
    has_ac = np.asarray(latents["has_ac"]) if "has_ac" in latents else np.ones(n, bool)
    return has_art, has_ac


def paired(latents):
    """The tokens with both streams -- what entropy and MI are estimated on."""
    has_art, has_ac = stream_masks(latents)
    keep = has_art & has_ac
    return (np.asarray(latents["art"])[keep], np.asarray(latents["ac"])[keep],
            np.asarray(latents["labels"])[keep], np.asarray(latents["groups"])[keep], keep)


def latent_lookup(latents, decimals=6):
    """
    (art_f, start, end) -> row number in a cached latent extraction.

    Token boundaries come from a csv and go back through float formatting, so
    they are rounded before use as keys rather than compared exactly.
    """
    key = lambda f, s, e: (str(f), round(float(s), decimals), round(float(e), decimals))
    return {key(f, s, e): i
            for i, (f, s, e) in enumerate(zip(latents["art_f"],
                                              latents["start"],
                                              latents["end"]))}, key


# --------------------------------------------------------------------------- #
# Conditional entropy by cross-validated MDN
# --------------------------------------------------------------------------- #

def _train_mdn(train_loader, val_loader, x_dim, y_dim, device,
               n_components=5, hidden_dim=256, lr=1e-3, patience=10,
               max_epochs=2000, verbose=False):
    import copy
    import torch
    from utils.models import MDN

    model = MDN(x_dim=x_dim, y_dim=y_dim, n_components=n_components,
                hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val, best_state, stale = float("inf"), copy.deepcopy(model.state_dict()), 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = -model.log_prob(xb, yb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        train_loss = total / len(train_loader.dataset)

        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                total += (-model.log_prob(xb, yb).mean()).item() * len(xb)
        val_loss = total / len(val_loader.dataset)

        if verbose:
            print(f"    epoch {epoch:4d}  train {train_loss:.4f}  val {val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val


def conditional_entropy(X, Y, labels, *, k=5, min_tokens=30, device=None,
                        seed=42, verbose=False, **mdn_kwargs):
    """
    H(Y | X) per label, in nats, by k-fold cross-validated MDN density estimation.

    Both streams are standardised on each fold's TRAINING rows only, and the
    held-out log density is mapped back to the original Y coordinates through
    the scaler's log-Jacobian, so the entropies are comparable across folds and
    across labels.

    Returns (entropies, logp) where `entropies` maps label -> H and `logp` is
    the per-token held-out log density, useful for diagnostics.
    """
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, Dataset, Subset

    device = device or best_device()
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    labels = np.asarray(labels)
    n = len(X)
    if len(Y) != n or len(labels) != n:
        raise ValueError("X, Y and labels must have the same length")

    class _Pairs(Dataset):
        def __init__(self, a, b):
            self.a = torch.from_numpy(a).float()
            self.b = torch.from_numpy(b).float()

        def __len__(self):
            return len(self.a)

        def __getitem__(self, i):
            return self.a[i], self.b[i]

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, k)

    logp = np.zeros(n, dtype=np.float64)

    for f in range(k):
        test_idx = folds[f]
        train_idx = np.hstack([folds[i] for i in range(k) if i != f])
        rng.shuffle(train_idx)
        split = int(0.9 * len(train_idx))
        inner_train, inner_val = train_idx[:split], train_idx[split:]

        sx, sy = StandardScaler().fit(X[inner_train]), StandardScaler().fit(Y[inner_train])
        Xn, Yn = sx.transform(X), sy.transform(Y)

        pairs = _Pairs(Xn, Yn)
        model, best = _train_mdn(
            DataLoader(Subset(pairs, inner_train), batch_size=64, shuffle=True),
            DataLoader(Subset(pairs, inner_val), batch_size=64, shuffle=False),
            x_dim=X.shape[1], y_dim=Y.shape[1], device=device, **mdn_kwargs)
        if verbose:
            print(f"  fold {f}: best val nll {best:.4f}")

        model.eval()
        with torch.no_grad():
            lp = model.log_prob(
                torch.from_numpy(Xn[test_idx]).float().to(device),
                torch.from_numpy(Yn[test_idx]).float().to(device),
            ).cpu().numpy()
        logp[test_idx] = lp - np.log(sy.scale_).sum()

    entropies = {}
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if len(idx) < min_tokens:
            continue
        entropies[label] = float(-logp[idx].mean())

    return entropies, logp


# --------------------------------------------------------------------------- #
# Pairwise distinctiveness
# --------------------------------------------------------------------------- #

def pair_distances(spec, df, art_model, ac_model, *, unit="phone", **kw):
    """
    Deprecated. `runs/compute_distances.py` reads the cached latents instead,
    which is both faster and guaranteed consistent with the entropy and MI
    stages, since all three then see exactly the same vectors.
    """
    raise NotImplementedError(
        "pair_distances has been replaced by runs/compute_distances.py, which "
        "joins the cached latents from runs/extract_latents.py by "
        "(recording, start, end).")
