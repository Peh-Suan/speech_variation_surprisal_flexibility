"""
Decode each recording once, not once per token.

The extractors take (file, start, end) and are called per token, so a training
epoch opens the same recording hundreds of times and seeks into it each time.
On compressed video a seek is not free: cv2 has to decode forward from the
preceding keyframe, so the cost per token is roughly the keyframe interval
rather than the token's own length. Measured on USC-TIMIT MRI over a mounted
external drive, per-token seeking runs at ~19 ms/token against ~1.5 ms/token
for one sequential pass -- about 13x, and three epochs pay it three times.

The fix is what you would do by hand: read a recording straight through once,
then cut every token out of the frames in memory.

Two things follow from that, and both are improvements rather than side effects:

  * `cap.set(CAP_PROP_POS_FRAMES, n)` is not exact on all codecs -- the warning
    is already in your notebooks. Reading sequentially and counting frames is
    exact, so cached features are, if anything, better aligned to the audio
    than seeking was. They are therefore NOT guaranteed bit-identical to an
    uncached run; see `compare_against_direct` below, which measures it.
  * For EMA and the tracked contours the win is not seeking at all but
    `loadmat`: those getters reload and reparse the whole .mat for every token.

Layout: one float16 memmap per (unit, corpus, window) holding every token's
frames end to end, plus a parquet index giving each token's offset and length.
Random access during training is then a slice of a memmap, which on a machine
with enough RAM is served from the page cache.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from utils.windows import get_window


# --------------------------------------------------------------------------- #
# Per-recording facts the slicing needs
# --------------------------------------------------------------------------- #

def recording_fps(spec, art_f):
    """
    True frame rate of one recording.

    Not `spec.art_fps`: that is the nominal rate, used to size a window's
    one-frame floor. Slicing needs the real one, and BE ultrasound varies
    between 59.96 and 60.20 across recordings -- enough to walk a token off by
    a frame over a long file.
    """
    from utils.mounts import local_path
    path = local_path(art_f)
    if path.endswith("_frames.npy"):
        fps_path = path.replace("_frames.npy", "_fps.npy")
        return float(np.asarray(np.load(fps_path)).ravel()[0])

    if path.endswith(".mat"):
        if spec.name == "aeema":
            from utils.articulation_processing import process_ema_mat
            data = process_ema_mat(path)
            return float(next(iter(data.values()))["fps"])
        from utils.articulation_processing import mat_to_mp4
        path = mat_to_mp4(path)

    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"could not read a frame rate from {art_f}")
    return float(fps)


def frame_span(start, end, fps):
    """
    The frame indices a token covers, matching the extractors exactly:
    round(t * fps) at both edges, end inclusive.
    """
    first = max(0, int(round(float(start) * fps)))
    last = int(round(float(end) * fps))
    return first, max(last, first)


# --------------------------------------------------------------------------- #
# Reading a whole recording
# --------------------------------------------------------------------------- #

def read_recording(spec, art_f, upto, stats=None, margin=1.0):
    """
    Features for one recording from time 0, as a single sequential read.

    `upto` is the last token end in this recording; a margin is added and the
    extractors truncate at the file's end, so the exact duration is never
    needed. Returns (features, fps) with features indexed from frame 0, or
    (None, fps) when the recording yields nothing.
    """
    fps = recording_fps(spec, art_f)
    features = spec.art_features(art_f, 0.0, float(upto) + margin, stats=stats)
    return features, fps


def slice_token(features, fps, start, end):
    first, last = frame_span(start, end, fps)
    if features is None or first >= len(features):
        return None
    out = features[first:last + 1]
    return out if len(out) else None


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #

class FeatureCache:
    """
    Token features on disk, addressed by (recording, start, end).

    `path` is the memmap; the index lives beside it as `<path>.index.parquet`.
    """

    def __init__(self, path):
        self.path = str(path)
        self.index_path = self.path + ".index.csv"
        self.meta_path = self.path + ".meta.json"
        self._data = None
        self._index = None
        self._meta = None

    # -- reading ------------------------------------------------------- #

    def exists(self):
        return all(os.path.exists(p) for p in
                   (self.path, self.index_path, self.meta_path))

    def open(self):
        if self._data is not None:
            return self
        if not self.exists():
            raise FileNotFoundError(
                f"no feature cache at {self.path} -- build it with "
                "runs/build_cache.py")
        import json
        with open(self.meta_path) as f:
            self._meta = json.load(f)
        index = pd.read_csv(self.index_path)
        shape = tuple(self._meta["frame_shape"])
        total = int(self._meta["total_frames"])
        self._data = np.memmap(self.path, dtype=np.dtype(self._meta.get("dtype", "float16")),
                               mode="r", shape=(total,) + shape)
        index["key"] = list(zip(index["art_f"], index["start"].round(6),
                                index["end"].round(6)))
        self._index = dict(zip(index["key"], zip(index["offset"], index["length"])))
        return self

    def get(self, row):
        """Features for one token as float32, or None if it is not cached."""
        self.open()
        key = (str(row["art_f"]), round(float(row["start"]), 6),
               round(float(row["end"]), 6))
        found = self._index.get(key)
        if found is None:
            return None
        offset, length = found
        if length <= 0:
            return None
        return np.asarray(self._data[offset:offset + length], dtype=np.float32)

    def __len__(self):
        self.open()
        return len(self._index)

    # -- building ------------------------------------------------------ #

    @classmethod
    def build(cls, spec, token_df, path, window=None, stats=None,
              progress=True, dtype=None):
        """
        One sequential pass per recording, writing every token's frames.

        Tokens are grouped by recording and the recording is decoded once from
        frame 0; each token is then a slice. Windowing is applied here, so a
        cache belongs to one (unit, corpus, window) -- which is why its path
        carries the same tag as everything else.
        """
        window = get_window(window)
        dtype = np.dtype(dtype or getattr(spec, "cache_dtype", "float16"))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        groups = list(token_df.groupby("art_f", sort=True))
        iterator = groups
        if progress:
            import tqdm
            iterator = tqdm.tqdm(groups, desc=f"caching {spec.name}")

        frame_shape = None
        expected = spec.expected_frame_shape      # None for non-video corpora
        offset = 0
        rows = []
        failures = []
        wrong_shape = {}
        tmp = str(path) + ".building"

        with open(tmp, "wb") as sink:
            for art_f, part in iterator:
                windows = [window.apply(s, e, path=art_f)
                           for s, e in zip(part["start"], part["end"])]
                upto = max(e for _, e in windows) if windows else 0.0
                try:
                    features, fps = read_recording(
                        spec, art_f, upto,
                        stats=stats.get(art_f) if stats is not None else None)
                except Exception as exc:
                    failures.append((art_f, f"{type(exc).__name__}: {exc}"))
                    features, fps = None, None

                for (row_start, row_end), (ws, we) in zip(
                        zip(part["start"], part["end"]), windows):
                    chunk = (None if features is None
                             else slice_token(features, fps, ws, we))
                    if chunk is None:
                        rows.append((str(art_f), float(row_start), float(row_end), 0, 0))
                        continue
                    chunk = np.ascontiguousarray(chunk, dtype=dtype)
                    # A recording whose frames are not the shape the encoder was
                    # built for is skipped, matching the original notebooks'
                    #   art.shape[-2:] != input_hw  ->  drop
                    # rule. Checked against the model rather than against the
                    # first recording seen, so one odd recording arriving first
                    # cannot define the shape and evict all the good ones.
                    if (expected is not None
                            and chunk.shape[-len(expected):] != expected):
                        wrong_shape[str(art_f)] = chunk.shape[-len(expected):]
                        rows.append((str(art_f), float(row_start), float(row_end), 0, 0))
                        continue
                    if frame_shape is None:
                        frame_shape = chunk.shape[1:]
                    elif chunk.shape[1:] != frame_shape:
                        rows.append((str(art_f), float(row_start), float(row_end), 0, 0))
                        continue
                    sink.write(chunk.tobytes())
                    rows.append((str(art_f), float(row_start), float(row_end),
                                 offset, len(chunk)))
                    offset += len(chunk)

        if offset == 0:
            os.remove(tmp)
            detail = ""
            if failures:
                examples = "\n    ".join(f"{p}: {why}" for p, why in failures[:3])
                detail = (f"\n  {len(failures)} of {len(groups)} recordings could "
                          f"not be read, for example:\n    {examples}")
            raise RuntimeError(
                "the feature cache would be empty -- no token produced any "
                "features." + detail +
                "\n  If the recordings live somewhere other than the path in "
                "the dataset csv, set SVSF_MOUNTS (see utils/mounts.py).")
        os.replace(tmp, path)

        import json
        index = pd.DataFrame(rows, columns=["art_f", "start", "end", "offset", "length"])
        index.to_csv(str(path) + ".index.csv", index=False)
        with open(str(path) + ".meta.json", "w") as f:
            json.dump({"frame_shape": list(frame_shape or ()),
                       "total_frames": int(offset),
                       "corpus": spec.name,
                       "window": window.name,
                       "dtype": dtype.name}, f, indent=1)

        cached = int((index["length"] > 0).sum())
        return {"tokens": len(index), "cached": cached,
                "dropped": len(index) - cached,
                "unreadable_recordings": len(failures),
                "wrong_shape_recordings": wrong_shape,
                "frames": int(offset), "frame_shape": tuple(frame_shape or ()),
                "dtype": dtype.name,
                "bytes": offset * int(np.prod(frame_shape or (1,))) * dtype.itemsize}


def compare_against_direct(spec, cache, token_df, stats=None, window=None,
                           n=25, seed=0):
    """
    How far cached features sit from what per-token seeking returns.

    Not expected to be zero: seeking is approximate on compressed video and the
    sequential pass is the accurate one. This reports how often and how much
    they disagree, so the difference is a measurement rather than a surprise.
    """
    window = get_window(window)
    sample = token_df.sample(n=min(n, len(token_df)), random_state=seed)
    same = differ = missing = 0
    deltas = []
    for _, row in sample.iterrows():
        cached = cache.get(row)
        s, e = window.apply(row["start"], row["end"], path=row["art_f"])
        direct = spec.art_features(
            row["art_f"], s, e,
            stats=stats.get(row["art_f"]) if stats is not None else None)
        if cached is None or direct is None:
            missing += 1
            continue
        if cached.shape != direct.shape:
            differ += 1
            continue
        d = float(np.abs(cached - direct).max())
        deltas.append(d)
        if d < 1e-2:
            same += 1
        else:
            differ += 1
    return {"checked": len(sample), "identical": same, "differing": differ,
            "unavailable": missing,
            "max_abs_diff": float(max(deltas)) if deltas else None,
            "median_abs_diff": float(np.median(deltas)) if deltas else None}
