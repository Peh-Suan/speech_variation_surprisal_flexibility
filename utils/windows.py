"""
How a token's analysis window is derived from its segmentation.

Every token in this project arrives with a [start, end] interval that came out
of a forced alignment of the *acoustics*. Using that same interval for the
articulatory signal encodes an assumption that the gesture and its acoustic
consequence are coextensive in time. Reviewers at LabPhon pushed back on
exactly that: an articulatory gesture is generally already underway before it
becomes audible and is still unwinding after the acoustic segment has ended,
so an acoustically-bounded articulatory token may be systematically clipped at
both edges.

This module turns that assumption into one explicit, configurable object, so
that "acoustic boundaries" and "acoustic boundaries widened by X" are two
settings of a single switch rather than two forks of the pipeline.

Default padding is PROPORTIONAL to the token's own duration:

    pad = amount * (end - start)      on each side

which scales with speech rate: a 60 ms flap gets a smaller absolute pad than a
250 ms diphthong. The trade-off that buys is worth stating in the manuscript:
because the pad is a function of duration, any effect of widening the window is
partly entangled with duration, which is itself a predictor in the downstream
models. `Window(kind="fixed", amount=0.05)` gives the duration-independent
alternative, and every call site already goes through here, so switching costs
one line in a config.

Typical use:

    from utils.windows import WINDOWS
    w = WINDOWS["prop25"]
    s, e = w.apply(row["start"], row["end"], path=row["art_f"])

Windows are clipped to the recording: never below 0, and never past the end of
the file when the duration is knowable. Clipping means the realised pad can be
smaller than the nominal one for tokens near a recording boundary, so
`realised_pad` is available when you want to report the distribution rather
than the nominal setting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

KINDS = ("none", "proportional", "fixed")

# Duration lookups hit the filesystem, and the same recording supplies hundreds
# of tokens, so they are cached for the life of the process.
_DURATION_CACHE: dict = {}


def media_duration(path):
    """
    Length of a media file in seconds, or None when it cannot be determined
    cheaply. None is not an error: the feature extractors all truncate at the
    end of the file, so an unclipped right edge is safe — it just means the
    realised pad is unknown until extraction.
    """
    if path is None:
        return None
    from utils.mounts import local_path
    path = local_path(path)
    if path in _DURATION_CACHE:
        return _DURATION_CACHE[path]

    duration = None
    try:
        ext = os.path.splitext(path)[1].lower()

        if ext in (".avi", ".mp4", ".mov", ".mkv"):
            import cv2
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if np.isfinite(fps) and fps > 0 and np.isfinite(n) and n > 0:
                    duration = float(n) / float(fps)
            cap.release()

        elif ext in (".wav", ".flac", ".ogg", ".aiff"):
            import soundfile as sf
            info = sf.info(path)
            duration = float(info.frames) / float(info.samplerate)

        elif ext == ".npy":
            # BE ultrasound stores frames and audio as .npy siblings.
            if path.endswith("_audio.npy"):
                sr_path = path.replace("_audio.npy", "_sr.npy")
                if os.path.exists(sr_path):
                    sr = int(np.asarray(np.load(sr_path)).ravel()[0])
                    n = np.load(path, mmap_mode="r").shape[0]
                    duration = float(n) / float(sr)
            elif path.endswith("_frames.npy"):
                fps_path = path.replace("_frames.npy", "_fps.npy")
                if os.path.exists(fps_path):
                    fps = float(np.asarray(np.load(fps_path)).ravel()[0])
                    n = np.load(path, mmap_mode="r").shape[0]
                    if fps > 0:
                        duration = float(n) / fps
        # .mat (EMA, annotated MRI) is deliberately not handled: opening one to
        # read its length costs as much as extracting from it, and those
        # extractors already truncate at the end of the record.
    except Exception:
        duration = None

    _DURATION_CACHE[path] = duration
    return duration


@dataclass(frozen=True)
class Window:
    """
    A rule for turning a segmentation interval into an analysis window.

    kind
        "none"          the interval is used as is (the LabPhon analysis)
        "proportional"  pad = amount * duration on each side
        "fixed"         pad = amount seconds on each side
    amount
        Fraction (proportional) or seconds (fixed).
    min_pad, max_pad
        Floor and ceiling in seconds, applied after `amount`. A floor is worth
        setting for the video corpora: at ~23 fps a 40 ms token padded by 25%
        gains 10 ms per side, which is less than half a frame and therefore
        usually no extra frame at all. A ceiling keeps a long token from
        swallowing a whole neighbouring word.
    """

    name: str = "base"
    kind: str = "none"
    amount: float = 0.0
    min_pad: float = 0.0
    max_pad: Optional[float] = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.amount < 0 or self.min_pad < 0:
            raise ValueError("amount and min_pad must be non-negative")
        if self.max_pad is not None and self.max_pad < self.min_pad:
            raise ValueError("max_pad must not be below min_pad")

    # ------------------------------------------------------------------ #

    @property
    def is_identity(self) -> bool:
        """True when this window leaves every interval untouched."""
        return self.kind == "none" or (self.amount == 0.0 and self.min_pad == 0.0)

    def pad_for(self, start, end) -> float:
        """Nominal pad in seconds for one side, before clipping."""
        if self.kind == "none":
            return 0.0
        duration = max(0.0, float(end) - float(start))
        pad = self.amount * duration if self.kind == "proportional" else float(self.amount)
        pad = max(pad, self.min_pad)
        if self.max_pad is not None:
            pad = min(pad, self.max_pad)
        return float(pad)

    def apply(self, start, end, path=None, duration=None):
        """
        Returns the padded (start, end), clipped to the recording.

        `path` is used only to look up the recording length; pass `duration`
        directly if you already have it. If the padded window would be empty
        or inverted — which should not happen, but a corrupt duration could do
        it — the original interval is returned unchanged rather than silently
        producing a degenerate token.
        """
        start, end = float(start), float(end)
        pad = self.pad_for(start, end)
        if pad <= 0.0:
            return start, end

        s = max(0.0, start - pad)
        e = end + pad

        if duration is None and path is not None:
            duration = media_duration(path)
        if duration is not None and duration > 0:
            e = min(e, float(duration))
            s = min(s, float(duration))

        if not (e > s):
            return start, end
        return s, e

    def realised_pad(self, start, end, path=None, duration=None):
        """(left, right) pad actually obtained after clipping, in seconds."""
        s, e = self.apply(start, end, path=path, duration=duration)
        return float(start) - s, e - float(end)

    def with_min_pad(self, min_pad):
        """
        A copy of this window whose floor is at least `min_pad` seconds.

        Used to raise the floor to one articulatory frame, which is corpus
        specific: at USC-TIMIT's 23.18 fps a frame is 43 ms, so a 25% pad on a
        60 ms token (15 ms) buys nothing at all, while at USC-75's 83 fps the
        same 15 ms is more than a frame. See CorpusSpec.window().
        """
        from dataclasses import replace
        if self.kind == "none" or min_pad <= self.min_pad:
            return self
        max_pad = self.max_pad
        if max_pad is not None and max_pad < min_pad:
            max_pad = min_pad
        return replace(self, min_pad=float(min_pad), max_pad=max_pad)

    def describe(self) -> str:
        if self.kind == "none":
            return "acoustic boundaries, unpadded"
        unit = "x duration" if self.kind == "proportional" else "s"
        out = f"{self.kind}: +/- {self.amount:g}{unit}"
        if self.min_pad:
            out += f", floor {self.min_pad * 1000:g} ms"
        if self.max_pad is not None:
            out += f", ceiling {self.max_pad * 1000:g} ms"
        return out


# --------------------------------------------------------------------------- #
# The windows the manuscript actually reports. Add here rather than inline, so
# that a variant name in a filename always maps back to a definition.
# --------------------------------------------------------------------------- #

WINDOWS = {
    # The LabPhon analysis: acoustic boundaries for both modalities.
    "base": Window(name="base", kind="none"),

    # The reviewers' alternative: the articulatory window widened by 25% of the
    # token's duration on each side.
    #
    # The floor matters more than it looks. USC-TIMIT MRI runs at 23.18 fps, so
    # one frame is 43 ms, while the median phone token is 60 ms: a bare 25% pad
    # is 15 ms per side and leaves roughly two thirds of tokens with exactly the
    # frames they already had. Pass this window through CorpusSpec.window(),
    # which raises `min_pad` to one articulatory frame for the corpus in hand,
    # so the rule reads "25% of the token's duration, but never less than one
    # frame". The ceiling stops a long vowel absorbing a neighbouring syllable.
    "prop25": Window(name="prop25", kind="proportional", amount=0.25,
                     min_pad=0.0, max_pad=0.100),

    # Robustness neighbours of prop25: same mechanism, different strength.
    "prop10": Window(name="prop10", kind="proportional", amount=0.10,
                     min_pad=0.0, max_pad=0.100),
    "prop50": Window(name="prop50", kind="proportional", amount=0.50,
                     min_pad=0.0, max_pad=0.150),

    # One articulatory frame per side and nothing more -- the most conservative
    # widening that can change anything at all for a video corpus.
    "frame1": Window(name="frame1", kind="proportional", amount=0.0,
                     min_pad=0.0, max_pad=0.150),

    # Duration-independent fallback, if the duration confound is challenged.
    "fixed50": Window(name="fixed50", kind="fixed", amount=0.050),
}


def get_window(name_or_window):
    """Accepts a Window, a key of WINDOWS, or None (meaning 'base')."""
    if name_or_window is None:
        return WINDOWS["base"]
    if isinstance(name_or_window, Window):
        return name_or_window
    try:
        return WINDOWS[name_or_window]
    except KeyError:
        raise KeyError(
            f"unknown window {name_or_window!r}; known: {sorted(WINDOWS)}"
        ) from None
