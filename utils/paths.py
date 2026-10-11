"""
Where each stage's outputs go.

An analysis in this project is identified by three things:

    unit    "phone" or "syllable"   -- which tree it lives in
    corpus  "aemri", "beus", ...    -- which recordings
    window  "base", "prop25", ...   -- how token boundaries were derived

The unit picks the directory tree (phone_based/ or syllable_based/); the corpus
and window together give the file tag. `base` produces exactly the filenames
the project already uses, so nothing that exists today has to move:

    phone_based/.../distances_aemri.csv              window="base"
    phone_based/.../distances_aemri__prop25.csv      window="prop25"

Two things are deliberately keyed by corpus rather than by tag, because
widening the ARTICULATORY window leaves them unchanged and retraining them
would be wasted compute:

    * the acoustic encoder and its per-speaker MFCC scalers
    * per-recording video intensity statistics (computed over whole recordings)

Surprisal and positional entropy are lexical, so they too are window-independent
(they do depend on the unit, and live in the unit's own tree).
"""

from __future__ import annotations

import os

from utils.windows import get_window

UNIT_DIRS = {"phone": "phone_based", "syllable": "syllable_based"}


def repo_root(start=None):
    """
    Walk up from `start` (default: cwd) until the directory containing utils/
    and phone_based/ is found. Lets a notebook sit at any depth without a
    hand-counted '../../..'.
    """
    d = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.isdir(os.path.join(d, "utils")) and \
           os.path.isdir(os.path.join(d, "phone_based")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(
                "could not locate the repository root above "
                f"{os.path.abspath(start or os.getcwd())}"
            )
        d = parent


class RunPaths:
    """Every path one (unit, corpus, window) analysis reads or writes."""

    def __init__(self, unit, corpus, window="base", root=None):
        if unit not in UNIT_DIRS:
            raise ValueError(f"unit must be one of {sorted(UNIT_DIRS)}, got {unit!r}")
        self.unit = unit
        self.corpus = corpus
        self.window = get_window(window)
        self.root = root or repo_root()
        self.tree = os.path.join(self.root, UNIT_DIRS[unit])

    # ---------------------------------------------------------------- #

    @property
    def tag(self):
        """Corpus plus window, the stem of every window-dependent file."""
        w = self.window.name
        return self.corpus if w == "base" else f"{self.corpus}__{w}"

    def _p(self, *parts):
        return os.path.join(self.tree, *parts)

    # -- window-independent ------------------------------------------ #

    @property
    def dataset(self):
        return self._p("data_processing", "dataset_construction", "datasets",
                       f"processed_{self.corpus}.csv")

    @property
    def surprisal(self):
        return self._p("data_processing", "surprisal_calculation",
                       f"surprisal_{self.corpus}.csv")

    @property
    def ac_scalers(self):
        return self._p("encoder_training", "scalers", f"{self.corpus}_ac.pkl")

    @property
    def video_stats(self):
        return self._p("encoder_training", "scalers",
                       f"{self.corpus}_video_stats.pkl")

    @property
    def ac_ckpt(self):
        return self._p("encoder_training", "ckpts", self.corpus, "ac_model_ckpt.pth")

    # -- window-dependent -------------------------------------------- #

    @property
    def art_scalers(self):
        return self._p("encoder_training", "scalers", f"{self.tag}_art.pkl")

    @property
    def ckpt_dir(self):
        return self._p("encoder_training", "ckpts", self.tag)

    @property
    def art_ckpt(self):
        return os.path.join(self.ckpt_dir, "art_model_ckpt.pth")

    @property
    def art_log(self):
        return os.path.join(self.ckpt_dir, "art_model_training_log.csv")

    @property
    def ac_log(self):
        return self._p("encoder_training", "ckpts", self.corpus,
                       "ac_model_training_log.csv")

    @property
    def feature_cache(self):
        """
        Decoded articulatory features for every token, one file per
        (unit, corpus, window). Large -- tens of GB for the ultrasound corpora
        -- so it lives under .cache/, which is gitignored, and SVSF_CACHE can
        move it to another disk.
        """
        base = os.environ.get("SVSF_CACHE") or os.path.join(self.root, ".cache")
        return os.path.join(base, "features", self.unit, f"{self.tag}.f16")

    @property
    def latents(self):
        """
        Cached encoder output for every token, shared by the three stages that
        need it. Keyed by tag: widening the articulatory window changes it.
        """
        return self._p("flexibility_estimation", "latents", f"{self.tag}.npz")

    @property
    def distances(self):
        return self._p("data_processing", "distinctiveness_calculation",
                       f"distances_{self.tag}.csv")

    def entropy(self, direction):
        assert direction in ("art_given_ac", "ac_given_art")
        return self._p("flexibility_estimation", "entropy_estimation", "entropies",
                       f"{self.tag}_{direction}.pickle")

    @property
    def mi(self):
        return self._p("flexibility_estimation", "mutual_information_estimation",
                       "res", f"{self.tag}.csv")

    @property
    def combined(self):
        return self._p("data_processing", "combine_data", f"results_{self.tag}.csv")

    # ---------------------------------------------------------------- #

    def ensure_dirs(self):
        """Create every output directory this analysis writes into."""
        for p in (self.surprisal, self.ac_scalers, self.art_scalers, self.art_ckpt,
                  self.ac_ckpt, self.latents, self.distances,
                  self.entropy("art_given_ac"), self.mi, self.combined):
            os.makedirs(os.path.dirname(p), exist_ok=True)
        return self

    def __repr__(self):
        return (f"RunPaths(unit={self.unit!r}, corpus={self.corpus!r}, "
                f"window={self.window.name!r}, tag={self.tag!r})")
