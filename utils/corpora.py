"""
One description of each corpus, in one place.

Every stage of this project (dataset construction, encoder training,
distinctiveness, entropy, mutual information, combination) previously carried
its own copy of the same six facts about each corpus: where the recordings
live, which articulatory extractor to call, what shape that extractor returns,
which autoencoder consumes it, and how to recover the speaker from a filename.
Six corpora times five stages is thirty copies, and they had already drifted
(the BE ultrasound branch, for instance, loads the FRENCH pronunciation
dictionary while passing lang="be" -- see the note on BEUS below).

Adding two more analyses on top of that -- syllable units and widened
articulatory windows -- would have meant sixty. So the per-corpus facts live
here and the stages take a CorpusSpec.

    from utils.corpora import CORPORA
    spec = CORPORA["aemri"]
    art = spec.art_features(row["art_f"], start, end, stats=stats)

Nothing heavy is imported at module load: the extractor and model references
are resolved on first use, so `import utils.corpora` does not pull in torch,
cv2 or librosa.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

# --------------------------------------------------------------------------- #
# Speaker identification
# --------------------------------------------------------------------------- #

def _subject_field(idx):
    """Speaker = the idx-th underscore-separated field of the art filename."""
    def extract(art_f):
        return os.path.basename(str(art_f)).split("_")[idx]
    return extract


# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CorpusSpec:
    name: str
    lang: str                       # "ae" | "be" | "fr"
    dataset_path: str               # where the recordings live, for reference
    modality: str                   # "video" | "sequence" -- which autoencoder
    getter: str                     # function name in utils.articulation_processing
    subject_from_art_f: Callable

    # Articulatory model
    art_model_kwargs: dict = field(default_factory=dict)

    # Post-extraction reshape, for the extractors that return flat coordinates.
    art_reshape: Optional[Tuple[int, ...]] = None

    # Per-recording intensity statistics: "mri", "beus", or None when the
    # extractor needs no stats argument.
    rec_stats: Optional[str] = None

    # Per-speaker feature scaler for the articulatory stream. The video corpora
    # are normalised by recording statistics instead, so this is None for them.
    art_scaler_mode: Optional[str] = None      # e.g. "isotropic"

    # Lexical resources, relative to the repository root.
    pronunciation_counts: str = ""
    pronunciation_dict: Optional[str] = None
    preferred_ipas: Optional[str] = None

    # Nominal articulatory sampling rate, Hz. Measured from the recordings;
    # used to floor a proportional window at one frame. None means unknown, in
    # which case the window's own floor stands.
    art_fps: Optional[float] = None

    # Storage precision for the feature cache. The video and ultrasound corpora
    # come off 8-bit sensors, so after normalisation float16's error (~2e-03)
    # sits an order of magnitude below the data's own quantisation step
    # (~2.2e-02) and nothing real is lost. EMA and the tracked contours are
    # continuous float64 coordinates with no such floor, so they are cached at
    # full precision -- it costs 22 MB and 194 MB respectively, which is nothing.
    cache_dtype: str = "float16"

    # Whether a feature cache pays for itself. True wherever the extractor
    # seeks into compressed video or reparses a .mat per token. False for BE
    # ultrasound: `_beus_streams` memmaps the .npy frames and slices before
    # converting, so random access is already cheap, and the cache would be
    # 34 GB at phone level (56 GB widened) for almost no gain.
    cache_recommended: bool = True

    # Phones whose articulation this instrument cannot see, so that a pair
    # containing one is flagged `is_limited` and excluded by the analysis
    # notebooks. Empty where the corpus images the whole vocal tract (MRI) --
    # only EMA and ultrasound have blind spots. Lists copied verbatim from
    # combine_data/aeema.ipynb and combine_data/beus.ipynb; see the note on
    # beus below before trusting that one.
    limited_phones: tuple = ()

    extra_getter_kwargs: dict = field(default_factory=dict)
    notes: str = ""

    # ------------------------------------------------------------------ #
    # Lazy resolution
    # ------------------------------------------------------------------ #

    def art_getter(self):
        from utils import articulation_processing
        return getattr(articulation_processing, self.getter)

    def art_features(self, art_f, start, end, stats=None):
        """
        Raw articulatory features for one window, reshaped as the model expects,
        or None when the extractor could not produce them.

        `start`/`end` are already-windowed times: widening belongs to
        utils.windows, not here, so that every stage widens identically.
        """
        from utils.mounts import local_path
        art_f = local_path(art_f)
        fn = self.art_getter()
        kwargs = dict(self.extra_getter_kwargs)
        if self.rec_stats is not None:
            kwargs["stats"] = stats
        out = fn(art_f, start, end, **kwargs)
        if out is None:
            return None
        if self.art_reshape is not None:
            out = out.reshape(out.shape[0], *self.art_reshape)
        return out

    @property
    def expected_frame_shape(self):
        """
        The spatial shape the articulatory encoder was built for, or None for
        non-video corpora. FR-MRI is the reason this exists: 160 of its 161
        recordings are 136x136 and one (speaker P4) is 600x450, so the frame
        shape is a property of the recording, not of the corpus. The original
        phone notebooks dropped the odd one with
            art.shape[-2:] != (136, 136)  ->  skip
        at both training and latent extraction; this reproduces that rule for
        every corpus from the model's own input_hw.
        """
        hw = self.art_model_kwargs.get("input_hw")
        return tuple(hw) if hw is not None else None

    def build_art_model(self, device="cpu"):
        from utils.models import SequenceAutoencoder, VideoAutoencoder
        cls = VideoAutoencoder if self.modality == "video" else SequenceAutoencoder
        return cls(**self.art_model_kwargs).to(device)

    def build_ac_model(self, device="cpu", feature_shape=13, latent_dim=64):
        from utils.models import SequenceAutoencoder
        return SequenceAutoencoder(feature_shape=feature_shape,
                                   latent_dim=latent_dim).to(device)

    def collate(self):
        from utils.train import collate_sequence, collate_video
        return collate_video if self.modality == "video" else collate_sequence

    @property
    def time_dim(self):
        """Axis along which the articulatory tensor runs in time, after collation."""
        return 2 if self.modality == "video" else 1

    def is_limited(self, unit1, unit2, unit="phone"):
        """
        True when either member of a pair contains a phone this instrument
        cannot see. MRI corpora have no such phones, so this is always False
        for them.

        For syllables the test is containment: a syllable is limited if any
        phone in it is. That is an extension of a rule written for phones, so
        it is a judgement rather than something copied from the notebooks.
        """
        if not self.limited_phones:
            return False
        limited = set(self.limited_phones)
        parts = (lambda u: str(u).split("_")) if unit == "syllable" else (lambda u: [str(u)])
        return any(p in limited for p in parts(unit1) + parts(unit2))

    def window(self, name_or_window):
        """
        A window adapted to this corpus's articulatory frame rate.

        A proportional pad is meaningless below one frame: it asks for extra
        signal that does not exist. This raises the floor to one frame period,
        so "prop25" reads as "25% of the token's duration on each side, but
        never less than one articulatory frame". For EMA, which samples far
        faster than any pad, the floor never binds.
        """
        from utils.windows import get_window
        w = get_window(name_or_window)
        if self.art_fps and self.art_fps > 0:
            w = w.with_min_pad(1.0 / float(self.art_fps))
        return w

    def resolve(self, root, attr):
        """Absolute path of a lexical resource, given the repository root."""
        rel = getattr(self, attr)
        return None if rel is None else os.path.join(root, rel)


# --------------------------------------------------------------------------- #
# The six corpora
# --------------------------------------------------------------------------- #

_PC = "phone_based/pronunciation_counts"

CORPORA = {

    "aemri": CorpusSpec(
        name="aemri",
        lang="ae",
        dataset_path="/Volumes/Transcend/USC-TIMIT/MRI/",
        modality="video",
        getter="get_aemri_articulatory_features",
        subject_from_art_f=_subject_field(2),
        art_model_kwargs=dict(input_hw=(68, 68), latent_dim=64, input_channels=1,
                              output_activation=None, stn_mode="shared",
                              stn_transform="rigid"),
        rec_stats="mri",
        pronunciation_counts=f"{_PC}/pronunciation_counts_ae.pkl",
        preferred_ipas=f"{_PC}/preferred_ipas_ae.pkl",
        art_fps=23.18,   # measured from the USC-TIMIT .avi files
        notes="USC-TIMIT real-time MRI, ~23 fps.",
    ),

    "aemri75": CorpusSpec(
        name="aemri75",
        lang="ae",
        dataset_path="/Volumes/Transcend/USC-75/USC-75SpeakerAnnot16/",
        modality="video",
        getter="get_aemri75_articulatory_features",
        subject_from_art_f=_subject_field(0),
        art_model_kwargs=dict(input_hw=(84, 84), latent_dim=64, input_channels=1,
                              output_activation=None, stn_mode="shared",
                              stn_transform="rigid"),
        rec_stats="mri",
        pronunciation_counts=f"{_PC}/pronunciation_counts_ae.pkl",
        preferred_ipas=f"{_PC}/preferred_ipas_ae.pkl",
        art_fps=83.278,   # measured from the USC-75 2drt .mp4 files
        notes="USC 75-speaker MRI, raw video.",
    ),

    "aemri75anno": CorpusSpec(
        name="aemri75anno",
        lang="ae",
        dataset_path="/Volumes/Transcend/USC-75/USC-75SpeakerAnnot16/",
        modality="sequence",
        getter="get_aemri75anno_articulatory_features",
        subject_from_art_f=_subject_field(0),
        art_model_kwargs=dict(feature_shape=(174, 2), latent_dim=64),
        art_reshape=(174, 2),
        art_scaler_mode="isotropic",
        pronunciation_counts=f"{_PC}/pronunciation_counts_ae.pkl",
        preferred_ipas=f"{_PC}/preferred_ipas_ae.pkl",
        art_fps=83.278,   # contours tracked from the same 2drt video
        cache_dtype="float32",   # continuous coordinates, not 8-bit pixels
        notes="USC 75-speaker MRI, tracked contours (87 points, x/y).",
    ),

    "aeema": CorpusSpec(
        name="aeema",
        lang="ae",
        dataset_path="/Volumes/Transcend/USC-TIMIT/EMA/",
        modality="sequence",
        getter="get_aeema_articulatory_features",
        subject_from_art_f=_subject_field(2),
        art_model_kwargs=dict(feature_shape=(6, 3), latent_dim=64),
        art_reshape=(6, 3),
        art_scaler_mode="isotropic",
        pronunciation_counts=f"{_PC}/pronunciation_counts_ae.pkl",
        preferred_ipas=f"{_PC}/preferred_ipas_ae.pkl",
        art_fps=100.08,  # measured from the .mat records; far faster than any pad
        cache_dtype="float32",   # continuous coordinates, not 8-bit pixels
        limited_phones=("h", "m", "n", "\u014b", "\u0279", "\u025d", "\u025a", "a\u026a", "a\u028a", "\u00e6", "\u0251"),   # glottal (no sensor), nasals (no velum sensor), rhotics (no pharyngeal sensor), low vowels
        notes="USC-TIMIT EMA: UL, LL, JAW, TD, TB, TT in x/y/z.",
    ),

    "frmri": CorpusSpec(
        name="frmri",
        lang="fr",
        dataset_path="/Volumes/Transcend/FR-MRI/",
        modality="video",
        getter="get_frmri_articulatory_features",
        subject_from_art_f=_subject_field(1),
        art_model_kwargs=dict(input_hw=(136, 136), latent_dim=64, input_channels=1,
                              output_activation=None, stn_mode="shared",
                              stn_transform="rigid"),
        rec_stats="mri",
        pronunciation_counts=f"{_PC}/pronunciation_counts_fr.pkl",
        pronunciation_dict=f"{_PC}/pronunciation_dict_fr.pkl",
        art_fps=50.0,   # measured from the FR-MRI .avi files
        notes="French real-time MRI.",
    ),

    "beus": CorpusSpec(
        name="beus",
        lang="be",
        dataset_path="/Volumes/Seagate Backup Plus Drive/ultrasound_llm/dataset/",
        modality="video",
        getter="get_beus_articulatory_features",
        # [0], not [2]: "00me_035_xaud_frames.npy" -> "00me". The surprisal
        # notebook used [2] ("xaud", the recording type), but that is the one
        # stage where `subject` is carried and never used, so it went unnoticed;
        # the encoder, distinctiveness and MI notebooks all use [0], and the 41
        # keys in beus_ac.pkl confirm it.
        subject_from_art_f=_subject_field(0),
        art_model_kwargs=dict(input_hw=(56, 112), latent_dim=64, input_channels=2,
                              output_activation=None, stn_mode="separate",
                              stn_transform="rigid"),
        rec_stats="beus",
        extra_getter_kwargs=dict(spatial_size=112),
        art_fps=60.10,   # measured from the *_fps.npy siblings (min 59.96, max 60.20)
        pronunciation_counts=f"{_PC}/pronunciation_counts_be.pkl",
        pronunciation_dict=f"{_PC}/pronunciation_dict_be.pkl",
        cache_recommended=False,
        # Ultrasound blind spots. The first twelve are verbatim from the
        # combine_data/beus.ipynb notebook; the three rhotics at the end were
        # added on 2026-09-20 because the notebook's list named 'r', which does
        # not occur in this corpus -- the rhotics BEUS actually labels are
        # \u0279, \u0259\u0279 and \u025c\u0279, and they were going unflagged. 'r' and '\u00e6' are kept
        # even though neither occurs, so the list still reads as the notebook's
        # plus an addition rather than a rewrite.
        limited_phones=("r", "m", "n", "\u014b", "\u0251", "\u0252", "\u0250", "\u00e6", "a", "aw", "aj", "\u0259",
                        "\u0279", "\u0259\u0279", "\u025c\u0279"),
        notes=(
            "BE ultrasound + lip video, two channels. NOTE: the existing "
            "dataset_construction notebooks for this corpus load "
            "pronunciation_dict_fr.pkl while passing lang='be'. That dictionary "
            "stores English entries as '(en)...(fr)', and normalize_ipa strips "
            "the brackets rather than the tags, so those words reach alignment "
            "with stray phones attached; 527 of the 7090 BEUS word types are "
            "absent from it altogether, and forms that also exist in French "
            "(ensemble, era, video, alliance, ...) get their FRENCH "
            "pronunciation. This spec points at the BE dictionary; rebuilding "
            "processed_beus.csv with it is a separate decision."
        ),
    ),
}


ALL = tuple(CORPORA)

AE = tuple(n for n, s in CORPORA.items() if s.lang == "ae")
VIDEO = tuple(n for n, s in CORPORA.items() if s.modality == "video")
SEQUENCE = tuple(n for n, s in CORPORA.items() if s.modality == "sequence")


def get(name):
    try:
        return CORPORA[name]
    except KeyError:
        raise KeyError(f"unknown corpus {name!r}; known: {sorted(CORPORA)}") from None
