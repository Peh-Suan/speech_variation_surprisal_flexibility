"""
Finding each corpus's recordings and alignments.

Stage 1 was the last stage still living only in per-corpus notebooks, which is
how the BE ultrasound branch came to load the FRENCH pronunciation dictionary
in both trees at once: the line was copied from frmri.ipynb and then copied
again into syllable_based/. With the discovery logic here and the dictionary
coming from the CorpusSpec, there is one place to be wrong.

Each `find_*` returns a list of (source, art_f, ac_f), where `source` is
whatever `alignment_to_dataframe` expects for that corpus -- a TextGrid object
for the TextGrid corpora, a DataFrame for the USC-TIMIT .trans files.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import re

import pandas as pd

from utils.mounts import canonical_path, local_path  # noqa: F401


def _read_textgrid(path, encodings=("utf-8", "utf-16")):
    import tgt
    last = None
    for enc in encodings:
        try:
            return tgt.read_textgrid(path) if enc == "utf-8" else \
                   tgt.read_textgrid(path, encoding=enc)
        except UnicodeError as e:
            last = e
    raise last


def _read_fr_textgrid(path):
    """
    FR-MRI TextGrids: some interval boundaries run past the file's own xmax,
    which tgt rejects. Clamp them to xmax and read from a temporary copy.
    """
    import tempfile
    import tgt
    try:
        return tgt.read_textgrid(path)
    except ValueError:
        pass
    with open(path) as f:
        lines = f.readlines()
    xmax = float(lines[4].split(" = ")[-1].strip())
    out = []
    for line in lines:
        for field in ("xmin", "xmax"):
            if line.startswith(f"\t\t\t{field}"):
                value = float(line.split(" = ")[-1].strip())
                line = f"\t\t\t{field} = {min(value, xmax)}\n"
        out.append(line)
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as tmp:
        tmp.writelines(out)
        tmp.seek(0)
        return tgt.read_textgrid(tmp)


def _usctimit(dataset_path, art_subdir, art_ext):
    """USC-TIMIT MRI and EMA: a .trans file per recording, beside wav and media."""
    out = []
    for trans_f in sorted(glob.glob(os.path.join(dataset_path, "Data/*/trans/*.trans"))):
        identifier = os.path.basename(trans_f).replace(".trans", "")
        subject = identifier.split("_")[2].upper()
        art_f = os.path.join(dataset_path, "Data", subject, art_subdir,
                             f"{identifier}.{art_ext}")
        ac_f = os.path.join(dataset_path, "Data", subject, "wav", f"{identifier}.wav")
        if os.path.exists(art_f) and os.path.exists(ac_f):
            out.append((pd.read_csv(trans_f, header=None), art_f, ac_f))
    return out


def find_aemri(spec, root):
    return _usctimit(spec.dataset_path, "avi", "avi")


def find_aeema(spec, root):
    return _usctimit(spec.dataset_path, "mat", "mat")


def _usc75(spec, want="video"):
    """
    USC-75: alignments under USC-75SpeakerAnnot16/, media two directories up.

    Restricted to speakers whose Origin is USA, and matched to media by a
    normalised filename -- the alignment and media trees disagree about an
    embedded repetition number, so an exact join misses most files.
    """
    annot = spec.dataset_path.rstrip("/")
    usc75 = os.path.dirname(annot)

    subjects = pd.read_excel(os.path.join(annot, "Subjects.xlsx"))
    native = set(subjects.loc[subjects["Origin"] == "USA", "Subject ID"].astype(str))

    def subject_of(path):
        return next(p for p in path.split("/") if p.startswith("sub"))

    norm = lambda p: re.sub(r"_(\d+)_", "_", p).lower()
    mp4s = {norm(p): p for p in glob.glob(os.path.join(usc75, "sub*/2drt/video/*_video.mp4"))}
    mats = {norm(p): p for p in glob.glob(os.path.join(annot, "sub*/track/*_track.mat"))}

    out = []
    for tgrd_f in sorted(glob.glob(os.path.join(annot, "sub*/alignment/*.TextGrid"))):
        if subject_of(tgrd_f) not in native:
            continue
        identifier = os.path.basename(tgrd_f).replace(".TextGrid", "")
        subject = next(i for i in identifier.split("_") if i.startswith("sub"))
        target = os.path.join(usc75, subject, "2drt/video",
                              identifier.replace("_text", "_video.mp4"))
        mp4 = mp4s.get(norm(target))
        if mp4 is None:
            continue
        ac_f = mp4.replace(".mp4", ".wav")
        if want == "video":
            art_f = mp4
        else:
            art_f = mats.get(norm(os.path.join(
                annot, subject, "track",
                os.path.basename(ac_f).replace("_video.wav", "_track.mat"))))
            if art_f is None:
                continue
        out.append((_read_textgrid(tgrd_f), art_f, ac_f))
    return out


def find_aemri75(spec, root):
    return _usc75(spec, want="video")


def find_aemri75anno(spec, root):
    return _usc75(spec, want="track")


def find_frmri(spec, root):
    out = []
    pattern = os.path.join(spec.dataset_path, "P*/OTHER/S*/*.textgrid")
    for tgrd_f in sorted(glob.glob(pattern)):
        _, _, subject, session = os.path.basename(tgrd_f).replace(".textgrid", "").split("_")
        base = os.path.join(spec.dataset_path, subject, "OTHER", session)
        art_f = os.path.join(base, f"VIDEO_{subject}_{session}.avi")
        ac_f = os.path.join(base, f"DENOISED_SOUND_{subject}_{session}.wav")
        out.append((_read_fr_textgrid(tgrd_f), art_f, ac_f))
    return out


def find_beus(spec, root):
    """
    BE ultrasound: frames, frame rate, audio and alignment as four siblings.

    Only the read-aloud and spontaneous sets ("aud"/"spo"), and only speakers
    whose fourth identifier character is "e" -- the same restriction the
    original notebook applied.
    """
    base = spec.dataset_path.rstrip("/")
    out = []
    for frames in sorted(glob.glob(os.path.join(base, "dataset/*_frames.npy"))):
        identifier = os.path.basename(frames).replace("_frames.npy", "")
        if not ("aud" in identifier or "spo" in identifier):
            continue
        if len(identifier) < 4 or identifier[3] != "e":
            continue
        fps = frames.replace("_frames", "_fps")
        tgrd_f = frames.replace("_frames.npy", ".TextGrid")
        ac_f = os.path.join(base, "dataset_audio", f"{identifier}_audio.npy")
        if not (os.path.isfile(fps) and os.path.isfile(tgrd_f)):
            continue
        out.append((_read_textgrid(tgrd_f), frames, ac_f))
    return out


FINDERS = {
    "aemri": find_aemri,
    "aemri75": find_aemri75,
    "aemri75anno": find_aemri75anno,
    "aeema": find_aeema,
    "frmri": find_frmri,
    "beus": find_beus,
}


def find(spec, root):
    """
    (source, art_f, ac_f) per recording, with art_f and ac_f canonical.

    Discovery runs against wherever the drive is readable now; the paths handed
    back are the canonical ones, so a dataset built through a remapped mount is
    identical to one built directly.
    """
    here = dataclasses.replace(spec, dataset_path=local_path(spec.dataset_path))
    found = FINDERS[spec.name](here, root)
    return [(source, canonical_path(art_f), canonical_path(ac_f))
            for source, art_f, ac_f in found]


def lexicon_kwargs(spec, root):
    """
    The pronunciation resources `alignment_to_dataframe` needs for this corpus.

    American English converts with eng_to_ipa plus a preferred-IPA table;
    British English and French look the word up in their own dictionary. The
    dictionary comes from the CorpusSpec, which is the fix for the BE branch
    having loaded the French one.
    """
    import pickle

    if spec.lang == "ae":
        return {"preferred_ipas_f": spec.resolve(root, "preferred_ipas")}

    path = spec.resolve(root, "pronunciation_dict")
    if path is None:
        raise ValueError(f"{spec.name} is lang={spec.lang} but has no "
                         "pronunciation_dict in its CorpusSpec")
    with open(path, "rb") as f:
        return {"pronunciation_dict": pickle.load(f)}
