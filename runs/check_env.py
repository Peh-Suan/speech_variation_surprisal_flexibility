#!/usr/bin/env python3
"""
Is this interpreter able to run the pipeline?

Run it first, in the environment you intend to use. It imports everything the
stages need, reports the compute device, and checks that the drives are
mounted where the corpus registry expects them -- the three things that
otherwise fail forty minutes into a run.

    conda activate <your env>
    python runs/check_env.py

If it reports a Homebrew or /usr/bin interpreter while your prompt shows an
environment name, `python3` is being shadowed on PATH -- see the warning this
prints. The stage runners avoid the problem by using the active environment's
own binary rather than whatever `python3` resolves to.
"""

import importlib
import os
import sys

# Mirror what the stage runners do (runs/_common.py), so this check reports the
# environment the pipeline actually trains in rather than a bare shell's.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REQUIRED = [
    ("numpy", "arrays"),
    ("pandas", "every dataset and result table"),
    ("scipy", "loading .mat files for EMA and tracked contours"),
    ("sklearn", "scalers, folds, PCA in the MI estimator"),
    ("torch", "the autoencoders and the MDN"),
    ("cv2", "decoding MRI and ultrasound video"),
    ("librosa", "MFCCs"),
    ("soundfile", "audio durations"),
    ("tgt", "reading TextGrids"),
    ("tqdm", "progress bars"),
]

OPTIONAL = [
    ("eng_to_ipa", "American English lookup -- stage 1 for the ae corpora"),
    ("phonecodes", "ARPAbet conversion -- stage 1"),
    ("openpyxl", "USC-75 Subjects.xlsx -- stage 1 for aemri75"),
    ("matplotlib", "the plots in the analysis notebooks"),
    ("seaborn", "same"),
]


def check(items, label):
    missing = []
    print(f"\n{label}")
    for name, why in items:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "")
            print(f"  ok       {name:14s} {version:12s} {why}")
        except Exception as exc:
            missing.append(name)
            print(f"  MISSING  {name:14s} {'':12s} {why}  ({type(exc).__name__})")
    return missing


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"python   {sys.version.split()[0]}")
    print(f"binary   {sys.executable}")

    env = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV")
    if env:
        print(f"env      {env}")
        if not sys.executable.startswith(env):
            print("\n  !! This interpreter is NOT from the active environment.")
            print("     Your prompt shows an environment, but PATH is resolving")
            print("     python3 to something else (often Homebrew's python@3.x).")
            print(f"     Re-run as:  {env}/bin/python runs/check_env.py")
            print("     Everything below describes the WRONG interpreter.")
    else:
        print("env      (no conda or virtualenv active)")

    missing = check(REQUIRED, "Required:")
    absent = check(OPTIONAL, "Optional (stage 1 and plotting only):")

    print("\nCompute:")
    try:
        import torch
        if torch.backends.mps.is_available():
            print("  mps available -- training will use the GPU")
        elif torch.cuda.is_available():
            print("  cuda available")
        else:
            print("  CPU only. Training will be very slow; check that this is "
                  "the environment you meant to use.")
    except Exception as exc:
        print(f"  cannot tell: {exc}")

    print("\nModel on the GPU:")
    print("  (a real forward and backward pass per corpus -- this is what "
          "catches\n   MPS gaps in seconds instead of mid-epoch)")
    print(f"  PYTORCH_ENABLE_MPS_FALLBACK="
          f"{os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')} "
          f"(set by the pipeline; the video corpora need it, because MPS has "
          f"no backward\n   kernel for the STN's grid_sample)")
    try:
        import torch
        if not torch.backends.mps.is_available():
            print("  skipped (no MPS)")
        else:
            from utils.corpora import CORPORA
            from utils.train import train_step
            for name in CORPORA:
                spec = CORPORA[name]
                try:
                    model = spec.build_art_model("mps")
                    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
                    if spec.modality == "video":
                        hw = spec.art_model_kwargs["input_hw"]
                        ch = spec.art_model_kwargs["input_channels"]
                        x = torch.randn(2, ch, 5, *hw, device="mps")
                        time_dim = 2
                    else:
                        shape = spec.art_model_kwargs["feature_shape"]
                        x = torch.randn(2, 5, *shape, device="mps")
                        time_dim = 1
                    lengths = torch.tensor([5, 4], device="mps")
                    durations = torch.tensor([0.2, 0.16], device="mps")
                    step = train_step(model, opt, x, lengths, durations,
                                      time_dim=time_dim, alpha=0.5,
                                      scales=(1.0, 1.0), duration_mode="off")
                    print(f"  ok       {name:12s} forward+backward, loss {step['total']:.4f}")
                except Exception as exc:
                    msg = str(exc).split("\n")[0][:110]
                    print(f"  FAILED   {name:12s} {type(exc).__name__}: {msg}")
    except Exception as exc:
        print(f"  could not run: {exc}")

    print("\nSpeaker ids vs. scalers:")
    try:
        import pickle
        from utils.corpora import CORPORA
        from utils.paths import RunPaths
        from utils.pipeline import load_dataset
        for name, spec in CORPORA.items():
            line = f"  {name:14s}"
            for unit in ("phone", "syllable"):
                paths = RunPaths(unit, name, "base")
                if not (os.path.exists(paths.ac_scalers) and os.path.exists(paths.dataset)):
                    continue
                with open(paths.ac_scalers, "rb") as f:
                    keys = set(pickle.load(f))
                subs = set(load_dataset(spec, paths.dataset, unit=unit)["subject"].unique())
                bad = sorted(subs - keys)
                line += (f" {unit}: {'ok' if not bad else 'MISMATCH ' + str(bad[:3])}  ")
            print(line)
    except Exception as exc:
        print(f"  could not check: {exc}")

    print("\nCorpora:")
    try:
        from utils.corpora import CORPORA
        from utils.sources import local_path
        for name, spec in CORPORA.items():
            path = local_path(spec.dataset_path)
            state = "found" if os.path.isdir(path) else "NOT MOUNTED"
            print(f"  {state:12s} {name:14s} {path}")
    except Exception as exc:
        print(f"  could not read the corpus registry: {exc}")

    print()
    if missing:
        print(f"Install these before running anything: {' '.join(missing)}")
        return 1
    print("Required packages are all present.")
    if absent:
        print(f"Missing optional packages ({' '.join(absent)}) only matter if you "
              "rebuild a dataset or open the analysis notebooks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
