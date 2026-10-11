"""
Per-phoneme mutual information between articulation and acoustics, estimated with
the MDN in mdn.py.

What this computes
------------------

Six cross-validated density models, all on the same folds and the same per-fold
preprocessing:

    art|ac,phone    p(art | ac, phone)     \  within-category MI  -- the predictor
    art|phone       p(art | phone)         /
    art|ac          p(art | ac)            \
    art|.           p(art)                  |  two-direction PMI identity
    ac|art          p(ac | art)             |  -- a free model-error diagnostic
    ac|.            p(ac)                  /

The predictor:

    pmi_within[i] = log p(art_i | ac_i, phone_i) - log p(art_i | phone_i)
    I_ph          = mean of pmi_within over that phoneme's tokens
    flexibility   = -I_ph      (low MI = acoustics constrains articulation weakly)

Note the sign flip relative to H_art_given_ac: whatever coefficient you had on
the H version of this predictor will change sign when you swap in -I.

The diagnostic: pointwise,

    log p(art|ac) - log p(art)   ==   log p(ac|art) - log p(ac)

exactly, because both equal log[ p(art,ac) / (p(art) p(ac)) ]. Any deviation is
pure model error, and its per-phoneme mean gives you a credibility weight per
phoneme -- which matters when this feeds a mixed model over ~7 phone2 levels.

No Jacobian correction is needed anywhere for MI: within a fold both terms of a
difference share the same target scaler, so log|det J| cancels token by token.
H values, which do need it, are corrected separately and are reported as NaN
when PCA is on (the projection is not invertible).
"""

import copy

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    StratifiedGroupKFold,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .models import MDN # -> from utils.models import MDN

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 0
K_FOLDS = 5
MIN_TOKENS = 30

N_PCA = None  # try 20. Strongly recommended: a 64-dim density on ~30 tokens
# per phoneme is hopeless, and PCA is what makes the marginals estimable.
# Fit inside the fold, so no leakage. With PCA on, MI is measured in the
# retained subspace and is a lower bound on the full-space MI.

N_COMPONENTS = 5
HIDDEN_DIM = 256
SIGMA_MIN = 0.01
LR = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 2000
PATIENCE = 10

# Acoustic typicality trim. The MDN is ALWAYS fit on every token -- this masks
# only which tokens are averaged when estimating MI and H. Nothing is lost from
# the density estimate, so the trimmed and untrimmed numbers come out of the
# same fit and can be compared for free.
#
# mode: "mahalanobis" keep d2 <= d + k*sqrt(2d), i.e. within k SD of the chi2_d
#                     distribution of squared distance. The multivariate reading
#                     of "within k SD" -- at k=2, d=64, drops ~3%.
#       "per_dim"     keep max|z| <= k over dimensions. Sounds equivalent and is
#                     not: at k=2 over 64 dims this keeps 0.9545^64 = 5% of
#                     tokens. Included so you can see that, not to use.
#       "quantile"    keep the closest fraction k (k=0.9 keeps 90%). Assumption-
#                     free, and the retention rate is exactly controlled.
#       None          no trim.
# by:   "phone" centres on each phoneme's own acoustic mean -- what you want.
#       "global" centres on the grand mean, which deletes whole categories that
#       simply live at the edge of acoustic space (/r/ and its very low F3 go
#       first). Available for comparison; do not report it.
TRIM = dict(mode="mahalanobis", k=2.0, by="phone")


def acoustic_mask(ac_raw, phon, mode="mahalanobis", k=2.0, by="phone"):
    """Which tokens count when ESTIMATING MI and H. Not used for fitting.

    Computed on all tokens including held-out ones, which is fine precisely
    because it never touches the density models.
    """
    ac_raw = np.asarray(ac_raw, dtype=float)
    phon = np.asarray(phon)
    N, d = ac_raw.shape
    keep = np.ones(N, dtype=bool)
    if mode is None:
        return keep

    groups = ([np.arange(N)] if by == "global"
              else [np.where(phon == p)[0] for p in np.unique(phon)])

    for idx in groups:
        if len(idx) < 3:
            continue
        X = ac_raw[idx]
        sd = X.std(0, ddof=1)
        sd[sd == 0] = np.inf            # constant dims contribute nothing
        Z = (X - X.mean(0)) / sd
        if mode == "per_dim":
            keep[idx] = np.abs(Z).max(1) <= k
        elif mode == "mahalanobis":
            keep[idx] = (Z**2).sum(1) <= d + k * np.sqrt(2.0 * d)
        elif mode == "quantile":
            d2 = (Z**2).sum(1)
            keep[idx] = d2 <= np.quantile(d2, k)
        else:
            raise ValueError(f"unknown trim mode: {mode!r}")
    return keep


def _device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(X_tr, Y_tr, X_va, Y_va, device, seed=SEED, verbose=False):
    torch.manual_seed(seed)

    model = MDN(
        x_dim=X_tr.shape[1],
        y_dim=Y_tr.shape[1],
        n_components=N_COMPONENTS,
        hidden_dim=HIDDEN_DIM,
        sigma_min=SIGMA_MIN,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    loader = DataLoader(
        TensorDataset(t(X_tr), t(Y_tr)), batch_size=BATCH_SIZE, shuffle=True
    )
    Xva_t, Yva_t = t(X_va).to(device), t(Y_va).to(device)

    best_val, best_state, counter = float("inf"), copy.deepcopy(model.state_dict()), 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            loss = -model.log_prob(Xb, Yb).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val = -model.log_prob(Xva_t, Yva_t).mean().item()

        if val < best_val - 1e-4:
            best_val, counter = val, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1

        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch}: val={val:.3f} best={best_val:.3f}")

        if counter >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val


# ---------------------------------------------------------------------------
# Cross-validated held-out log densities
# ---------------------------------------------------------------------------


def make_folds(N, groups=None, stratify=None, k=K_FOLDS, seed=SEED):
    """Group by utterance (art_f), not by token. Random token-level splits put
    tokens from the same utterance in train and test, which inflates held-out
    likelihood unevenly across phonemes (frequent phonemes have more near-
    duplicate neighbours available).

    StratifiedGroupKFold keeps whole groups on one side of the split AND keeps
    the phoneme distribution balanced across folds, so a rare phoneme cannot end
    up nearly absent from one fold's training set.
    """

    idx = np.arange(N)
    if groups is not None:
        groups = np.asarray(groups)  # a list won't accept an array index
    if stratify is not None:
        stratify = np.asarray(stratify)

    if groups is None:
        print("WARNING: no groups given, falling back to token-level KFold. "
              "Pass groups=<utterance id per token> -- collect art_f in the "
              "same loop that builds art_vectors, so the indices stay aligned.")
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        outer = [te for _, te in splitter.split(idx)]
    else:
        y = np.zeros(N, dtype=int) if stratify is None else stratify
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        outer = [te for _, te in splitter.split(idx, y=y, groups=groups)]

        if stratify is not None:
            counts = np.array([
                [(stratify[te] == lab).sum() for te in outer]
                for lab in np.unique(stratify)
            ])
            thin = counts.min(axis=1)
            if (thin < 3).any():
                bad = np.unique(stratify)[thin < 3]
                print(f"  WARNING: {list(bad)} have <3 tokens in some fold; "
                      f"their per-phoneme means will be unstable")

    inner = []
    for te in outer:
        tr = np.setdiff1d(idx, te)
        if groups is None:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(tr)
            cut = int(0.9 * len(perm))
            inner.append((perm[:cut], perm[cut:]))
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
            a, b = next(gss.split(tr, groups=groups[tr]))
            inner.append((tr[a], tr[b]))

    return outer, inner


def make_prep(arrays, inner, n_pca=N_PCA, seed=SEED):
    """Per-fold StandardScaler (+ optional PCA), fit on the inner training set
    only. Shared by every model in that fold, so target transforms are identical
    across models and log|det J| cancels in any difference."""

    prep = []
    for tr, _ in inner:
        fold = {}
        for key, raw in arrays.items():
            sc = StandardScaler().fit(raw[tr])
            Z = sc.transform(raw)
            if n_pca:
                pca = PCA(n_components=n_pca, random_state=seed).fit(Z[tr])
                Z, logJ = pca.transform(Z), np.nan
            else:
                logJ = float(np.log(sc.scale_).sum())
            fold[key] = (Z.astype(np.float32), logJ)
        prep.append(fold)
    return prep


def fit_cv(build_x, target, prep, outer, inner, N, device, label=""):
    """-> (logp_model, logp_orig)

    logp_model is in the fold's model coordinates: use this for MI.
    logp_orig  adds back the Jacobian: use this for H, NaN under PCA.
    """

    logp_model = np.zeros(N)
    logp_orig = np.zeros(N)

    for f, te in enumerate(outer):
        tr, va = inner[f]
        X = build_x(prep[f]).astype(np.float32)
        Y, logJ = prep[f][target]

        model, best_val = train_model(X[tr], Y[tr], X[va], Y[va], device, seed=SEED + f)

        with torch.no_grad():
            Xte = torch.from_numpy(X[te]).float().to(device)
            Yte = torch.from_numpy(Y[te]).float().to(device)
            lp = model.log_prob(Xte, Yte).cpu().numpy()
            floor = model.floor_fraction(Xte)

        logp_model[te] = lp
        logp_orig[te] = lp - (0.0 if np.isnan(logJ) else logJ)

        print(f"  [{label}] fold {f}: val NLL {best_val:8.2f}   "
              f"sigmas at floor {floor:5.1%}")

    return logp_model, logp_orig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def compute_all(art_raw, ac_raw, phonemes, groups=None):
    art_raw, ac_raw = np.asarray(art_raw), np.asarray(ac_raw)
    N = len(art_raw)
    assert len(ac_raw) == N == len(phonemes)
    if groups is not None:
        groups = np.asarray(groups)
        assert len(groups) == N, "groups must be aligned with art_raw"

    device = _device()
    phon = np.asarray(phonemes)
    uniq = np.unique(phon)
    phone_oh = np.eye(len(uniq), dtype=np.float32)[np.searchsorted(uniq, phon)]
    print(f"N={N}  art={art_raw.shape[1]}d  ac={ac_raw.shape[1]}d  "
          f"phones={len(uniq)}  device={device}"
          f"{'  PCA->' + str(N_PCA) if N_PCA else ''}")

    d = max(art_raw.shape[1], ac_raw.shape[1])
    if N_PCA is None and N < 50 * d:
        print(f"\n  WARNING: {N} tokens for a {d}-dim conditional density.\n"
              f"  A 5-component mixture in {d}d has {5 * (2 * d + 1)} outputs per\n"
              f"  token and you have ~{int(0.72 * N)} training tokens per fold. The\n"
              f"  fits will be dominated by misfit, and the misfit is phoneme-\n"
              f"  dependent, which is exactly the confound MI is meant to remove.\n"
              f"  Set N_PCA = 10-16 and rerun.\n")

    outer, inner = make_folds(N, groups, stratify=phon)
    prep = make_prep({"art": art_raw, "ac": ac_raw}, inner)

    zero = np.zeros((N, 0), dtype=np.float32)
    configs = {
        # within-category MI -- the predictor
        "art|ac,phone": (lambda Z: np.hstack([Z["ac"][0], phone_oh]), "art"),
        "art|phone": (lambda Z: phone_oh, "art"),
        # two-direction identity -- the diagnostic
        "art|ac": (lambda Z: Z["ac"][0], "art"),
        "art|.": (lambda Z: zero, "art"),
        "ac|art": (lambda Z: Z["art"][0], "ac"),
        "ac|.": (lambda Z: zero, "ac"),
    }

    lp, lp_orig = {}, {}
    for name, (build, target) in configs.items():
        print(f"\n=== {name} ===")
        lp[name], lp_orig[name] = fit_cv(
            build, target, prep, outer, inner, N, device, label=name
        )

    # Acoustic typicality mask -- reporting only, the fits above used everything
    mask = acoustic_mask(ac_raw, phon, **TRIM)
    print(f"\ntrim {TRIM}: keeping {mask.sum():,}/{N:,} tokens ({mask.mean():.1%})")
    if TRIM["mode"] is not None:
        for m in ("mahalanobis", "per_dim", "quantile"):
            kk = 0.9 if m == "quantile" else TRIM["k"]
            alt = acoustic_mask(ac_raw, phon, mode=m, k=kk, by=TRIM["by"])
            print(f"    for comparison, {m:12s} k={kk}: {alt.mean():6.1%}")
        print("    (three reasonable readings of the same phrase -- if your "
              "result\n     depends on which, it is not a result)")

    # lp      : held-out log density in fold coordinates -> use for MI
    # lp_orig : same, Jacobian-corrected -> H = -mean(lp_orig)
    return dict(lp=lp, lp_orig=lp_orig, phon=phon, uniq=uniq, N=N, mask=mask)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def per_phoneme(res, min_tokens=MIN_TOKENS, n_boot=2000, seed=SEED, use_mask=True):
    lp, phon, uniq = res["lp"], res["phon"], res["uniq"]
    rng = np.random.default_rng(seed)
    mask = res["mask"] if use_mask else np.ones(res["N"], dtype=bool)

    pmi_within = lp["art|ac,phone"] - lp["art|phone"]  # the predictor
    pmi_a = lp["art|ac"] - lp["art|."]                 # identity, direction 1
    pmi_b = lp["ac|art"] - lp["ac|."]                  # identity, direction 2

    # How much does knowing the category add beyond the acoustics? That gap is
    # cross-category confusability, and in the unconditioned H it was being
    # scored as flexibility. Expect it largest for the weak fricatives.
    confus = lp["art|ac,phone"] - lp["art|ac"]

    print("\n" + "=" * 74)
    print("Two-direction PMI identity (these are the same quantity pointwise)")
    print("=" * 74)
    print(f"  pearson r                {np.corrcoef(pmi_a, pmi_b)[0, 1]: .4f}")
    print(f"  mean |disagreement|      {np.abs(pmi_a - pmi_b).mean(): .3f} nats")
    print(f"  mean PMI, direction 1    {pmi_a.mean(): .3f} nats")
    print(f"  mean PMI, direction 2    {pmi_b.mean(): .3f} nats")
    print("  (a large gap means at least one density model is badly misfit; "
          "fix that before reading anything else)")

    rows = []
    for ph in uniq:
        all_i = np.where(phon == ph)[0]
        i = all_i[mask[all_i]]          # estimate on typical tokens only
        if len(i) < min_tokens:
            continue

        v = pmi_within[i]
        boot = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])

        j = rng.permutation(i)
        h = len(j) // 2

        rows.append(dict(
            phone=ph,
            n=len(i),
            n_all=len(all_i),
            kept=len(i) / len(all_i),
            mi=v.mean(),
            flexibility=-v.mean(),
            ci_lo=lo,
            ci_hi=hi,
            half_a=pmi_within[j[:h]].mean(),
            half_b=pmi_within[j[h:]].mean(),
            model_err=np.abs(pmi_a - pmi_b)[i].mean(),
            confusability=confus[i].mean(),
            # entropy is MINUS the mean log density. Primary H is now the
            # phone-conditioned one; the old unconditioned H is kept beside it.
            H_art_given_ac_phone=-res["lp_orig"]["art|ac,phone"][i].mean(),
            H_art_given_ac=-res["lp_orig"]["art|ac"][i].mean(),
        ))

    import pandas as pd

    df = pd.DataFrame(rows).sort_values("flexibility", ascending=False)

    r = np.corrcoef(df.half_a, df.half_b)[0, 1]
    print("\n" + "=" * 74)
    print("Reliability and convergent validity")
    print("=" * 74)
    print(f"  split-half r             {r: .3f}   "
          f"(Spearman-Brown {2 * r / (1 + r): .3f})")
    print("  -> this is the ceiling on any effect this predictor can support")

    from scipy.stats import spearmanr

    ok = df.H_art_given_ac.notna()
    if ok.any():
        # all three are "flexibility": -MI, and the two H variants
        print(f"  spearman(-MI, H|ac,phone) "
              f"{spearmanr(df.flexibility[ok], df.H_art_given_ac_phone[ok]).statistic: .3f}")
        print(f"  spearman(-MI, H|ac)       "
              f"{spearmanr(df.flexibility[ok], df.H_art_given_ac[ok]).statistic: .3f}")
        print(f"  spearman(H|ac,phone, H|ac)"
              f"{spearmanr(df.H_art_given_ac_phone[ok], df.H_art_given_ac[ok]).statistic: .3f}")
        print("  -> near 1 means the change is cosmetic; well below 1 means the "
              "old\n     measure was tracking something these remove")

    print("\n" + "=" * 74)
    print("Category confusability (how much the phone label adds beyond ac)")
    print("=" * 74)
    top = df.nlargest(8, "confusability")[["phone", "confusability", "H_art_given_ac"]]
    print(top.to_string(index=False, float_format="%.3f"))
    print("  -> these phonemes' scores under the UNCONDITIONED H were partly\n"
          "     cross-category ambiguity, not within-category flexibility")

    # Did the trim eat the effect? If it removed more tokens from exactly the
    # phonemes that come out flexible, it truncated what you set out to measure.
    if (df.kept < 1).any():
        r_trim = spearmanr(df.kept, df.flexibility).statistic
        print("\n" + "=" * 74)
        print("Trim balance")
        print("=" * 74)
        spread = df.kept.max() - df.kept.min()
        print(f"  retention: min {df.kept.min():.1%}  median {df.kept.median():.1%}"
              f"  max {df.kept.max():.1%}  (spread {spread:.1%})")
        print(f"  spearman(retention, flexibility) {r_trim: .3f}")
        # A rank correlation is +/-1 by construction on a handful of near-tied
        # values, so require a real spread and enough phonemes before warning.
        if r_trim < -0.3 and spread > 0.05 and len(df) >= 8:
            print("  !! the trim is removing MORE tokens from the phonemes that "
                  "score as\n     flexible -- it is truncating the effect. Rerun "
                  "with TRIM['mode']=None\n     and compare before reporting "
                  "anything.")
        elif spread <= 0.05:
            print("  retention is even across phonemes, so the trim is not "
                  "differentially\n  truncating the categories you are comparing")

    return df.reset_index(drop=True)


def compare_trim(res, **kw):
    """Trimmed vs untrimmed, from the same fits. Costs nothing -- the trim is a
    reporting mask, so both come out of one set of models. Report this as a
    sensitivity analysis rather than choosing a threshold and hoping."""
    from scipy.stats import spearmanr

    full = per_phoneme(res, use_mask=False, **kw)
    trim = per_phoneme(res, use_mask=True, **kw)
    m = full[["phone", "flexibility"]].merge(
        trim[["phone", "flexibility", "kept"]], on="phone",
        suffixes=("_full", "_trim"))
    print("\n" + "=" * 74)
    print("Sensitivity: does the trim change the ranking?")
    print("=" * 74)
    print(f"  phonemes surviving min_tokens: full {len(full)}, trimmed {len(trim)}")
    print(f"  spearman(flex_full, flex_trim) "
          f"{spearmanr(m.flexibility_full, m.flexibility_trim).statistic: .3f}")
    print(f"  largest rank shifts:")
    m["shift"] = (m.flexibility_full.rank(ascending=False)
                  - m.flexibility_trim.rank(ascending=False))
    print(m.reindex(m.shift.abs().sort_values(ascending=False).index)
           .head(6).to_string(index=False, float_format="%.3f"))
    return m


if __name__ == "__main__":
    # art_raw  (N, 64) float32   ac_raw (N, 64) float32
    # phonemes (N,) str          groups (N,) utterance id  <- collect art_f!
    #
    # res = compute_all(art_raw, ac_raw, phonemes, groups)
    # df  = per_phoneme(res)
    # df.to_csv("entropies/aeema_mi.csv", index=False)
    pass