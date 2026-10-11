# Running the analyses

**The manuscript reports three analyses, not four.** There is no
syllable + widened-window run:

| Analysis | unit | window | Status |
| --- | --- | --- | --- |
| Baseline (LabPhon) | `phone` | `base` | complete except BEUS stages 4-8 |
| Syllable units | `syllable` | `base` | aemri complete; five corpora remain |
| Extended articulatory window | `phone` | `prop25` | not started |

Three things identify an analysis, and every filename carries all three:

| | |
|---|---|
| **unit** | `phone` (the LabPhon analysis) or `syllable` (Jason Shaw's alternative) — picks the tree, `phone_based/` or `syllable_based/` |
| **corpus** | `aemri`, `aemri75`, `aemri75anno`, `aeema`, `frmri`, `beus` |
| **window** | `base` (articulatory tokens bounded by the acoustics) or a widened one — `prop25`, `prop10`, `prop50`, `frame1`, `fixed50` |

`base` writes exactly the filenames the project already used, so nothing that
exists has moved. A widened window appends its name: `distances_aemri.csv` for
base, `distances_aemri__prop25.csv` for prop25.

Two stages are deliberately **not** keyed by window, because widening the
articulatory token leaves them unchanged and recomputing them would be wasted:

* the acoustic encoder and its per-speaker MFCC scalers
* surprisal, positional entropy and unit frequency (lexical quantities)

---

## Run order

Every stage is both a command-line script in `runs/` and a cell in
`runs/pipeline.ipynb`. The scripts take `--unit`, `--corpus` (one or more, or
all by default), `--window`, `--limit` and `--overwrite`.

Or run the whole chain for one analysis with a single command:

```bash
./runs/run_all.sh syllable aemri            # base window
./runs/run_all.sh syllable aemri prop25     # widened articulatory window
nohup ./runs/run_all.sh phone beus base --rebuild-dataset > /dev/null 2>&1 &
tail -f runs/logs/phone_beus_base.log
```

It skips stages whose output already exists, so re-running after an
interruption picks up where it stopped, and it stops at the first failure with
the stage named. Stage 1 is destructive and only runs with `--rebuild-dataset`.

| # | Stage | Command | Needs | Rough cost |
|---|---|---|---|---|
| 1 | dataset | `runs/build_dataset.py` | recordings | tens of minutes, destructive |
| 2 | surprisal | `runs/compute_surprisal.py` | repo only | minutes — **done for both units** |
| 3a | feature cache | `runs/build_cache.py` | recordings | minutes — do this first |
| 3 | encoders | `runs/train_encoders.py` | recordings, torch | the long one |
| 4 | latents | `runs/extract_latents.py` | recordings, torch | one pass over the video |
| 5 | distances | `runs/compute_distances.py` | latents only | seconds |
| 6 | entropy | `runs/estimate_entropy.py` | latents, torch | tens of minutes |
| 7 | mutual information | `runs/estimate_mi.py` | latents, torch | tens of minutes |
| 8 | combine | `runs/combine.py` | repo only | seconds |

A full syllable analysis of one corpus:

```bash
python runs/compute_surprisal.py  --unit syllable --corpus aemri
python runs/train_encoders.py     --unit syllable --corpus aemri
python runs/extract_latents.py    --unit syllable --corpus aemri
python runs/compute_distances.py  --unit syllable --corpus aemri
python runs/estimate_entropy.py   --unit syllable --corpus aemri
python runs/estimate_mi.py        --unit syllable --corpus aemri
python runs/combine.py            --unit syllable --corpus aemri
```

The same corpus with widened articulatory windows. Surprisal is reused, and
`--only art` reuses the acoustic encoder too:

```bash
python runs/train_encoders.py     --unit syllable --corpus aemri --window prop25 --only art
python runs/extract_latents.py    --unit syllable --corpus aemri --window prop25
python runs/compute_distances.py  --unit syllable --corpus aemri --window prop25
python runs/estimate_entropy.py   --unit syllable --corpus aemri --window prop25
python runs/estimate_mi.py        --unit syllable --corpus aemri --window prop25
python runs/combine.py            --unit syllable --corpus aemri --window prop25
```

Start with `--limit 2000` on a new corpus: it runs the whole chain on a subset
in minutes and catches a wrong path or a missing scaler before an overnight job
does.

Stage 3 checkpoints every 50 batches and resumes from where it stopped, so it is
safe to interrupt. Stage 4 is new and saves a real amount of time: the three
stages that need encoder output used to decode the recordings three times over,
and now read one cached `.npz`.

---

## Four things worth knowing before you run anything

**1. A 25% pad buys nothing on USC-TIMIT unless it is floored at one frame.**
That corpus runs at 23.18 fps, so one frame is 43 ms, while the median phone
token is 60 ms. A bare 25% pad is 15 ms per side, and **68% of tokens would have
gained no extra frame at all** — the analysis would have come back null for
mechanical reasons. So `CorpusSpec.window()` raises the floor to one
articulatory frame for the corpus in hand, and the rule reads *25% of the
token's duration on each side, but never less than one frame*. Measured frame
rates: aemri 23.18, frmri 50, aemri75/aemri75anno 83.28, aeema ~100.08 Hz. BEUS is
60.10 (measured from the `*_fps.npy` siblings; min 59.96, max 60.20).

Verified on real MRI: with the floor in place every token gains exactly one
frame per side and the acoustic side is untouched. Worth seeing in context —
base articulatory tokens are **1 to 4 frames long**, so the widening roughly
doubles the frames available for a short token. That is a large change, not a
robustness check.

**2. The BEUS pronunciation dictionary — fixed, and the dataset rebuilt.**
`beus.ipynb` loaded `pronunciation_dict_fr.pkl` while passing `lang="be"`, in
both trees. Both notebooks now load `pronunciation_dict_be.pkl`, and
`runs/build_dataset.py` takes the dictionary from the corpus registry, so there
is one place left to get it wrong. `processed_beus.csv` has been rebuilt for
both units from all 7,152 recordings; the old versions are in
`_superseded/beus_french_dictionary/`.

**The effect is a correction, not a rescue — my earlier estimate was wrong and
this is what actually happened.** I had projected the fix would roughly double
usable BEUS data, from a comparison showing `check_alignment` succeeding on 36%
of word forms with the French dictionary and 67% with the British one. That
number turned out not to govern anything: `alignment_to_dataframe` falls back
to the observed phone tier when the dictionary form will not align, so a failed
lookup cost a *label*, not a row. The rebuilt phone dataset has exactly the same
163,319 pairs, the same 7,090 word types and the same token boundaries as
before.

What did change:

| | phone | syllable |
|---|---|---|
| unit pairs | 163,319 → 163,319 | 24,208 → 23,589 |
| phone/syllable labels changed | 2.4% of tokens | 4.8% of shared tokens |
| context strings changed | 6.2% of rows | — |
| surprisal coverage | 60.2% → 62.6% | 45.3% → 50.7% |

The label changes are exactly the French-flavoured vowels you would expect:
*bob* `b_ɔ` → `b_ɒ`, *big* `b_i` → `b_ɪ`, *plastic* `…s_t_i_k` → `…s_t_ɪ_k`,
*train* `tʃ_ɹ_ej_n` → `t_ɹ_ej_n`, *raindrops* `ɹ_e_n…` → `ɹ_ej_n…`. So roughly
one BEUS token in forty carried the wrong vowel category, and the surprisal
values were being computed against the BE lexicon from phone labels sourced
from a French one. Worth fixing, and now fixed — but it does not change the
amount of data, and it will not by itself rescue a null result.

**What still has to be rerun for BEUS.** The phone token *windows* are
byte-identical to before (226,488 tokens, none added or removed), and the
encoders never see labels, so **the trained BEUS phone encoders are still
valid** — skip stage 3 there and you save the most expensive step:

```bash
python3 runs/extract_latents.py   --unit phone --corpus beus
python3 runs/compute_distances.py --unit phone --corpus beus
python3 runs/estimate_entropy.py  --unit phone --corpus beus
python3 runs/estimate_mi.py       --unit phone --corpus beus
python3 runs/combine.py           --unit phone --corpus beus
```

The syllable token set *did* change (1,313 tokens removed, 260 added), so that
tree needs stage 3 as well — and it has never been trained anyway:
`./runs/run_all.sh syllable beus base`.

The stale BEUS phone outputs — distances, entropies, MI and the combined
table — have been moved to `_superseded/beus_french_dictionary/downstream_phone/`
rather than deleted. That matters: the runners skip a stage whose output already
exists, so leaving them in place would have silently reused results computed
from the old labels.

**3. Syllable bigrams are much sparser than phone bigrams.** Lexical coverage —
the share of pairs that get a surprisal value at all:

| corpus | pairs (phone → syllable) | types | coverage (phone → syllable) |
|---|---|---|---|
| aemri | 98,935 → 18,530 | 39 → 652 | 75% → 85% |
| aemri75 / aemri75anno | 26,856 → 3,046 | 42 → 329 | 72% → 57% |
| aeema | 43,459 → 8,069 | 39 → 638 | 76% → 86% |
| frmri | 22,437 → 5,172 | 37 → 247 | 68% → 75% |
| beus | 163,319 → 24,208 | 47 → 1,382 | 60% → 45% |

Two things follow. The per-unit estimates (flexibility, conditional entropy)
need ≥30 tokens per unit, and with 652 syllable types in aemri against 39 phone
types, far fewer units will clear that bar — expect the mixed models to rest on
a much smaller set of unit levels. And aemri75 at 3,046 pairs with 57% coverage
may simply be too thin to carry a syllable analysis; check how many syllable
types survive the 30-token threshold before spending a training run on it.

**3b. The two streams are measured independently.** `extract_latents` encodes
articulatory and acoustic features separately and records `has_art` / `has_ac`
per token, so a token whose MFCCs failed still contributes an articulatory
distance — which is what the original per-corpus notebooks did. Requiring both
streams had silently cost 4,001 articulatory observations on BEUS phone.
Entropy and mutual information still need paired observations and take the
`has_art & has_ac` subset via `utils.pipeline.paired`; only the distinctiveness
stage uses the single-stream tokens. Latent files written before this change
contain paired tokens only, and `stream_masks` treats a missing mask as
all-True, which is correct for them.

**4. Token order was not reproducible, which silently broke training resume.**
`dataset_df_to_token_df` deduplicates through a Python set and then sorts only
by `(art_f, ac_f, subject)`, leaving the order *within* a recording to set
iteration. Python randomises string hashing per process, so that order came out
different on every run — three fingerprints of the same aemri token frame in
three processes gave three different answers.

The serious consequence is resume. Training shuffles with
`sample(random_state=42 + epoch)` and resumes by batch index, on the stated
assumption that the same seed reproduces the same order. It does not, if the
frame being shuffled arrived in a different order. So every time one of these
multi-hour jobs was interrupted and resumed, it continued on a different subset
than the one it left off in, and nothing said so. `utils.pipeline.token_frame`
now sorts on the full token identity, which makes the order a function of the
data; the same fingerprint check now returns the same value every time. The
legacy function is untouched, and `token_frame(..., sort=False)` reproduces the
old behaviour if you ever need to match an earlier run.

**5. Widened windows need more memory during training.** A `prop25` syllable
token carries ~65% more frames, and `collate_video` pads every token in a batch
to the batch's longest. Training at the default batch size of 32 was killed by
the OOM killer in the sandbox where `base` had been fine; `--batch-size 8`
completed. If stage 3 dies without a traceback on a widened window, that is
what happened — lower `--batch-size` rather than looking for a bug.

**6. `cross_syllable_boundary` is dropped from the syllable output.** Every
syllable bigram spans a syllable boundary by definition, so the variable is
constant and `combine.py` omits it for `--unit syllable`. Any model formula
carried over from the phone analysis needs that term removed.

---

## What is done and what is not

**Done and verified.** The BEUS rebuild (both units, dataset and surprisal),
with all six corpus frame rates now measured and set in `utils/corpora.py` —
BEUS ultrasound runs at 60.10 fps, so one frame is 16.6 ms. Surprisal,
positional entropy and unit frequency for all six corpora at the syllable
level — the stage that had never been finished. The
refactor reproduces the published phone numbers bit-for-bit (checked against
`surprisal_aemri.csv`: max absolute difference 1.8e-15 on surprisal, exact on
frequencies).

Two changes were needed to get there. Positional entropy used to scan the whole
unit inventory per row — free with 39 phones, 10,329 tree descents per row with
syllables, which is why that stage never finished. It now reads the context
node's children directly; identical output, and the difference between minutes
and never. And the French lexicon's 7.2M pronunciations build a tree too large
to hold in memory, so the tree is now built only over the prefixes the data
actually queries, with the whole-lexicon total restored at the root. Both were
checked to produce bit-identical results before being used.

Stage 1 is now a script too, and its recording discovery was checked against
the published datasets: every recording in `processed_aemri.csv`,
`processed_aeema.csv`, `processed_frmri.csv`, `processed_aemri75.csv` and
`processed_aemri75anno.csv` is rediscovered, with a couple of extra recordings
that previously contributed no pairs.

**Run end to end, on real data, at small scale.** Stages 3–8 were exercised in
a scratch tree on 400 aemri syllable pairs (638 tokens, 15 recordings), one
epoch, CPU:

| | |
|---|---|
| stage 3 | both encoders trained; art loss 0.984 → 0.745, ac 1.063 → 0.897 |
| stage 4 | 638 of 638 tokens encoded, 64-d art and 64-d ac |
| stage 5 | 400 of 400 pairs measured, no unresolvable lookups |
| stage 6 | both conditional entropy directions estimated |
| stage 7 | MI and flexibility written, with the full diagnostic report |
| stage 8 | 800-row long table, 26 columns, schema matching the published output |

Then the same chain at `--window prop25`, and the two compared token by token:
the articulatory latents differ (different frames went in) while the acoustic
latents are **bit-identical** (they were never widened). Mean articulatory pair
distance moved 0.717 → 0.313; mean acoustic pair distance was identical to
three decimals. That is exactly the asymmetry the analysis is supposed to have,
and it is now demonstrated rather than assumed.

This does **not** mean the numbers from a full run will be sensible — one epoch
on 638 tokens is a plumbing test, not a model. It means the plumbing is sound.

**Not run.** Full-scale training for any corpus, and the BEUS rebuild.

**Suggested order.** Syllable/`base` for `aemri` first: it is the corpus with
the most data and the cleanest lexical coverage, and it exercises every stage.
If that comes out sane, fan out to the other corpora and then to `prop25`.

---

## The feature cache

The extractors take `(file, start, end)` and are called once per token, so a
training epoch reopens each recording hundreds of times and seeks into it. On
compressed video a seek is not free — cv2 decodes forward from the preceding
keyframe — so the cost per token is the keyframe interval, not the token's
length. Measured on USC-TIMIT MRI: **19.1 ms/token seeking against 1.5 ms
sequential**, and 24× on a shuffled training loop. The whole aemri corpus (849
recordings, 30,694 syllable tokens) decodes sequentially in **30 seconds** —
against roughly 88 minutes per epoch the other way. Three epochs paid it three
times, and stage 4 paid it again.

`runs/build_cache.py` reads each recording once, slices out every token, and
writes a float16 memmap plus an index. `train_encoders.py` and
`extract_latents.py` use it automatically when present; `--no-cache` opts out.
Cached features were identical to per-token extraction on every token checked
(max difference 0.0019 — float16 rounding, not misalignment), so this is a
speed change and not a results change.

Build one per (unit, corpus, window) — a widened window covers different
frames, so it needs its own:

```bash
python runs/build_cache.py --unit syllable --corpus aemri aemri75 aemri75anno aeema frmri
```

**Which corpora benefit, and why they differ:**

| corpus | storage | why |
|---|---|---|
| aemri, aemri75, frmri | compressed video | keyframe seeking per token — the big win |
| aeema, aemri75anno | `.mat` | worse in proportion: `loadmat` reparses the whole file for *every* token, at ~155 and ~197 tokens per recording. Caches are tiny (0.04 and 0.2 GB) |
| beus | `.npy` | **skipped automatically.** `_beus_streams` memmaps and slices before converting, so random access is already cheap, and the cache would be 34 GB at phone level. `cache_recommended=False` in its spec; `--force` overrides |

Cache sizes at float16, base window: aemri 1.6 GB syllable / 3.5 GB phone,
aemri75 1.8 / 5.0, frmri 3.9 / 7.0, aeema and aemri75anno negligible. They live
under `.cache/` (gitignored); `SVSF_CACHE` moves them to another disk.

## If a drive is mounted somewhere else

`CorpusSpec.dataset_path` is the canonical path, and it is what gets written
into every dataset csv — which is what keeps those files portable. If the drive
is readable elsewhere right now, say so with `SVSF_MOUNTS`. Dataset building
records the canonical path while reading the real one, and every later stage
resolves recording paths through it too, so a dataset built on one machine can
be trained from on another with one environment variable and no edits:

```bash
export SVSF_MOUNTS="/Volumes/Transcend=/mnt/backup/Transcend"
```

Several mappings separate with `:`. Unset, nothing is rewritten, which is the
normal case on your laptop. This is how the BEUS rebuild ran from a sandbox
that mounts your drives under different names, while still writing
`/Volumes/Seagate Backup Plus Drive/...` into the csv.

## Where things live

```
utils/
  windows.py    how a segmentation interval becomes an analysis window
  corpora.py    one description of each corpus; extractor, model, speaker rule, frame rate
  paths.py      where each (unit, corpus, window) analysis reads and writes
  pipeline.py   the stages themselves: features, latents, entropy, distances
  lexicon.py    surprisal, positional entropy, unit frequency
  sources.py    finding each corpus's recordings and alignments
  training.py   the autoencoder loop, with resume
runs/
  pipeline.ipynb   all stages as notebook cells, with a config cell at the top
  *.py             the same stages as command-line scripts
  run_all.sh       one analysis end to end, logged and resumable
  finish_beus.sh   BEUS phone/base, stages 4-8
  queue.sh         several analyses back to back, unattended
_superseded/
  syllable_surprisal_calculation/   the old per-corpus syllable notebooks
```

The five old syllable surprisal notebooks were moved, not deleted — they never
wrote a csv and would not have finished, and `runs/compute_surprisal.py`
replaces them. Delete `_superseded/` when you are satisfied.
