"""
Lexical statistics: bigram surprisal, positional entropy, unit frequency.

Phone and syllable analyses differ here in four small ways that were previously
spread across twelve notebooks:

    separator   phones inside a word are joined by "_", syllables by "."
    counts      the pronunciation counts are re-keyed to the unit in question
    context     the phone analysis injects syllable boundaries into the
                context string ("_._"), so that a bigram spanning a boundary is
                conditioned on a context that records the boundary; the
                syllable analysis has boundaries implicitly, in its separator
    columns     "phone2_pos_entropy" vs "syllable2_pos_entropy", and so on

All four follow from (unit, lang), so they are decided here once.

These quantities are lexical, not phonetic: they depend on the corpus's word
forms and on nothing about how token boundaries were drawn. A widened
articulatory window therefore does not change them, which is why
utils.paths keys the surprisal file by corpus rather than by window tag.
"""

from __future__ import annotations

import os
import pickle

import pandas as pd
import tqdm

SEPARATORS = {"phone": "_", "syllable": "."}


def load_counts(spec, root, unit):
    """
    Pronunciation counts re-keyed to `unit`, with the separator that unit uses.

    For phones the syllabified key keeps its boundaries as an explicit "_._"
    token, so `get_contextual_surprisal` can condition on them. For syllables
    the boundary IS the separator, so nothing extra is needed.
    """
    from utils.ipa_processing import syllabify_ipa

    path = spec.resolve(root, "pronunciation_counts")
    with open(path, "rb") as f:
        counts = pickle.load(f)

    out = {}
    for key, count in counts.items():
        if not key or "*" in key:
            continue
        syllabified = syllabify_ipa(key, lang=spec.lang)
        if unit == "phone":
            syllabified = syllabified.replace(".", "_._")
        out[syllabified] = count
    return out


def unit_counts(pronunciation_counts, unit):
    """Token frequency of each unit across the lexicon."""
    sep = SEPARATORS[unit]
    totals = {}
    for pronunciation, count in pronunciation_counts.items():
        for u in pronunciation.split(sep):
            totals[u] = totals.get(u, 0) + count
    return totals


def queried_contexts(df, unit):
    """
    Every prefix the surprisal stage will actually look up.

    Both quantities condition on a word-initial prefix: positional entropy on
    `context`, surprisal on `context` minus its final unit. Nothing else in the
    lexicon is ever consulted.
    """
    sep = SEPARATORS[unit]
    out = set()
    for context in df["context"].astype(str):
        out.add(context)
        out.add(sep.join(context.split(sep)[:-1]))
    out.discard("")
    return out


def restrict_counts(counts, contexts, sep):
    """
    Drop the pronunciations that cannot affect any lookup, exactly.

    Both counts the calculator needs are subtree totals under a queried prefix,
    so a pronunciation that does not begin with one of those prefixes
    contributes to no numerator and no denominator. Dropping it leaves every
    queried node's count untouched.

    The one exception is the root, whose count is the whole lexicon and is the
    denominator for a word-initial bigram (empty context). That total is
    returned separately and restored after the tree is built.

    This is what makes French tractable: 7.2M pronunciations build a tree that
    does not fit in memory, while the few thousand prefixes this corpus
    actually asks about do.
    """
    total = sum(counts.values())

    root = {}
    for context in contexts:
        node = root
        for unit in context.split(sep):
            node = node.setdefault(unit, {})
        node[True] = True                      # marks "a queried prefix ends here"

    kept = {}
    for key, count in counts.items():
        node = root
        for unit in key.split(sep):
            node = node.get(unit)
            if node is None:
                break
            if True in node:
                kept[key] = count
                break
    return kept, total


def build_calculator(spec, root, unit, contexts=None, verbose=True):
    """
    (calculator, unit_frequency_table) for one corpus and unit.

    Pass `contexts` (from `queried_contexts`) to build the restricted tree.
    Frequencies are always computed over the FULL lexicon, since unit frequency
    is a lexicon-wide quantity rather than a conditional one.
    """
    from utils.information_theoretic_calculation import InformationTheoreticCalculator

    sep = SEPARATORS[unit]
    counts = load_counts(spec, root, unit)
    freqs = unit_counts(counts, unit)
    n_full = len(counts)

    total = None
    if contexts:
        counts, total = restrict_counts(counts, contexts, sep)

    if verbose:
        kept = f"{len(counts):,} of {n_full:,}" if total is not None else f"{n_full:,}"
        print(f"{spec.name}/{unit}: {kept} pronunciations, "
              f"{len(freqs):,} distinct units")

    calc = InformationTheoreticCalculator(
        pronunciation_counts=counts,
        phone_counts=freqs,
        sep=sep,
    )
    if total is not None:
        # Restore the whole-lexicon total that restriction removed. Only the
        # root is affected; every queried node below it is already exact.
        calc.pronunciation_count_tree.count = total
    return calc, freqs


def prepare_context(df, spec, unit):
    """
    The context column as the calculator expects it.

    The phone analysis needs syllable boundaries injected; the syllable
    analysis already has them. Both are done on a copy: `format_context_with_
    boundaries` rewrites `word_id` in place, and a stage that re-read the frame
    afterwards would otherwise see mangled ids.
    """
    df = df.copy()
    if unit == "phone":
        from utils.data_processing import format_context_with_boundaries
        df = format_context_with_boundaries(df, lang=spec.lang)
    return df


def pos_entropy(calc, context):
    """
    Entropy over the next unit given `context` -- same quantity as
    InformationTheoreticCalculator.get_target_pos_entropy, computed from the
    context node's children instead of by scanning the whole unit inventory.

    The two agree exactly: a unit that never follows this context contributes a
    zero-probability term, which the original loop skips anyway. The difference
    is cost. With 40 phones the scan is free; with ~20,000 distinct syllables it
    is 20,000 tree descents per row, which is why the syllable stage never
    finished. Here it is one dictionary iteration over the handful of units that
    actually occur.
    """
    import numpy as np

    units = str(context).split(calc.sep) if context else []
    node = calc.pronunciation_count_tree
    for u in units:
        node = node.followings.get(u)
        if node is None:
            return None
    if node.count == 0:
        return None

    total = node.count
    entropy = 0.0
    for child in node.followings.values():
        if child.count:
            p = child.count / total
            entropy -= p * np.log(p)
    return entropy


def surprisal_table(spec, df, root, unit="phone", verbose=True):
    """
    Bigram surprisal, positional entropy and unit frequencies, one row per row
    of `df` and in the same order, ready to be written beside it.

    Surprisal is -log P(unit1 unit2 | context before unit1): the context passed
    to the calculator drops its last element, because `context` always ends
    with unit1 and unit1 is part of the target.
    """
    sep = SEPARATORS[unit]
    df = prepare_context(df, spec, unit)
    calc, freqs = build_calculator(
        spec, root, unit, contexts=queried_contexts(df, unit), verbose=verbose)

    # Row-wise apply with a tqdm bar. (tqdm's pandas integration, `progress_apply`,
    # breaks on newer pandas: it imports the removed pandas.core.common.is_builtin_func.)
    def progress_apply(frame, fn, desc):
        rows = frame.to_dict("records")
        return pd.Series([fn(r) for r in tqdm.tqdm(rows, desc=desc, disable=not verbose)],
                         index=frame.index)

    def preceding(context):
        return sep.join(str(context).split(sep)[:-1])

    out = pd.DataFrame(index=df.index)
    out["bigram_surprisal"] = progress_apply(df,
        lambda r: calc.get_contextual_surprisal(
            preceding(r["context"]), f"{r[f'{unit}1']}{sep}{r[f'{unit}2']}"),
        desc=f"{spec.name}/{unit} surprisal")
    out[f"{unit}2_pos_entropy"] = progress_apply(df,
        lambda r: pos_entropy(calc, r["context"]), desc=f"{spec.name}/{unit} entropy")
    out[f"{unit}1_freq"] = df[f"{unit}1"].map(freqs)
    out[f"{unit}2_freq"] = df[f"{unit}2"].map(freqs)

    if verbose:
        n = len(out)
        miss = out["bigram_surprisal"].isna().sum()
        print(f"  {n - miss:,}/{n:,} rows have a surprisal value "
              f"({miss:,} unseen in the lexicon)")
    return out.reset_index(drop=True)
