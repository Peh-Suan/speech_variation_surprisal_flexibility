from .ipa_processing import (
    get_ipas,
    normalize_ipa,
    PHONE_NORMALIZATION,
    be_mfa_phone_to_phoneme,
    is_ae_non_speech,
    is_ae_arpabet_phone,
    syllabify_ipa,
)

SAMPA2IPA = {
        "2": "ø",
        "9": "œ",
        "@": "ə",
        "E": "ɛ",
        "E/": ("e", "ɛ"),        # close or open-mid front
        "H": "ɥ",
        "J": "ɲ",
        "O": "ɔ",
        "O/": ("o", "ɔ"),        # close or open-mid back
        "R": "ʁ",
        "S": "ʃ",
        "U~/": ("ɛ̃", "œ̃"),     # indeterminate nasal front-mid
        "Z": "ʒ",
        "a": "a",
        "a~": "ɑ̃",
        "b": "b",
        "d": "d",
        "e": "e",
        "f": "f",
        "g": "g",
        "i": "i",
        "j": "j",
        "k": "k",
        "l": "l",
        "m": "m",
        "n": "n",
        "o": "o",
        "o~": "ɔ̃",
        "p": "p",
        "s": "s",
        "t": "t",
        "u": "u",
        "v": "v",
        "w": "w",
        "y": "y",
        "z": "z",
    }
def alignment_to_dataframe(
    source,
    lang,
    *,
    pronunciation_dict=None,
    preferred_ipas_f=None,
    start_col=None,
    end_col=None,
    arpa_col=None,
    word_col=None,
    word_id_col=None,
    unit_type="phoneme",
):
    """
    Convert phone-level alignment data into a standardized contextual
    dataframe at either the phoneme or syllable level.

    Parameters
    ----------
    source :
        Input alignment data.

        For ``fr`` and ``be``:
            A TextGrid object containing word- and phone-level tiers.

        For ``ae``:
            Either a pandas DataFrame containing phone-level ARPAbet
            annotations, or a TextGrid object containing word- and
            phone-level tiers.

    lang : {"fr", "be", "ae"}
        Language / variety:

        - ``"fr"``: French
        - ``"be"``: British English
        - ``"ae"``: American English

    pronunciation_dict : dict, optional
        Pronunciation dictionary used by ``ipa_processing.word2ipa()``
        for French and British English.

    preferred_ipas_f : str, optional
        Path to the preferred-IPA file used by
        ``ipa_processing.word2ipa()`` for American English.

    start_col, end_col, arpa_col, word_col, word_id_col : str or int, optional
        Column names or indices used for American-English dataframe input.

        If omitted, the function attempts to infer the following formats:

        Named-column format:
            ``"start"``, ``"end"``, ``"arpa"``, ``"word"``

        Legacy AE EMA / MRI format:
            columns ``0``, ``1``, ``2``, ``3``

        ``word_id_col`` identifies individual word tokens. If it is omitted
        and the dataframe has no ``word_id`` column, word-token IDs are
        inferred from contiguous changes in the word label.

        TextGrid inputs receive word-token IDs directly from their word
        intervals.

    unit_type : {"phoneme", "syllable"}, default="phoneme"
        Linguistic unit used to construct the output dataframe.

        If ``"phoneme"``, each output row represents an adjacent phoneme
        pair within a word:

            phone1 -> phone2

        If ``"syllable"``, the aligned phone sequence is first syllabified
        with ``ipa_processing.syllabify_ipa()``, and each output row
        represents an adjacent syllable pair within a word:

            syllable1 -> syllable2

        Syllable start and end times are derived from the first and last
        phone intervals belonging to each syllable.

    Returns
    -------
    pandas.DataFrame
        A contextual dataframe containing adjacent-unit pairs within each
        word token.

        If ``unit_type="phoneme"``, the columns are:

            phone1_start
            phone1_end
            phone2_start
            phone2_end
            phone1
            phone2
            context
            word
            word_id

        If ``unit_type="syllable"``, the columns are:

            syllable1_start
            syllable1_end
            syllable2_start
            syllable2_end
            syllable1
            syllable2
            context
            word
            word_id

    Notes
    -----
    The function first reconciles the observed phone alignment with the
    lexical pronunciation of each word token. Timing information remains
    tied to the original phone intervals.

    ``context`` contains all units preceding the target unit within the
    current word.

    For phoneme output, phones in ``context`` are separated by ``"_"``.
    Therefore, the final phone in ``context`` is always ``phone1``.

    For syllable output, syllables in ``context`` are separated by ``"."``,
    while phones within each syllable remain separated by ``"_"``.
    Therefore, the final syllable in ``context`` is always ``syllable1``.

    Monophonemic words do not contribute rows to phoneme-level output, and
    monosyllabic words do not contribute rows to syllable-level output,
    because there is no adjacent unit pair to represent.
    """

    import os
    import pandas as pd

    if lang not in {"fr", "be", "ae"}:
        raise ValueError(
            "lang must be 'fr', 'be', or 'ae'"
        )

    # ============================================================
    # Helpers
    # ============================================================

    def ae_textgrid_to_dataframe(tgrd):
        """
        Convert an AE MRI75/MRI75-annotated TextGrid into the canonical
        phone-level dataframe expected by the AE pipeline.
        """

        data = {
            "word": [],
            "word_id": [],
            "arpa": [],
            "start": [],
            "end": [],
        }

        def overlap_ratio(a_start, a_end, b_start, b_end):
            """
            Ratio of the portion of interval A overlapping B to the
            portion of A not overlapping B.
            """

            len_a = a_end - a_start

            if len_a <= 0:
                raise ValueError(
                    "Phone interval must have positive length."
                )

            overlap_start = max(a_start, b_start)
            overlap_end = min(a_end, b_end)
            overlap_len = max(
                0,
                overlap_end - overlap_start,
            )

            nonoverlap_len = len_a - overlap_len

            if nonoverlap_len == 0:
                return float("inf")

            return overlap_len / nonoverlap_len


        def is_in_word(phone_start, phone_end, word_start, word_end):
            return (
                overlap_ratio(
                    phone_start,
                    phone_end,
                    word_start,
                    word_end,
                )
                >= 1
            )


        for word_id, word_interval in enumerate(tgrd.tiers[0]):

            word = word_interval.text

            if not word:
                continue

            word_start = word_interval.start_time
            word_end = word_interval.end_time

            phone_intervals = [
                interval
                for interval in tgrd.tiers[1]
                if is_in_word(
                    interval.start_time,
                    interval.end_time,
                    word_start,
                    word_end,
                )
            ]

            for interval in phone_intervals:

                # Preserve your original MRI75 behavior
                arpa = interval.text.replace("'", "")

                if not arpa.strip():
                    continue

                if is_ae_non_speech(arpa):
                    continue

                if not is_ae_arpabet_phone(arpa):
                    continue

                data["word"].append(word)
                data["word_id"].append(word_id)
                data["arpa"].append(arpa)
                data["start"].append(interval.start_time)
                data["end"].append(interval.end_time)

        return pd.DataFrame(data)
    
    def clean_word(word):
        """Corpus-specific word-label cleanup."""

        word = str(word)

        if lang == "fr":
            return (
                word
                .replace(".", "")
                .replace(":", "")
                .replace("#", "")
                .lower()
            )

        return word

    # Each record represents one word token:
    #
    # {
    #     "word": ...,
    #     "word_id": ...,
    #     "phones": [...],
    #     "times": [(start, end), ...]
    # }
    #
    word_records = []

    # ============================================================
    # French / British English:
    # TextGrid input
    # ============================================================

    if lang in {"fr", "be"}:

        # --------------------------------------------------------
        # Read TextGrid if a path was supplied
        # --------------------------------------------------------

        tgrd = source

        # --------------------------------------------------------
        # Canonical phone-level dataframe for get_ipas()
        # --------------------------------------------------------

        base_data = {
            "word": [],
            "word_id": [],
            "original_ipa": [],
            "start": [],
            "end": [],
        }

        for word_id, word_interval in enumerate(tgrd.tiers[0]):

            word = clean_word(word_interval.text)

            if not word:
                continue

            start_time = word_interval.start_time
            end_time = word_interval.end_time

            phone_intervals = [
                interval
                for interval in tgrd.tiers[1]
                if start_time <= interval.start_time < end_time
            ]

            if not phone_intervals:
                continue

            phones = []
            times = []

            for interval in phone_intervals:

                raw_phone = interval.text

                # ------------------------------------------------
                # French
                # ------------------------------------------------

                if lang == "fr":

                    # Preserve your original French behavior:
                    # phones ending in "/" are already special labels.
                    if raw_phone.endswith("/"):
                        phone = raw_phone
                    else:
                        phone = SAMPA2IPA[raw_phone]

                    # Original FR-MRI transformation
                    phone = PHONE_NORMALIZATION[lang].get(phone, phone)

                    alignment_phone = phone

                # ------------------------------------------------
                # British English
                # ------------------------------------------------

                else:

                    if raw_phone.strip().lower() == "spn":
                        continue
                    
                    alignment_phone = be_mfa_phone_to_phoneme(raw_phone)

                phones.append(alignment_phone)

                times.append(
                    (
                        interval.start_time,
                        interval.end_time,
                    )
                )

                base_data["word"].append(word)
                base_data["word_id"].append(word_id)
                base_data["original_ipa"].append(
                    alignment_phone
                )
                base_data["start"].append(
                    interval.start_time
                )
                base_data["end"].append(
                    interval.end_time
                )

            word_records.append(
                {
                    "word": word,
                    "word_id": word_id,
                    "phones": phones,
                    "times": times,
                }
            )

        base_df = pd.DataFrame(base_data)

        # --------------------------------------------------------
        # Align whole-word pronunciations
        # --------------------------------------------------------

        aligned_ipas, converted_ipas, aligned_words, alignments = (
            get_ipas(
                base_df,
                lang=lang,
                pronunciation_dict=pronunciation_dict,
                word_id_col="word_id",
            )
        )

    # ============================================================
    # American English:
    # dataframe / CSV-derived input
    # ============================================================

    else:

        from phonecodes import phonecodes

        if isinstance(source, pd.DataFrame):

            df = source.copy()

        else:

            tgrd = source
            df = ae_textgrid_to_dataframe(tgrd)

        # --------------------------------------------------------
        # Infer dataframe format
        # --------------------------------------------------------

        if start_col is None:
            start_col = (
                "start"
                if "start" in df.columns
                else 0
            )

        if end_col is None:
            end_col = (
                "end"
                if "end" in df.columns
                else 1
            )

        if arpa_col is None:
            arpa_col = (
                "arpa"
                if "arpa" in df.columns
                else 2
            )

        if word_col is None:
            word_col = (
                "word"
                if "word" in df.columns
                else 3
            )

        # --------------------------------------------------------
        # Ensure every phone row has a word-token ID
        # --------------------------------------------------------

        if word_id_col is None:
            word_id_col = "word_id"

        if word_id_col not in df.columns:
            inferred_word_ids = []
            current_label = None
            current_id = -1

            for value in df[word_col]:
                if pd.isna(value):
                    inferred_word_ids.append(pd.NA)
                    continue

                label = str(value).strip()

                if not label:
                    inferred_word_ids.append(pd.NA)
                    continue

                if current_label is None or label != current_label:
                    current_id += 1
                    current_label = label

                inferred_word_ids.append(current_id)

            df[word_id_col] = inferred_word_ids

        # --------------------------------------------------------
        # Build word-token records
        # --------------------------------------------------------

        current_word = None
        current_word_id = None
        current_phones = []
        current_times = []

        def finish_word():
            nonlocal current_word
            nonlocal current_word_id
            nonlocal current_phones
            nonlocal current_times

            if current_word_id is None or not current_phones:
                return

            word_records.append(
                {
                    "word": current_word,
                    "word_id": current_word_id,
                    "phones": current_phones,
                    "times": current_times,
                }
            )

        for start, end, arpa, word, word_id in zip(
            df[start_col],
            df[end_col],
            df[arpa_col],
            df[word_col],
            df[word_id_col],
        ):

            if pd.isna(word) or pd.isna(arpa):
                continue

            word = str(word).strip()
            arpa = str(arpa).strip()

            if not word or not arpa:
                continue

            if is_ae_non_speech(arpa):
                continue

            if pd.isna(word_id):
                raise ValueError(
                    f"Missing word_id for word {word!r} with phone {arpa!r}."
                )

            ipa = phonecodes.arpabet2ipa(
                arpa.upper()
            )

            ipa = normalize_ipa(ipa, lang=lang)

            if "ˈ" in ipa:
                raise ValueError(
                    f"Stress marker still in IPA: {ipa}"
                )

            # ----------------------------------------------------
            # New word token
            # ----------------------------------------------------

            if current_word_id is None or word_id != current_word_id:

                finish_word()

                current_word = word
                current_word_id = word_id
                current_phones = [ipa]
                current_times = [(start, end)]

            else:

                if word != current_word:
                    raise ValueError(
                        f"word_id {word_id!r} spans multiple word labels: "
                        f"{current_word!r} and {word!r}."
                    )

                current_phones.append(ipa)
                current_times.append((start, end))

        # Do not lose the final word
        finish_word()

        # --------------------------------------------------------
        # Align whole-word pronunciations
        # --------------------------------------------------------

        aligned_ipas, converted_ipas, aligned_words, alignments = (
            get_ipas(
                df,
                lang="ae",
                preferred_ipas_f=preferred_ipas_f,
                arpa_col=arpa_col,
                word_col=word_col,
                word_id_col=word_id_col,
            )
        )

    # ============================================================
    # Keep only words capable of contributing a phone pair
    # ============================================================

    word_records = [
        record
        for record in word_records
        if len(record["phones"]) > 1
    ]

    # ============================================================
    # Sanity check:
    # get_ipas output should correspond one-to-one to word records
    # ============================================================

    if len(word_records) != len(aligned_ipas):
        raise ValueError(
            "Mismatch between word records and get_ipas() output:\n"
            f"  word records: {len(word_records)}\n"
            f"  aligned IPAs: {len(aligned_ipas)}"
        )

    # ============================================================
    # Produce standardized phoneme- or syllable-pair dataframe
    # ============================================================

    unit_name = (
        "phone"
        if unit_type == "phoneme"
        else "syllable"
    )

    context_sep = (
        "_"
        if unit_type == "phoneme"
        else "."
    )

    data = {
        f"{unit_name}1_start": [],
        f"{unit_name}1_end": [],
        f"{unit_name}2_start": [],
        f"{unit_name}2_end": [],
        f"{unit_name}1": [],
        f"{unit_name}2": [],
        "context": [],
        "word": [],
        "word_id": [],
    }

    for record, aligned_word, aligned_ipa in zip(
        word_records,
        aligned_words,
        aligned_ipas,
    ):

        word = record["word"]
        word_id = record["word_id"]
        times = record["times"]

        # --------------------------------------------------------
        # Make sure get_ipas() and the interval grouping agree
        # --------------------------------------------------------

        if word != aligned_word:
            raise ValueError(
                "Word mismatch between alignment data and "
                "get_ipas():\n"
                f"  alignment word: {word!r}\n"
                f"  get_ipas word: {aligned_word!r}"
            )

        phones = aligned_ipa.split("_")

        if len(phones) != len(times):
            raise ValueError(
                f"Phone/timing mismatch for {word!r}:\n"
                f"  IPA: {aligned_ipa}\n"
                f"  phones: {len(phones)}\n"
                f"  intervals: {len(times)}"
            )

        # ========================================================
        # PHONEME
        # ========================================================

        if unit_type == "phoneme":

            unit_records = [
                {
                    "label": phone,
                    "start": start,
                    "end": end,
                }
                for phone, (start, end) in zip(
                    phones,
                    times,
                )
            ]

        # ========================================================
        # SYLLABLE
        # ========================================================

        else:

            # Example:
            #
            #   m_æ_s_k_ə_ɹ_eɪ_d
            #
            # might become:
            #
            #   m_æ_s.k_ə.ɹ_eɪ_d
            #

            syllabified = syllabify_ipa(
                aligned_ipa,
                lang=lang,
                sep="_",
            )

            syllable_phone_lists = [
                syllable.split("_")
                for syllable in syllabified.split(".")
                if syllable
            ]

            # ----------------------------------------------------
            # Strong sanity check:
            #
            # Syllabification must ONLY add syllable boundaries.
            # It must not add, remove, reorder, or change phones.
            # ----------------------------------------------------

            reconstructed_phones = [
                phone
                for syllable in syllable_phone_lists
                for phone in syllable
            ]

            if reconstructed_phones != phones:
                raise ValueError(
                    "Syllabification changed the phone sequence.\n"
                    f"  word: {word!r}\n"
                    f"  original: {aligned_ipa}\n"
                    f"  syllabified: {syllabified}\n"
                    f"  original phones: {phones}\n"
                    f"  reconstructed phones: "
                    f"{reconstructed_phones}"
                )

            # ----------------------------------------------------
            # Aggregate phone timings into syllable timings
            # ----------------------------------------------------

            unit_records = []

            phone_position = 0

            for syllable_phones in syllable_phone_lists:

                n_phones = len(
                    syllable_phones
                )

                first_phone_idx = (
                    phone_position
                )

                last_phone_idx = (
                    phone_position
                    + n_phones
                    - 1
                )

                syllable_start = (
                    times[first_phone_idx][0]
                )

                syllable_end = (
                    times[last_phone_idx][1]
                )

                unit_records.append(
                    {
                        "label": "_".join(
                            syllable_phones
                        ),
                        "start": syllable_start,
                        "end": syllable_end,
                    }
                )

                phone_position += n_phones

        # ========================================================
        # Generate adjacent unit pairs
        #
        # phoneme:
        #
        #   p1 -> p2
        #   p2 -> p3
        #   ...
        #
        # syllable:
        #
        #   σ1 -> σ2
        #   σ2 -> σ3
        #   ...
        #
        # context contains everything before unit2.
        # ========================================================

        for i in range(
            1,
            len(unit_records),
        ):

            unit1 = unit_records[i - 1]
            unit2 = unit_records[i]

            cumulative_context = (
                context_sep.join(
                    unit["label"]
                    for unit in unit_records[:i]
                )
            )

            data[
                f"{unit_name}1_start"
            ].append(
                unit1["start"]
            )

            data[
                f"{unit_name}1_end"
            ].append(
                unit1["end"]
            )

            data[
                f"{unit_name}2_start"
            ].append(
                unit2["start"]
            )

            data[
                f"{unit_name}2_end"
            ].append(
                unit2["end"]
            )

            data[
                f"{unit_name}1"
            ].append(
                unit1["label"]
            )

            data[
                f"{unit_name}2"
            ].append(
                unit2["label"]
            )

            data["context"].append(
                cumulative_context
            )

            data["word"].append(
                word
            )

            data["word_id"].append(
                word_id
            )

    output_columns = [
        f"{unit_name}1_start",
        f"{unit_name}1_end",
        f"{unit_name}2_start",
        f"{unit_name}2_end",
        f"{unit_name}1",
        f"{unit_name}2",
        "context",
        "word",
        "word_id",
    ]

    return pd.DataFrame(
        data,
        columns=output_columns,
    )


def format_context_with_boundaries(df, lang):
    """
    Injects syllable boundaries (_._) into the context string.
    """
    import os
    df["word_id"] = df.apply(lambda x: f"{os.path.basename(x['art_f'])}_{x['word_id']}", axis=1)
    
    def get_full_ipa(group):
        idx = group['context'].str.len().argmax() 
        last_row = group.iloc[idx] 
        return f"{last_row['context']}_{last_row['phone2']}"
    
    try:
        full_ipas = df.groupby('word_id').apply(get_full_ipa, include_groups=False).to_dict()
    except TypeError:
        full_ipas = df.groupby('word_id').apply(get_full_ipa).to_dict()
        
    syllabified_words = {occ_id: syllabify_ipa(ipa, lang=lang) for occ_id, ipa in full_ipas.items()}
    
    def inject_boundaries(row):
        occ_id = row['word_id']
        context_len = len(row['context'].split('_')) 
        
        syllabified = syllabified_words[occ_id]
        tokens = syllabified.replace('.', '_._').split('_')
        
        collected_tokens = []
        phoneme_count = 0
        
        # FIXED: Using enumerate so we know our current index (i)
        for i, token in enumerate(tokens):
            collected_tokens.append(token)
            
            if token != '.':  
                phoneme_count += 1
                
            if phoneme_count == context_len:
                # PEEK AHEAD: If the next token in the list is a '.', grab it!
                if i + 1 < len(tokens) and tokens[i+1] == '.':
                    collected_tokens.append('.')
                break
                
        return '_'.join(collected_tokens)
        
    df['context'] = df.apply(inject_boundaries, axis=1)

    return df

import pandas as pd

def add_boundary_column(df, lang):
    """
    Takes a DataFrame with sequential phoneme data and adds a 'cross_syllable_boundary' column.
    Safely handles duplicate indices and multiple occurrences of the same word.
    """
    # 1. Create a unique ID for each continuous sequence of a word.
    # This prevents merging Subject A's "the" with Subject B's "the".
    import os
    df["word_id"] = df.apply(lambda x: f"{os.path.basename(x['art_f'])}_{x['word_id']}", axis=1)
    
    # Helper to reconstruct the full IPA word for THIS SPECIFIC occurrence
    def get_full_ipa(group):
        # Using argmax and iloc completely avoids duplicate-index crashes
        idx = group['context'].str.len().argmax()
        last_row = group.iloc[idx]
        return f"{last_row['context']}_{last_row['phone2']}"
    
    # 2. Find the full IPA sequence for each unique occurrence
    full_ipas = df.groupby('word_id').apply(get_full_ipa).to_dict()
    
    # 3. Pre-calculate the exact numeric positions of boundaries
    boundary_maps = {}
    for occ_id, ipa in full_ipas.items():
        syllabified = syllabify_ipa(ipa, lang=lang)
        syllables = syllabified.split('.')
        
        boundary_positions = set()
        current_pos = 0
        
        # We don't check the last syllable because it's the end of the word
        for i in range(len(syllables) - 1): 
            num_phonemes = len(syllables[i].split('_'))
            current_pos += num_phonemes
            boundary_positions.add(current_pos)
            
        boundary_maps[occ_id] = boundary_positions

    # 4. Apply the logic to each row based on its specific occurrence
    def check_boundary(row):
        occ_id = row['word_id']
        phoneme_count = len(row['context'].split('_'))
        
        # Returns 1 if True, 0 if False
        return int(phoneme_count in boundary_maps[occ_id])
        
    df['cross_syllable_boundary'] = df.apply(check_boundary, axis=1)
    
    return df

def dataset_df_to_token_df(dataset_df, unit_type="phone"):

    tokens = {
        "start": [],
        "end": [],
        unit_type: [],
        "art_f": [],
        "ac_f": [],
        "subject": [],
    }

    metadata = set(
        [(start, end, target, art_f, ac_f, subject) for start, end, target, art_f, ac_f, subject in zip(dataset_df[f"{unit_type}1_start"], dataset_df[f"{unit_type}1_end"], dataset_df[f"{unit_type}1"], dataset_df["art_f"], dataset_df["ac_f"], dataset_df["subject"])] + 
        [(start, end, target, art_f, ac_f, subject) for start, end, target, art_f, ac_f, subject in zip(dataset_df[f"{unit_type}2_start"], dataset_df[f"{unit_type}2_end"], dataset_df[f"{unit_type}2"], dataset_df["art_f"], dataset_df["ac_f"], dataset_df["subject"])]
    )

    metadata = sorted(list(metadata), key=lambda x: (x[3], x[4], x[5]))

    for start, end, target, art_f, ac_f, subject in metadata:
        tokens["start"].append(start)
        tokens["end"].append(end)
        tokens[unit_type].append(target)
        tokens["art_f"].append(art_f)
        tokens["ac_f"].append(ac_f)
        tokens["subject"].append(subject)

    return pd.DataFrame(tokens)