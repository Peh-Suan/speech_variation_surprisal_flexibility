from .ipa_processing import (
    get_ipas,
    normalize_ipa,
    PHONE_NORMALIZATION,
    be_mfa_phone_to_phoneme,
    is_ae_non_speech,
    is_ae_arpabet_phone
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
):
    """
    Convert an alignment file / dataframe into a standardized phone-pair
    dataframe.

    Parameters
    ----------
    source :
        For fr/be:
            A TextGrid object, or a path to a TextGrid.

        For ae:
            Either a pandas DataFrame containing phone-level ARPAbet
            annotations, or a TextGrid object (MRI75 / MRI75-annotated).

    lang : {"fr", "be", "ae"}
        fr = French
        be = British English
        ae = American English

    pronunciation_dict : dict, optional
        Required by ipa_processing.word2ipa() for fr/be.

    preferred_ipas_f : str, optional
        Required by ipa_processing.word2ipa() for ae.

    start_col, end_col, arpa_col, word_col, word_id_col :
        Column names/indices for American-English dataframe input.

        If `word_id_col` is omitted and the input dataframe has no
        ``word_id`` column, token IDs are inferred from contiguous changes
        in the word label. TextGrid inputs receive word IDs directly from
        their word intervals.

        If omitted, these are inferred:
            named format:
                "start", "end", "arpa", "word"

            old ae_ema / ae_mri format:
                0, 1, 2, 3

    Returns
    -------
    pandas.DataFrame
        Columns:

        phone1_start
        phone1_end
        phone2_start
        phone2_end
        phone1
        phone2
        context
        word
        word_id

    Notes
    -----
    `context` contains all preceding phones in the current word before
    `phone2`. Therefore the last phone in `context` is always `phone1`.
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
    # Produce standardized phone-pair dataframe
    # ============================================================

    data = {
        "phone1_start": [],
        "phone1_end": [],
        "phone2_start": [],
        "phone2_end": [],
        "phone1": [],
        "phone2": [],
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

        # --------------------------------------------------------
        # Generate:
        #
        #   p1 p2
        #   p2 p3
        #   p3 p4
        #   ...
        #
        # context includes everything before phone2.
        # --------------------------------------------------------

        for i in range(1, len(phones)):

            phone1 = phones[i - 1]
            phone2 = phones[i]

            phone1_start, phone1_end = times[i - 1]
            phone2_start, phone2_end = times[i]

            context = "_".join(
                phones[:i]
            )

            data["phone1_start"].append(
                phone1_start
            )
            data["phone1_end"].append(
                phone1_end
            )

            data["phone2_start"].append(
                phone2_start
            )
            data["phone2_end"].append(
                phone2_end
            )

            data["phone1"].append(
                phone1
            )
            data["phone2"].append(
                phone2
            )

            data["context"].append(
                context
            )

            data["word"].append(
                word
            )

            data["word_id"].append(
                word_id
            )

    return pd.DataFrame(
        data,
        columns=[
            "phone1_start",
            "phone1_end",
            "phone2_start",
            "phone2_end",
            "phone1",
            "phone2",
            "context",
            "word",
            "word_id",
        ],
    )