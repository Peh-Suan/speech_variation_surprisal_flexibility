import re
import pickle

PHONE_NORMALIZATION = {
    "ae": {
        "ər": "ɝ",
        "r": "ɹ",
        "ə": "ʌ",
        "ʤ": "dʒ",
        "ʧ": "tʃ",
        "ɡ": "g",
    },

    "be": {
        "ɡ": "g",
        "ʤ": "dʒ",
        "ʧ": "tʃ",
        "r": "ɹ",

        # Canonical British-English vowel notation
        "ɝ": "ɜɹ",
        "ɚ": "əɹ",
        "aɪ": "aj",
        "əʊ": "ow",
        "əw": "ow",
        "oʊ": "ow",
        "ɔɪ": "ɔj",
        "eɪ": "ej",
        "aʊ": "aw",
    },

    "fr": {
        "ɡ": "g",
        "ɥ": "y",
    },
}

BE_MFA_TO_PHONEME = {
    "aː": "a",
    "æ": "a",

    "bʲ": "b",
    "dʲ": "d",
    "d̪": "d",
    "fʲ": "f",

    "iː": "i",

    "kʰ": "k",
    "kʷ": "k",

    "mʲ": "m",
    "m̩": "m",
    "n̩": "n",

    "pʰ": "p",
    "pʲ": "p",

    "tʰ": "t",
    "tʲ": "t",
    "t̪": "t",
    "tʷ": "t",

    "uː": "u",
    "vʲ": "v",

    "ɑː": "ɑ",
    "ɒː": "ɒ",
    "ɛː": "ɛ",
    "ɜː": "ɜ",

    "ɾ": "t",
    "ɾʲ": "t",
    "ɾ̃": "t",

    "ʎ": "l",
    "ɫ": "l",
    "ɫ̩": "l",

    "ʉ": "u",
    "ʉː": "u",

    "ɲ": "n",

    "ç": "h",

    "c": "k",
    "cʰ": "k",
    "cʷ": "k",

    "ɟ": "g",
    "ɟʷ": "g",

    "ɱ": "n",
}

AE_NON_SPEECH_LABELS = {
    "sil",
    "sp",
    "spn",
    "ga",
}

def is_ae_non_speech(phone):
    return str(phone).strip().lower() in AE_NON_SPEECH_LABELS

AE_ARPABET_PHONES = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH",
    "EH", "ER", "EY",
    "F", "G", "HH",
    "IH", "IY",
    "JH", "K", "L", "M", "N", "NG",
    "OW", "OY",
    "P", "R", "S", "SH",
    "T", "TH",
    "UH", "UW",
    "V", "W", "Y", "Z", "ZH",
}

def is_ae_arpabet_phone(phone):
    phone = str(phone).strip().upper()

    # Allow stress markers, e.g. AA1, IH0, AH2
    if phone and phone[-1] in {"0", "1", "2"}:
        phone = phone[:-1]

    return phone in AE_ARPABET_PHONES

def be_mfa_phone_to_phoneme(phone):
    """
    Convert one BE-US MFA phone interval to the canonical
    phonemic representation.

    This mapping is deliberately exact: one TextGrid interval
    must remain one phone.
    """
    phone = BE_MFA_TO_PHONEME.get(phone, phone)
    return normalize_ipa(phone, lang="be")

def normalize_ipa(ipa, lang):

    assert lang in ["ae", "be", "fr"], f"Language {lang} not supported for IPA normalization."

    deletions = ["'", '(', ')', '[', ']', 'ˈ', 'ˌ', 'ː']
    for deletion in deletions:
        ipa = ipa.replace(deletion, "")

    for replacement in PHONE_NORMALIZATION[lang].items():
        ipa = ipa.replace(replacement[0], replacement[1])

    ipa = ipa.strip()

    def separate(ipa, lang):
        import re
        long_symbols = {
            "ae": [
                "aɪ", "oʊ", "eɪ", "dʒ",
                "aʊ", "tʃ", "ɔɪ"
            ],

            "be": [
                "aj", "aw", "ej", "ow", "ɔj",
                "əw",
                "ɜɹ", "əɹ",
                "dʒ", "tʃ",
                "ju",
            ],

            "fr": [
                "ɛ̃", "ɑ̃", "ɔ̃", "œ̃"
            ],
        }

        single_symbols = r"[^\s_]"

        if long_symbols[lang]:
            pattern = "|".join(map(re.escape, long_symbols[lang])) + "|" + single_symbols
            ipa_list = re.findall(pattern, ipa)
        else:
            ipa_list = list(ipa)

        return ipa_list

    return "_".join(separate(ipa, lang))



def word2ipa(word, lang, pronunciation_dict=None, preferred_ipas_f=None):
    assert lang in ["ae", "be", "fr"], f"Language {lang} not supported for IPA conversion."

    match lang:
        case "ae":
            assert preferred_ipas_f is not None, "Preferred IPA file must be provided for American English."
            import eng_to_ipa
            preferred_ipas = pickle.load(open(preferred_ipas_f, "rb"))
            preferred_ipas = {k: normalize_ipa(v, "ae") for k, v in preferred_ipas.items()}

            def try_process(ipa):
                replacements = {
                    "ʌ": "ɪ",
                }
                for replacement in replacements.items():
                    ipa = ipa.replace(replacement[0], replacement[1])
                return ipa
            
            custom_replacements = {
                "him": "hɪm",
                "auburn": "ɔbɝn",
                "to": "tu",
                "are": "ɝ",
                "from": "fɝm"
            }
            
            if word in custom_replacements:
                ipa = normalize_ipa(custom_replacements[word], "ae")
            else:
                possible_ipas = [normalize_ipa(ipa, "ae") for ipa in eng_to_ipa.convert(word, retrieve_all=True)]
                max_len = max(len(ipa) for ipa in possible_ipas)
                possible_ipas = [ipa for ipa in possible_ipas if len(ipa) == max_len]

                if len(possible_ipas) == 1:
                    if (ipa := try_process(possible_ipas[0])) != preferred_ipas.get(word, None):
                        ipa = possible_ipas[0]

                elif word in preferred_ipas and preferred_ipas[word] in possible_ipas+[try_process(ipa) for ipa in possible_ipas]:
                    ipa = preferred_ipas[word]
                    
                else:
                    ipa = possible_ipas[0]
                    if try_process(ipa)==preferred_ipas.get(word, None):
                        ipa = preferred_ipas[word]
                    
            return ipa
        case "be" | "fr":
            assert pronunciation_dict is not None, f"Pronunciation dictionary must be provided for language {lang}."
            ipa = pronunciation_dict.get(word, None)
            if not ipa: return None
            ipa = normalize_ipa(ipa, lang)
    
            return ipa

def syllabify_ipa(ipa_str, lang, sep="_"):
    """
    Adds syllable boundaries to a separated IPA string.

    Parameters
    ----------
    ipa_str : str
        IPA transcription with phones separated by `sep`.
        Example: "m_æ_s_k_ə_ɹ_eɪ_d"

    lang : {"fr", "be", "ae"}
        Language/variety:
        - "fr" = French
        - "be" = British English
        - "ae" = American English

    sep : str, default="_"
        Separator between phones.

    Returns
    -------
    str
        Syllabified IPA string, with syllables separated by ".".
    """

    # ------------------------------------------------------------------
    # Language-specific phonological inventories
    # ------------------------------------------------------------------

    if lang == "fr":
        vowels = {
            "a", "ɑ", "e", "ɛ", "i", "o", "ɔ", "u", "y",
            "ø", "œ", "ə", "ɛ̃", "ɑ̃", "ɔ̃", "œ̃"
        }

        liquids = {"ʁ", "l"}
        obstruents = {"p", "b", "t", "d", "k", "g", "f", "v"}

    elif lang == "be":
        vowels = {
            "a", "aj", "aw", "e", "ej", "i", "o", "ow", "u", "ɐ",
            "ɑ", "ɒ", "ɔ", "ɔj", "ə", "əɹ", "ɛ", "ɜ", "ɜɹ", "ɪ", "ʊ"
        }

        valid_onsets_2 = {
            # Stop + Approximant
            "p_ɹ", "b_ɹ", "t_ɹ", "d_ɹ", "k_ɹ", "g_ɹ",
            "p_l", "b_l", "k_l", "g_l",
            "t_w", "d_w", "k_w", "g_w", "p_w", "b_w",
            "p_j", "b_j", "t_j", "d_j", "k_j", "g_j",
            "m_j", "n_j", "f_j", "v_j", "θ_j", "s_j", "h_j", "l_j",

            # Fricative + Approximant
            "f_ɹ", "θ_ɹ", "ʃ_ɹ",
            "f_l", "s_l",
            "s_w", "θ_w", "ʃ_w",

            # s + Nasal/Stop
            "s_m", "s_n", "s_p", "s_t", "s_k"
        }

        valid_onsets_3 = {
            "s_p_ɹ", "s_t_ɹ", "s_k_ɹ",
            "s_p_l", "s_k_l",
            "s_p_j", "s_t_j", "s_k_j",
            "s_k_w", "s_p_w"
        }

    elif lang == "ae":
        vowels = {
            "aɪ", "aʊ", "eɪ", "i", "oʊ", "u", "æ", "ɑ", "ɔ", "ɔɪ",
            "ɛ", "ɝ", "ɪ", "ʊ", "ʌ", "ə", "ɚ"
        }

        valid_onsets_2 = {
            # Stop + Approximant
            "p_ɹ", "b_ɹ", "t_ɹ", "d_ɹ", "k_ɹ", "g_ɹ",
            "p_l", "b_l", "k_l", "g_l",
            "t_w", "d_w", "k_w", "g_w", "p_w", "b_w",
            "p_j", "b_j", "t_j", "d_j", "k_j", "g_j",
            "m_j", "n_j", "f_j", "v_j", "θ_j", "s_j", "h_j", "l_j",

            # Fricative + Approximant
            "f_ɹ", "θ_ɹ", "ʃ_ɹ",
            "f_l", "s_l",
            "s_w", "θ_w", "ʃ_w",

            # s + Nasal/Stop
            "s_m", "s_n", "s_p", "s_t", "s_k"
        }

        valid_onsets_3 = {
            "s_p_ɹ", "s_t_ɹ", "s_k_ɹ",
            "s_p_l", "s_k_l",
            "s_p_j", "s_t_j", "s_k_j",
            "s_k_w", "s_p_w"
        }

    else:
        raise ValueError(
            f"Unknown language {lang!r}. "
            "Expected 'fr', 'be', or 'ae'."
        )

    # ------------------------------------------------------------------
    # Find vowel nuclei
    # ------------------------------------------------------------------

    phones = ipa_str.split(sep)
    v_idx = [i for i, phone in enumerate(phones) if phone in vowels]

    # No vowels -> treat the entire sequence as one syllable
    if not v_idx:
        return ipa_str

    syllables = []
    start = 0

    # ------------------------------------------------------------------
    # Syllabification
    # ------------------------------------------------------------------

    for i, curr_v in enumerate(v_idx):

        # Last vowel gets all remaining consonants
        if i == len(v_idx) - 1:
            syllables.append(phones[start:])
            break

        next_v = v_idx[i + 1]
        interlude = phones[curr_v + 1 : next_v]

        # --------------------------------------------------------------
        # French
        # --------------------------------------------------------------

        if lang == "fr":

            if len(interlude) <= 1:
                # V.V or V.CV
                split_point = 0

            elif len(interlude) == 2:
                # Obstruent + liquid can form a complex onset
                if (
                    interlude[0] in obstruents
                    and interlude[1] in liquids
                ):
                    split_point = 0
                else:
                    split_point = 1

            else:
                # For 3+ consonants, preserve an obstruent + liquid
                # cluster at the beginning of the next syllable
                if (
                    interlude[-2] in obstruents
                    and interlude[-1] in liquids
                ):
                    split_point = len(interlude) - 2
                else:
                    split_point = len(interlude) - 1

        # --------------------------------------------------------------
        # English
        # --------------------------------------------------------------

        else:

            if len(interlude) == 0:
                split_point = 0

            elif len(interlude) == 1:
                # ŋ cannot be an English onset
                split_point = 1 if interlude[0] == "ŋ" else 0

            else:
                # Try the largest valid onset first
                if (
                    len(interlude) >= 3
                    and sep.join(interlude[-3:]) in valid_onsets_3
                ):
                    split_point = len(interlude) - 3

                elif (
                    sep.join(interlude[-2:]) in valid_onsets_2
                ):
                    split_point = len(interlude) - 2

                else:
                    # Otherwise assign the final consonant to the onset,
                    # unless that consonant is ŋ
                    if interlude[-1] == "ŋ":
                        split_point = len(interlude)
                    else:
                        split_point = len(interlude) - 1

        # --------------------------------------------------------------
        # Create current syllable
        # --------------------------------------------------------------

        end = curr_v + 1 + split_point

        syllables.append(phones[start:end])
        start = end

    return ".".join(sep.join(syllable) for syllable in syllables)

def check_alignment(source_str, target_str, lang):
    """
    Align a phonetic IPA sequence (source) to a phonemic IPA sequence (target).

    Parameters
    ----------
    source_str : str
        Underscore-separated observed/phonetic IPA.
    target_str : str
        Underscore-separated target/phonemic IPA.
    lang : {"fr", "be", "ae"}
        fr = French
        be = British English
        ae = American English

    Returns
    -------
    aligned : bool
    steps : list
    converted_ipa : str
    """

    if lang not in {"fr", "be", "ae"}:
        raise ValueError("lang must be 'fr', 'be', or 'ae'")

    if not source_str or not target_str:
        return False, [], ""

    # ------------------------------------------------------------
    # Language-specific mappings
    # ------------------------------------------------------------

    if lang == "fr":
        mapping = {
            "O/": ["o", "ɔ"],
            "E/": ["e", "ɛ"],
            "U~/": ["ɛ̃", "œ̃"],
            "ɔ": ["o"],
            "o": ["ɔ"],
            "u": ["w"],
            "i": ["j"],
        }

    elif lang == "be":
        mapping = {
            "s": ["t", "z", "θ"],
            "z": ["s", "ð"],
            "ʌ": ["ɪ", "ɛ", "ɑ", "æ", "i", "ð"],
            "ɛ": ["æ"],
            "ə": ["a", "ɪ", "ɐ", "ɒ", "u"],
            "ɒ": ["ɔ"],
            "ɔ": ["ɒ"],
            "ɑ": ["a"],
            "d": ["ð"],
            "t": ["θ"],
            "ɪ": ["ʌ", "i"],
            "æ": ["ɛ", "a"],
            "tʃ": ["t"],
            "i": ["ɪ"],
            "e": ["ej"],
            "u": ["ʌ", "ʊ", "ju"],
            "ɹ": ["ɝ"],
            "θ": ["ð"],
            "ð": ["θ"],
            "ʊ": ["u", "ʌ"],
            "o": ["əw"],
            "ɐ": ["ʌ"],
        }

    else:  # ae
        # Union of the mappings in the four AE datasets.
        # ae_ema additionally contained ɛ -> æ.
        mapping = {
            "s": ["t", "z"],
            "z": ["s"],
            "ʌ": ["ɪ", "ɛ", "ɑ", "æ", "i"],
            "ɛ": ["æ"],
            "ɪ": ["ʌ", "i"],
            "æ": ["ɛ"],
            "i": ["ɪ"],
            "u": ["ʌ"],
            "ɹ": ["ɝ"],
            "θ": ["ð"],
            "ð": ["θ"],
            "ʊ": ["u", "ʌ"],
        }

    # ------------------------------------------------------------
    # Dynamic-programming alignment
    # ------------------------------------------------------------

    s = source_str.split("_")
    t = target_str.split("_")

    n = len(s)
    m = len(t)

    dp = [[False] * (m + 1) for _ in range(n + 1)]

    # Remaining target phones may be deleted.
    for j in range(m + 1):
        dp[n][j] = True

    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):

            can_match = (
                s[i] == t[j]
                or t[j] in mapping.get(s[i], [])
            )

            dp[i][j] = (
                (can_match and dp[i + 1][j + 1])
                or dp[i][j + 1]
            )

    if not dp[0][0]:
        return False, [], source_str

    # ------------------------------------------------------------
    # Backtrack
    # ------------------------------------------------------------

    i = 0
    j = 0

    steps = []
    converted = []

    while i < n or j < m:

        # No source phones left:
        # remaining target phones are deleted
        if i == n:
            steps.append(
                ("DELETE_TARGET", None, t[j])
            )
            j += 1
            continue

        can_match = (
            j < m
            and (
                s[i] == t[j]
                or t[j] in mapping.get(s[i], [])
            )
            and dp[i + 1][j + 1]
        )

        if can_match:
            steps.append(
                (
                    "MATCH" if s[i] == t[j] else "MAP",
                    s[i],
                    t[j],
                )
            )

            converted.append(t[j])

            i += 1
            j += 1
            continue

        if j < m and dp[i][j + 1]:
            steps.append(
                ("DELETE_TARGET", None, t[j])
            )
            j += 1
            continue

        # Should not normally happen if dp[0][0] is True
        break

    return True, steps, "_".join(converted)

def force_align(word, ipa, lang):
    """
    Apply hard-coded language-specific alignment exceptions.
    """

    if lang not in {"fr", "be", "ae"}:
        raise ValueError("lang must be 'fr', 'be', or 'ae'")

    word = word.lower()

    # ------------------------------------------------------------
    # French
    # ------------------------------------------------------------

    if lang == "fr":
        return False, ipa

    # ------------------------------------------------------------
    # British English
    # ------------------------------------------------------------

    if lang == "be":

        mapping = {
            "to": "t_u",
            "of": "ɒ_v",
        }

        if word in mapping:
            return True, mapping[word]

        return False, ipa

    # ------------------------------------------------------------
    # American English
    # ------------------------------------------------------------

    mapping = {
        "to": "t_u",
    }

    if word in mapping:

        aligned, _, out = check_alignment(
            ipa,
            mapping[word],
            lang="ae",
        )

        return aligned, out

    return False, ipa

def get_ipas(
    df,
    lang,
    pronunciation_dict=None,
    preferred_ipas_f=None,
    ipa_col="original_ipa",
    word_col="word",
    arpa_col=None,
    word_id_col="word_id",
):
    """
    Group phone-level rows into word tokens and reconcile observed IPA
    with ipa_processing.word2ipa().

    Parameters
    ----------
    df : pandas.DataFrame
        Phone-level dataframe.

    lang : {"fr", "be", "ae"}
        fr = French
        be = British English
        ae = American English

    pronunciation_dict : dict, optional
        Required for fr/be and passed to word2ipa().

    preferred_ipas_f : str, optional
        Required for ae and passed to word2ipa().

    ipa_col : str, default="original_ipa"
        Phone-level IPA column for French/British English.

    word_col : str or int, default="word"
        Word column. For old AE dataframes using integer column names,
        column 3 is automatically used if "word" is absent.

    arpa_col : str or int, optional
        ARPAbet column for AE. If None, uses "arpa" if present,
        otherwise column 2.

    word_id_col : str or int or None, default="word_id"
        Column identifying individual word tokens. Phones sharing a word
        token must share the same ID. If the requested column is absent (or
        word_id_col is None), IDs are inferred from contiguous changes in
        the word label for backward compatibility.

    Returns
    -------
    observed_ipas : list
    converted_ipas : list
    words : list
    alignments : list
    """

    import pandas as pd
    from phonecodes import phonecodes

    if lang not in {"fr", "be", "ae"}:
        raise ValueError("lang must be 'fr', 'be', or 'ae'")

    # ------------------------------------------------------------
    # Check word2ipa requirements
    # ------------------------------------------------------------

    if lang in {"fr", "be"} and pronunciation_dict is None:
        raise ValueError(
            f"pronunciation_dict is required for lang={lang!r}"
        )

    if lang == "ae" and preferred_ipas_f is None:
        raise ValueError(
            "preferred_ipas_f is required for lang='ae'"
        )

    # ------------------------------------------------------------
    # word -> phonemic IPA
    # ------------------------------------------------------------

    def word_to_ipa(word):

        if lang in {"fr", "be"}:
            return word2ipa(
                word,
                lang=lang,
                pronunciation_dict=pronunciation_dict,
            )

        return word2ipa(
            word,
            lang="ae",
            preferred_ipas_f=preferred_ipas_f,
        )

    # ------------------------------------------------------------
    # Language-specific secondary alignment attempts
    # ------------------------------------------------------------

    def try_align(word, ipa, converted_ipa):

        if not ipa or not converted_ipa:
            return False, [], ipa

        word = word.lower()

        if lang == "fr":
            candidates = [
                converted_ipa + "_ə",
                converted_ipa + "_" + ipa.split("_")[-1],
            ]

        elif lang == "be":
            candidates = {
                "the": ["ð_ə", "ð_i"],
            }.get(word, [])

        else:
            return False, [], ipa

        for possible_ipa in candidates:
            aligned, steps, out = check_alignment(
                ipa,
                possible_ipa,
                lang=lang,
            )

            if aligned:
                return True, steps, out

        return False, [], ipa

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    observed_ipas = []
    converted_ipas = []
    words = []
    alignments = []

    # ------------------------------------------------------------
    # Determine input columns
    # ------------------------------------------------------------

    if lang == "ae":

        if arpa_col is None:
            if "arpa" in df.columns:
                arpa_col = "arpa"
            else:
                arpa_col = 2

        if word_col not in df.columns:
            if 3 in df.columns:
                word_col = 3
            else:
                raise KeyError(
                    f"Could not find word column "
                    f"{word_col!r} or fallback column 3."
                )

        phone_series = df[arpa_col]

    else:

        if ipa_col not in df.columns:
            raise KeyError(
                f"Could not find IPA column {ipa_col!r}."
            )

        if word_col not in df.columns:
            raise KeyError(
                f"Could not find word column {word_col!r}."
            )

        phone_series = df[ipa_col]

    # ------------------------------------------------------------
    # Determine / infer word-token IDs
    # ------------------------------------------------------------

    if word_id_col is not None and word_id_col in df.columns:
        word_id_series = df[word_id_col]

    else:
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

        word_id_series = inferred_word_ids

    phone_iter = zip(
        phone_series,
        df[word_col],
        word_id_series,
    )

    # ------------------------------------------------------------
    # Group phones by word token (word_id), not by word spelling
    # ------------------------------------------------------------

    current_word = None
    current_word_id = None
    context = []

    def add_data():

        nonlocal current_word, current_word_id, context

        if current_word_id is None or not context:
            return

        original_len = len(context)

        observed_ipa = "_".join(context)
        converted_ipa = word_to_ipa(current_word)

        # --------------------------------------------------------
        # Attempt alignment
        # --------------------------------------------------------

        if observed_ipa != converted_ipa:

            alignable, _, aligned_ipa = check_alignment(
                observed_ipa,
                converted_ipa,
                lang=lang,
            )

            if not alignable:
                alignable, _, aligned_ipa = try_align(
                    current_word,
                    observed_ipa,
                    converted_ipa,
                )

            if not alignable:
                alignable, aligned_ipa = force_align(
                    current_word,
                    observed_ipa,
                    lang=lang,
                )

        else:
            alignable = True
            aligned_ipa = observed_ipa

        # --------------------------------------------------------
        # Store
        # --------------------------------------------------------

        if len(aligned_ipa.split("_")) > 1:

            observed_ipas.append(aligned_ipa)
            converted_ipas.append(converted_ipa)
            words.append(current_word)
            alignments.append(alignable)

            assert len(aligned_ipa.split("_")) == original_len, (
                f"original: {'_'.join(context)}; "
                f"final: {aligned_ipa}"
            )

    # ------------------------------------------------------------
    # Iterate through phones
    # ------------------------------------------------------------

    for phone, word, word_id in phone_iter:

        if pd.isna(word) or pd.isna(phone):
            continue

        word = str(word).strip()
        phone = str(phone).strip()

        if not word or not phone:
            continue

        if pd.isna(word_id):
            raise ValueError(
                f"Missing word_id for word {word!r} with phone {phone!r}."
            )

        # --------------------------------------------------------
        # AE input starts as ARPAbet
        # --------------------------------------------------------

        if lang == "ae":

            if is_ae_non_speech(phone):
                continue

            if not is_ae_arpabet_phone(phone):
                continue

            phone = phonecodes.arpabet2ipa(
                phone.upper()
            )

            phone = normalize_ipa(phone, lang=lang)

        # --------------------------------------------------------
        # New word token
        # --------------------------------------------------------

        if current_word_id is None or word_id != current_word_id:

            if current_word_id is not None:
                add_data()

            current_word_id = word_id
            current_word = word
            context = [phone]

        else:

            if word != current_word:
                raise ValueError(
                    f"word_id {word_id!r} spans multiple word labels: "
                    f"{current_word!r} and {word!r}."
                )

            context.append(phone)

    # Process the final word token too.
    add_data()

    return (
        observed_ipas,
        converted_ipas,
        words,
        alignments,
    )

