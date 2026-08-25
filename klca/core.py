from __future__ import annotations
version = ".01"

from functools import lru_cache
from itertools import count
import random
import sys
import os
import pickle
import json
import sqlite3
import statistics as stat
from collections import Counter
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import argparse


statusd = {"spld": False, "models": []}

try:
	import stanza
except ModuleNotFoundError:
	stanza = None

try:
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError:
    hf_hub_download = None


DEFAULT_HF_DATASET_REPO_ID = "hksung/klca_deps"


def dep_file_path(relative: str) -> str:
    """Download a dependency file from a Hugging Face dataset repo and return its local cache path."""
    if hf_hub_download is None:
        raise ModuleNotFoundError(
            "huggingface_hub is required to download dependency files. "
            "Install it with `pip install huggingface_hub`."
        )

    repo_id = (
        os.getenv("KLCA_HF_REPO_ID", "").strip()
        or os.getenv("HF_DEPFILES_REPO_ID", "").strip()
        or DEFAULT_HF_DATASET_REPO_ID
    )
    if not repo_id:
        raise FileNotFoundError(
            f"Missing dependency configuration for `{relative}`. "
            "Set KLCA_HF_REPO_ID to a Hugging Face dataset repo containing the required files."
        )

    revision = os.getenv("KLCA_HF_REVISION", "").strip() or os.getenv("HF_DEPFILES_REVISION", "").strip() or None
    repo_subdir = os.getenv("KLCA_HF_SUBDIR", "dep_files").strip().strip("/")
    token = os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip() or None

    relative_path = Path(relative).as_posix()
    basename = Path(relative).name
    candidates = [relative_path]
    if repo_subdir and not relative_path.startswith(f"{repo_subdir}/"):
        candidates.append(f"{repo_subdir}/{basename}")
    candidates.append(basename)

    seen = set()
    last_error: Exception | None = None
    for filename in candidates:
        if filename in seen:
            continue
        seen.add(filename)
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                token=token,
            )
        except Exception as exc:
            last_error = exc

    raise FileNotFoundError(
        f"Could not download dependency file `{relative}` from dataset repo `{repo_id}`. "
        f"Tried filenames: {', '.join(seen)}. Last error: {last_error}"
    )


_SQLITE_LEVEL_TABLE_CANDIDATES = {
    "unigram": ("unigram_stats", "unigram"),
    "bigram": ("bigram_stats", "bigram"),
}

_SQLITE_KEY_COLUMN_CANDIDATES = ("token", "eojeol")

_SQLITE_COMBINED_TABLE_CANDIDATES = ("eojeol_stats", "token_stats", "stats")


class SQLiteMetricLookup:
    def __init__(self, db: "SQLiteTokenStatsDB", level: str, columns: tuple[str, ...]):
        self.db = db
        self.level = level
        self.columns = columns

    def get(self, key: str, default=None):
        row = self.db.lookup(key, level=self.level)
        if row is None:
            return default

        for column in self.columns:
            if column in row and row[column] is not None:
                return row[column]
        return default


class SQLiteTokenStatsDB:
    """SQLite-backed token stats with `unigram_stats` and `bigram_stats` tables."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.table_columns = self._load_table_columns()
        self.level_tables = self._resolve_level_tables()
        self.level_key_columns = self._resolve_level_key_columns()
        self.combined_table, self.combined_key_column = self._resolve_combined_table()

        if not any(self.level_tables.values()) and self.combined_table is None:
            raise ValueError(f"Unsupported SQLite stats schema: {db_path}")

    def _load_table_columns(self) -> Dict[str, set[str]]:
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in cur.fetchall()]

        table_columns: Dict[str, set[str]] = {}
        for table_name in table_names:
            pragma_cur = self.conn.cursor()
            pragma_cur.execute(f"PRAGMA table_info({table_name})")
            table_columns[table_name] = {row[1] for row in pragma_cur.fetchall()}
            pragma_cur.close()

        cur.close()
        return table_columns

    def _resolve_level_tables(self) -> Dict[str, Optional[str]]:
        resolved: Dict[str, Optional[str]] = {}
        for level, candidates in _SQLITE_LEVEL_TABLE_CANDIDATES.items():
            resolved[level] = next(
                (table for table in candidates if table in self.table_columns),
                None,
            )
        return resolved

    def _resolve_level_key_columns(self) -> Dict[str, Optional[str]]:
        resolved: Dict[str, Optional[str]] = {}
        for level, table_name in self.level_tables.items():
            if table_name is None:
                resolved[level] = None
                continue

            columns = self.table_columns[table_name]
            resolved[level] = next(
                (column for column in _SQLITE_KEY_COLUMN_CANDIDATES if column in columns),
                None,
            )
        return resolved

    def _resolve_combined_table(self) -> tuple[Optional[str], Optional[str]]:
        for table_name in _SQLITE_COMBINED_TABLE_CANDIDATES:
            columns = self.table_columns.get(table_name)
            if not columns:
                continue

            key_column = next(
                (column for column in _SQLITE_KEY_COLUMN_CANDIDATES if column in columns),
                None,
            )
            if key_column is not None:
                return table_name, key_column

        return None, None

    def _resolve_table(self, level: str) -> tuple[str, str]:
        table_name = self.level_tables.get(level)
        key_column = self.level_key_columns.get(level)
        if table_name is not None and key_column is not None:
            return table_name, key_column

        if self.combined_table is not None and self.combined_key_column is not None:
            return self.combined_table, self.combined_key_column

        raise KeyError(f"No SQLite table available for level `{level}` in {self.db_path}")

    def lookup(self, key: str, level: str = "unigram") -> Optional[Dict[str, Any]]:
        table_name, key_column = self._resolve_table(level)
        cur = self.conn.cursor()
        cur.execute(
            f"""
            SELECT *
            FROM {table_name}
            WHERE {key_column} = ?
            """,
            (key,),
        )
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        return dict(row)

    def metric_lookup(self, level: str, metric: str) -> SQLiteMetricLookup:
        columns = (f"{level}_{metric}", metric)
        return SQLiteMetricLookup(self, level, columns)


@lru_cache(maxsize=1)
def load_morpheme_sqlite_db() -> Optional[SQLiteTokenStatsDB]:
    try:
        return SQLiteTokenStatsDB(dep_file_path("dep_files/morpheme_db.sqlite"))
    except (FileNotFoundError, ModuleNotFoundError):
        return None


@lru_cache(maxsize=1)
def load_eojeol_sqlite_db() -> Optional[SQLiteTokenStatsDB]:
    try:
        return SQLiteTokenStatsDB(dep_file_path("dep_files/eojeol_db.sqlite"))
    except (FileNotFoundError, ModuleNotFoundError):
        return None


@lru_cache(maxsize=1)
def load_grade_level_rows() -> list[dict[str, Any]]:
    db_path = dep_file_path("dep_files/gradeLevel.sqlite")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT Form, Level, Type
        FROM grade_level
        """
    )
    rows = [dict(row) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return rows

def load_ko_stanza_default(use_gpu=False, processors="tokenize,pos,lemma"):
    """
    Load the default Stanza Korean GSD pipeline.
    """
    return load_ko_stanza_local(model_dir=None, use_gpu=use_gpu, processors=processors)


def load_ko_stanza_local(model_dir, use_gpu=False, processors="tokenize,pos,lemma"):
    if stanza is None:
        raise ModuleNotFoundError("stanza is required. Install it with `pip install stanza`.")
    pipeline_kwargs = {
        "lang": "ko",
        "processors": processors,
        "package": "gsd",
        "tokenize_no_ssplit": False,
        "use_gpu": use_gpu,
    }
    if model_dir:
        pipeline_kwargs["dir"] = model_dir

    nlp = stanza.Pipeline(**pipeline_kwargs)
    statusd["models"].append("stanza-ko-gsd")
    return nlp

nlp = None

ko_rwl_mo = pickle.load(open(dep_file_path("dep_files/ko_rwl_mo.pkl"), "rb"))  # morphemes (>=5)
ko_rwl_eo = pickle.load(open(dep_file_path("dep_files/ko_rwl_eo.pkl"), "rb"))  # eojeols (>=5)

class parameters:
    """
    Korean preprocessing defaults (Stanza).
    Unit-options:
      - unit = "morpheme": process Stanza Word (morpheme-level); POS/XPOS available; functionPOS usable
      - unit = "eojeol":   process Stanza Token (eojeol-level); 
    Real-word list:
      - rwl is selected automatically based on `unit` unless you override it manually.
    """

    # -------------------
    # Language / Model
    # -------------------
    lang = "ko"
    model = "stanza-ko-gsd"
    nlp = None
    use_gpu = False

    # -------------------
    # Granularity options
    # -------------------
    unit = "morpheme"          # {"morpheme", "eojeol"}
    use_lemma = True           # only meaningful when unit="morpheme"
    lower = True               # lowercase text/lemma

    # Choose the real-word list by unit (you can override rwl manually if needed)
    rwl_mo = ko_rwl_mo
    rwl_eo = ko_rwl_eo

    rwl_mo_key_sep = "_"   

    rwl = None            

    # -------------------
    # Filtering / ignore lists
    # -------------------
    punctuation = [
        "``", "''", "'", ".", ",", "?", "!", ")", "(", "%", "/", "-", "_",
        "-LRB-", "-RRB-", "SYM", ":", ";", '"'
    ]
    punctse = [".", "?", "!"]
    spaces = [" ", "  ", "   ", "    "]
    splitter = "\n"            # paragraph splitter

    # Stanza: Korean function morphemes (Sejong XPOS) - relevant only in morpheme mode
    functionPOS = [
        "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ",
        "JC", "JX",
        "EP", "EF", "EC", "ETN", "ETM",
        "XPN", "XSN", "XSV", "XSA",
    ]

    # Tags to ignore (mostly punctuation/symbol/foreign/number-like Sejong tags)
    posignore = ["SF", "SE", "SS", "SP", "SO", "SW", "SH", "SL", "SN", "NF", "NV", "NA"]

    # Optional lists
    abbrvs = []
    removel = []               # surface-form removal list
    override = []              # keep list even if otherwise ignored
    contentLemIgnore = []      # lemma ignore list (morpheme mode recommended)
    attested = False           # if True, pre-ignore tokens not found in the real-word list

    # Numbers
    numbers = []               # optional POS tags to mark as numbers
    nonumbers = False

    # -------------------
    # Join rules
    # -------------------
    morph_tag_sep = "+"     # morpheme 내부: lemma + "+" + XPOS  => 나+NP
    morph_ngram_sep = "_"   # morpheme ngram: mor1 + "_" + mor2  => 나+NP_는+JK
    eojeol_ngram_sep = "+"  # eojeol ngram: eoj1 + "+" + eoj2    => 나는+학교에

    # -------------------
    # Stanza processors presets
    # -------------------
    # eojeol mode: do NOT run pos/lemma/depparse
    processors_morpheme = "tokenize,pos,lemma"
    processors_eojeol = "tokenize"

    @classmethod
    def stanza_processors(cls) -> str:
        return cls.processors_morpheme if cls.unit == "morpheme" else cls.processors_eojeol

    @classmethod
    def sync(cls) -> None:
        """
        Call after changing unit/lower/etc:
          parameters.unit = "eojeol"; parameters.sync()
        """
        if cls.unit == "morpheme":
            cls.rwl = set(cls.rwl_mo) 
        else:
            cls.rwl = set(cls.rwl_eo)

    @classmethod
    def ngram_sep(cls) -> str:
        """Return ngram separator based on unit."""
        return cls.morph_ngram_sep if cls.unit == "morpheme" else cls.eojeol_ngram_sep
	
############################################
### TokObject Processing ###################
############################################

class TokObject:
    """Container for token-level annotations used by normalization and n-gram logic (Stanza only)."""

    def num_check(self, item):
        """Return True when the item contains any numeric character."""
        return any(ch.isdigit() for ch in str(item))

    def _lower_if(self, s, params):
        if s is None:
            return ""
        return s.lower() if getattr(params, "lower", False) else s

    def _xpos_parts(self, xpos):
        if xpos is None:
            return []
        # some pipelines may produce "NNG+JKS" style tags; handle both
        return [p.strip() for p in str(xpos).split("+") if p.strip()]

    def _is_xpos_ignored(self, xpos, params):
        parts = self._xpos_parts(xpos)
        return any(p in params.posignore for p in parts)

    def _is_function_morpheme(self, xpos, params):
        parts = self._xpos_parts(xpos)
        return any(p in params.functionPOS for p in parts)
    
    def _lower_lemma_only(self, lemma: str, params):
        return lemma.lower() if getattr(params, "lower", False) else lemma

    def _parse_morph_token_str(self, s: str, params):
        """
        raw morpheme token string expected: lemma+XPOS (e.g., '나+NNG')
        Return (lemma, xpos). If parse fails, (s, 'UNK').
        """
        sep = getattr(params, "morph_tag_sep", "+")
        if sep in s:
            lemma, xpos = s.rsplit(sep, 1)   
            return lemma, xpos
        return s, "UNK"
    
    def __init__(self, token=None, counter=0, params=None):
        if params is None:
            raise ValueError("TokObject requires a parameters object: params=parameters")

        self.idx = counter
        self.preIgnore = False
        self.preIgnoreReasons = []
        self.indexIgnore = False

        # -------------------------
        # Initialize defaults
        # -------------------------
        self.text = ""
        self.lemma_ = None
        self.pos_ = None
        self.tag_ = None   # XPOS in Korean (Sejong)
        self.dep_ = None
        self.head = None
        self.nchars = 0
        self.postype = None     # "cw" / "fw" (morpheme mode), None in eojeol mode by design
        self.repr = ""          # string used for ngram keys (your desired format)

        # -------------------------
        # Accept: Stanza Word (morpheme) OR Stanza Token (eojeol) OR raw string
        # -------------------------
        is_stanza_word = hasattr(token, "upos") and hasattr(token, "xpos")  # stanza.models.common.doc.Word
        is_stanza_token = hasattr(token, "words") and hasattr(token, "text") and (not is_stanza_word)  # stanza Token

        if is_stanza_word:
            # ---- Morpheme-level ----
            self.text = token.text or ""
            lemma = token.lemma or self.text
            xpos = token.xpos or "UNK"

            self.lemma_ = lemma
            self.pos_ = token.upos
            self.tag_ = xpos
            self.dep_ = getattr(token, "deprel", None)
            self.head = getattr(token, "head", None)
            self.nchars = len(self.text)

            # representation: lemma_XPOS (or text_XPOS if you disable lemma)
            base = lemma if getattr(params, "use_lemma", True) else self.text
            base = self._lower_if(base, params)
            self.repr = f"{base}+{xpos}"

            # function/content classification (ONLY meaningful in morpheme mode)
            # functionPOS => fw, otherwise cw
            self.postype = "fw" if self._is_function_morpheme(xpos, params) else "cw"

        elif is_stanza_token:
            # ---- Eojeol-level ----
            self.text = token.text or ""
            self.nchars = len(self.text)

            # eojeol mode: POS/XPOS/lemma NOT used
            self.lemma_ = None
            self.pos_ = None
            self.tag_ = None
            self.dep_ = None
            self.head = None
            self.postype = None

            self.repr = self._lower_if(self.text, params)

        elif isinstance(token, str):
            # raw string (already normalized upstream)
            self.text = token
            self.nchars = len(token)
            self.repr = token

            # Morpheme-mode strings are expected as "lemma+XPOS".
            # Recover tag/postype so downstream indices can classify cw/fw.
            if params.unit == "morpheme":
                lemma, xpos = self._parse_morph_token_str(token, params)
                self.lemma_ = lemma
                self.tag_ = xpos
                self.postype = "fw" if self._is_function_morpheme(xpos, params) else "cw"

        else:
            raise TypeError(f"Error: Expected Stanza Word/Token or string, got {type(token)}")

        self.attrs = {}

        # -------------------------
        # Real words list filtering (unit-specific rwl already set in params.sync())
        # Check based on *repr base*:
        #  - morpheme: check lemma/text only (not including _XPOS)
        #  - eojeol: check surface
        # -------------------------
        if params.unit == "morpheme":
            # morpheme key = lemma_lower + "_" + XPOS (XPOS case preserved)
            if is_stanza_word:
                lemma = (token.lemma or token.text or "")
                xpos = (token.xpos or "UNK")
            elif isinstance(token, str):
                lemma, xpos = self._parse_morph_token_str(token, params)
            else:
                lemma, xpos = (self.text or ""), "UNK"

            lemma = self._lower_lemma_only(lemma, params)
            rwl_key = f"{lemma}{params.rwl_mo_key_sep}{xpos}"

        else:
            # eojeol key = surface (optionally lower)
            rwl_key = self._lower_if(self.text, params)

        self.rwl_key = rwl_key

        if rwl_key in params.rwl:
            self.isreal = True
        else:
            self.isreal = False
            if params.attested:
                self.preIgnore = True
                self.preIgnoreReasons.append("Not in real word list")

        # -------------------------
        # Punctuation / spaces (surface-based)
        # -------------------------
        if self.text in params.punctuation:
            self.ispunct = True
            self.preIgnore = True
            self.preIgnoreReasons.append("Is punctuation")
        else:
            self.ispunct = False

        if self.text in params.spaces:
            self.isspace = True
            self.preIgnore = True
            self.preIgnoreReasons.append("Is a space")
        else:
            self.isspace = False

        # -------------------------
        # Numbers
        # -------------------------
        if self.num_check(self.text) or (self.pos_ in params.numbers if self.pos_ is not None else False):
            self.isnumber = True
        else:
            self.isnumber = False

        if params.nonumbers is True and self.isnumber is True:
            self.preIgnore = True
            self.preIgnoreReasons.append("Numbers Ignored")

        # -------------------------
        # Override / removel (surface-based)
        # -------------------------
        if self._lower_if(self.text, params) in params.override:
            self.override = True
        else:
            self.override = False

        if self.text in params.removel:
            self.inremove = True
            self.preIgnore = True
            self.preIgnoreReasons.append("In remove list")
        else:
            self.inremove = False

        # -------------------------
        # POS ignore (morpheme mode ONLY; eojeol has no POS)
        # -------------------------
        if params.unit == "morpheme" and self.tag_ is not None and self._is_xpos_ignored(self.tag_, params):
            self.inposignore = True
            self.preIgnore = True
            self.preIgnoreReasons.append("Ignored POS")
        else:
            self.inposignore = False

############################################
### Normalize ##############################
############################################

class Normalize:
    """
    Build token/sentence/paragraph views and normalized outputs for a text (Korean processing, unit-aware, Stanza-only).

    Assumptions:
      - params.unit in {"morpheme", "eojeol"}
      - params.nlp is a Stanza pipeline
      - TokObject(unit, counter, params) works for:
          * morpheme mode: unit is stanza Word (sent.words)
          * eojeol mode:   unit is stanza Token (sent.tokens)
        and exposes:
          tok.text, tok.repr, tok.preIgnore, tok.override,
          tok.ispunct, tok.isspace, tok.isnumber, tok.isreal,
          tok.inremove, tok.inposignore
      - Join rules:
          * morpheme unigram: 나+NP
          * morpheme bigram: 나+NP_는+JK
          * eojeol bigram:    나는+학교에
    """

    def _ngram_sep(self, params):
        return "_" if params.unit == "morpheme" else "+"

    def _iter_units_in_sent(self, sent, params):
        return sent.words if params.unit == "morpheme" else sent.tokens

    def text2para(self, text, params):
        paras = []
        for x in text.split(params.splitter):
            if len(x) == 0:
                continue
            paras.append(x)
        return paras

    def _expand_morpheme_units(self, stanza_token, params):
        units = []
        for w in stanza_token.words:
            lemma = (w.lemma or w.text or "")
            xpos = (w.xpos or "UNK")

            lemma_parts = [p.strip() for p in str(lemma).split("+") if p.strip()]
            xpos_parts  = [p.strip() for p in str(xpos).split("+") if p.strip()]

            if len(lemma_parts) == len(xpos_parts) and len(lemma_parts) > 0:
                for l, x in zip(lemma_parts, xpos_parts):
                    l = l.lower() if getattr(params, "lower", False) else l
                    units.append(f"{l}+{x}")
            else:
                m = min(len(lemma_parts), len(xpos_parts))
                if m > 0:
                    for i in range(m):
                        l = lemma_parts[i]
                        x = xpos_parts[i]
                        l = l.lower() if getattr(params, "lower", False) else l
                        units.append(f"{l}+{x}")
                else:
                    base = lemma.lower() if getattr(params, "lower", False) else lemma
                    units.append(f"{base}+{xpos}")
        return units
    # ---- tokenization pipelines ----

    def text2tok(self, text, params):
        """
        Flat TokObject list for the whole text (sentence boundaries ignored).
        Keeps a global counter across the full text.
        """
        counter = 0
        tok_text = []

        if params.model is None or params.nlp is None:
            # fallback whitespace mode
            text = text.replace("\n", " ")
            for token in text.split(" "):
                if not token:
                    continue
                tok_text.append(TokObject(token, counter, params))
                counter += 1
            return tok_text

        # Stanza
        doc = params.nlp(text.replace("\n", " "))
        for sent in doc.sentences:
            for unit in self._iter_units_in_sent(sent, params):
                tok_text.append(TokObject(unit, counter, params))
                counter += 1
        return tok_text

    def text2toks(self, text, params):
        """
        Sentence-tokenized TokObject list: [sent[tok]].
        Morpheme mode: expand eojeol into lemma+XPOS units.
        """
        tok_texts = []

        if params.model is None or params.nlp is None:
            # fallback whitespace mode
            text = text.replace("\n", " ")
            toks = []
            counter = 0
            for token in text.split(" "):
                if not token:
                    continue
                toks.append(TokObject(token, counter, params))
                counter += 1
            tok_texts.append(toks)
            return tok_texts

        doc = params.nlp(text)

        for sent in doc.sentences:
            toks = []
            counter = 0

            if params.unit == "morpheme":
                for token in sent.tokens:  # eojeol
                    for word in token.words:
                        lemma = word.lemma or word.text or ""
                        xpos = word.xpos or "UNK"

                        lemma_parts = [p.strip() for p in str(lemma).split("+") if p.strip()]
                        xpos_parts  = [p.strip() for p in str(xpos).split("+") if p.strip()]

                        if len(lemma_parts) == len(xpos_parts) and len(lemma_parts) > 0:
                            for l, x in zip(lemma_parts, xpos_parts):
                                if getattr(params, "lower", False):
                                    l = l.lower()
                                toks.append(TokObject(f"{l}+{x}", counter, params))
                                counter += 1
                        else:
                            m = min(len(lemma_parts), len(xpos_parts))
                            for i in range(m):
                                l = lemma_parts[i]
                                x = xpos_parts[i]
                                if getattr(params, "lower", False):
                                    l = l.lower()
                                toks.append(TokObject(f"{l}+{x}", counter, params))
                                counter += 1

            else:
                for unit in sent.tokens:
                    toks.append(TokObject(unit, counter, params))
                    counter += 1

            tok_texts.append(toks)

        return tok_texts

    def text2tokp(self, text, params):
        """Convert text into [para[sent[tok]]] structure."""
        tok_texts = []
        for para in self.text2para(text, params):
            tok_texts.append(self.text2toks(para, params))
        return tok_texts

    # ---- normalization / ngrams ----

    def normalize(self, fl_paras, params):
        """
        Apply ignore and override rules to produce normalized token strings.
        Uses TokObject.repr as the single source of truth.
        """
        normalized = []
        ignored = []

        for paras in fl_paras:
            sents = []
            for sent in paras:
                toks = []
                for tok in sent:
                    if tok.override:
                        toks.append(tok.repr)
                        continue
                    if tok.preIgnore:
                        ignored.append(tok.repr)
                        continue
                    toks.append(tok.repr)
                sents.append(toks)
            normalized.append(sents)

        return normalized, ignored

    def ngramize(self, fl_paras, params, n=2):
        """Create cleaned n-grams from TokObjects and track ignored outputs."""
        sep = self._ngram_sep(params)

        def ngrammer(tokenized, number):
            cleaned = []
            for tok in tokenized:
                if tok.override:
                    cleaned.append(tok)
                    continue
                if tok.ispunct or tok.isspace:
                    continue
                cleaned.append(tok)

            ngram_list = []
            last_index = len(cleaned)
            for i in range(last_index - number + 1):
                ngram_list.append(cleaned[i:i + number])
            return ngram_list

        normalized = []
        ignored = []

        for paras in fl_paras:
            sents = []
            for sent in paras:
                ngramtoks = ngrammer(sent, n)
                ngrams = []

                for ngram in ngramtoks:
                    problem = False

                    for tok in ngram:
                        if tok.override:
                            continue
                        if tok.inremove:
                            problem = True
                        if params.nonumbers and tok.isnumber:
                            problem = True
                        if params.attested and (not tok.isreal):
                            problem = True
                        # POS ignore only applies in morpheme mode
                        if params.unit == "morpheme" and tok.inposignore:
                            problem = True

                    ngram_tok = sep.join([x.repr for x in ngram])

                    if problem:
                        ignored.append(ngram_tok)
                    else:
                        ngrams.append(ngram_tok)

                sents.append(ngrams)
            normalized.append(sents)

        return normalized, ignored

    # ---- view helpers ----

    def paratok2text(self, paratok):
        texttoks = []
        for paras in paratok:
            para = []
            for sent in paras:
                para.append([tok.text for tok in sent])
            texttoks.append(para)
        return texttoks

    def para2sent(self, paratok):
        senttoks = []
        for paras in paratok:
            for sent in paras:
                senttoks.append(sent)
        return senttoks

    def senttok2text(self, senttok):
        senttext = []
        for sent in senttok:
            senttext.append([tok.text for tok in sent])
        return senttext

    def sent2tok(self, senttok):
        return [y for x in senttok for y in x]

    def tok2text(self, toks):
        return [x.text for x in toks]

    # ---- init  ----

    def __init__(self, text=None, params=None):
        if params is None:
            raise ValueError("no parameters file loaded")

        if text is None:
            self.paras = None
            self.sents = None
            self.toks = None
            self.paratxt = None
            self.senttxt = None
            self.toktxt = None
            self.ignored = None
            self.paras_bg = None
            self.sents_bg = None
            self.toks_bg = None
            self.ignored_bg = None
            return

        # TokObject token views
        self.tokens = self.text2tok(text, params)        # flat TokObjects (global counter)
        self.parasto = self.text2tokp(text, params)      # [para[sent[tok]]]
        self.sentsto = self.para2sent(self.parasto)      # [sent[tok]]
        self.toksto = self.sent2tok(self.sentsto)        # [tok]

        # Raw text views (surface forms)
        self.paratxt = self.paratok2text(self.parasto)
        self.senttxt = self.senttok2text(self.sentsto)
        self.toktxt = self.tok2text(self.toksto)

        # Normalized tokens (repr)
        self.normout = self.normalize(self.parasto, params)
        self.paras = self.normout[0]
        self.sents = self.para2sent(self.paras)
        self.toks = self.sent2tok(self.sents)
        self.ignored = self.normout[1]

        # Normalized bigrams by default
        self.bgout = self.ngramize(self.parasto, params, 2)
        self.paras_bg = self.bgout[0]
        self.sents_bg = self.para2sent(self.paras_bg)
        self.toks_bg = self.sent2tok(self.sents_bg)
        self.ignored_bg = self.bgout[1]


#########################
### Raw frequency #######
#########################

class FrequencyIndices:
    """
    Raw frequency indices requested in the analysis table.

    Required inputs:
      - morpheme_norm: Normalize(text, params) built with params.unit == "morpheme"
      - eojeol_norm:   Normalize(text, params) built with params.unit == "eojeol"
    """

    def _all_morpheme_frequency(self, morpheme_norm):
        # Count lexical morpheme tokens only (exclude POS-ignored items like punctuation tags).
        return sum(
            1
            for tok in morpheme_norm.toksto
            if getattr(tok, "postype", None) in {"cw", "fw"} and not getattr(tok, "inposignore", False)
        )

    def _content_morpheme_frequency(self, morpheme_norm):
        return sum(
            1
            for tok in morpheme_norm.toksto
            if getattr(tok, "postype", None) == "cw" and not getattr(tok, "inposignore", False)
        )

    def _functional_morpheme_frequency(self, morpheme_norm):
        return sum(
            1
            for tok in morpheme_norm.toksto
            if getattr(tok, "postype", None) == "fw" and not getattr(tok, "inposignore", False)
        )

    def _all_eojeol_frequency(self, eojeol_norm):
        return sum(
            1
            for tok in eojeol_norm.toksto
            if not getattr(tok, "ispunct", False) and not getattr(tok, "isspace", False)
        )

    def __init__(self, morpheme_norm=None, eojeol_norm=None):
        self.AM_freq = self._all_morpheme_frequency(morpheme_norm) if morpheme_norm is not None else None
        self.CM_freq = self._content_morpheme_frequency(morpheme_norm) if morpheme_norm is not None else None
        self.FM_freq = self._functional_morpheme_frequency(morpheme_norm) if morpheme_norm is not None else None
        self.AE_freq = self._all_eojeol_frequency(eojeol_norm) if eojeol_norm is not None else None
        self.AM_type = len(set(tok.repr for tok in morpheme_norm.toksto if getattr(tok, "postype", None) in {"cw", "fw"} and not getattr(tok, "inposignore", False))) if morpheme_norm is not None else None
        self.CM_type = len(set(tok.repr for tok in morpheme_norm.toksto if getattr(tok, "postype", None) == "cw" and not getattr(tok, "inposignore", False))) if morpheme_norm is not None else None
        self.FM_type = len(set(tok.repr for tok in morpheme_norm.toksto if getattr(tok, "postype", None) == "fw" and not getattr(tok, "inposignore", False))) if morpheme_norm is not None else None
        self.AE_type = len(set(tok.repr for tok in eojeol_norm.toksto if not getattr(tok, "ispunct", False) and not getattr(tok, "isspace", False))) if eojeol_norm is not None else None

        self.vald = {
            "AM_freq": self.AM_freq,
            "CM_freq": self.CM_freq,
            "FM_freq": self.FM_freq,
            "AE_freq": self.AE_freq,
            "AM_type": self.AM_type,
            "CM_type": self.CM_type,
            "FM_type": self.FM_type,
            "AE_type": self.AE_type,
        }

#########################
### Diversity ###########
#########################

class DiversityIndices(FrequencyIndices):

    def safe_divide(self, numerator, denominator):
        if denominator == 0 or denominator == 0.0:
            return 0.0
        return numerator / denominator

    def TTR(self, text):
        return self.safe_divide(len(set(text)), len(text))

    def MATTR(self, text, window_length=50):
        if len(text) == 0:
            return 0.0
        if len(text) < (window_length + 1):
            return self.TTR(text)
        vals = []
        for x in range(len(text)):
            small_text = text[x:(x + window_length)]
            if len(small_text) < window_length:
                break
            vals.append(self.safe_divide(len(set(small_text)), float(window_length)))
        return stat.mean(vals) if vals else 0.0

    def HDD(self, text, samples=42):
        def choose(n, k):
            if 0 <= k <= n:
                ntok = 1
                ktok = 1
                for t in range(1, min(k, n - k) + 1):
                    ntok *= n
                    ktok *= t
                    n -= 1
                return ntok // ktok
            return 0

        def hyper(successes, sample_size, population_size, freq):
            try:
                prob_1 = 1.0 - (
                    float(choose(freq, successes) * choose((population_size - freq), (sample_size - successes)))
                    / float(choose(population_size, sample_size))
                )
                prob_1 = prob_1 * (1 / sample_size)
            except ZeroDivisionError:
                prob_1 = 0.0
            return prob_1

        ntokens = len(text)
        if ntokens == 0:
            return 0.0
        if ntokens < samples:
            samples = ntokens
        prob_sum = 0.0
        frequency_dict = Counter(text)
        for item in set(text):
            prob_sum += hyper(0, samples, ntokens, frequency_dict[item])
        return prob_sum

    def MTLDER(self, text, mn=10, ttrval=.720):
        ft = []
        fl = []
        fp = []
        start = 0
        for x in range(len(text)):
            factor_text = text[start:x + 1]
            if x + 1 == len(text):
                ft.append(factor_text)
                fact_prop = self.safe_divide((1 - self.TTR(factor_text)), (1 - ttrval))
                fp.append(fact_prop)
                fl.append(len(factor_text))
            else:
                if self.TTR(factor_text) < ttrval and len(factor_text) >= mn:
                    ft.append(factor_text)
                    fl.append(len(factor_text))
                    fp.append(1)
                    start = x + 1
        return ft, fl, fp

    def MTLD_MFL(self, windowl, factorprop):
        factorls = []
        for wl, fp in zip(windowl, factorprop):
            if fp == 0:
                continue
            factorls.append(self.safe_divide(wl, fp))
        return stat.mean(factorls) if factorls else 0.0

    def MTLD(self, text, mn=10, ttrval=.720):
        if len(text) == 0:
            return 0.0
        _, fwfl, fwfp = self.MTLDER(text, mn, ttrval)
        _, bwfl, bwfp = self.MTLDER(list(reversed(text)), mn, ttrval)
        return self.MTLD_MFL(fwfl + bwfl, fwfp + bwfp)

    def _am_tokens(self, morpheme_norm):
        return [
            tok.repr for tok in morpheme_norm.toksto
            if getattr(tok, "postype", None) in {"cw", "fw"} and not getattr(tok, "inposignore", False)
        ]

    def _cm_tokens(self, morpheme_norm):
        return [
            tok.repr for tok in morpheme_norm.toksto
            if getattr(tok, "postype", None) == "cw" and not getattr(tok, "inposignore", False)
        ]

    def _fm_tokens(self, morpheme_norm):
        return [
            tok.repr for tok in morpheme_norm.toksto
            if getattr(tok, "postype", None) == "fw" and not getattr(tok, "inposignore", False)
        ]

    def _ae_tokens(self, eojeol_norm):
        return [
            tok.repr for tok in eojeol_norm.toksto
            if not getattr(tok, "ispunct", False) and not getattr(tok, "isspace", False)
        ]

    def _bundle(self, tokens, mattr_window=50, hdd_samples=42, mtld_mn=10, mtld_ttr=.720):
        return {
            "MATTR": self.MATTR(tokens, window_length=mattr_window),
            "HDD": self.HDD(tokens, samples=hdd_samples),
            "MTLD": self.MTLD(tokens, mn=mtld_mn, ttrval=mtld_ttr),
        }

    def __init__(
        self,
        morpheme_norm=None,
        eojeol_norm=None,
        mattr_window=50,
        hdd_samples=42,
        mtld_mn=10,
        mtld_ttr=.720,
    ):
        am_tokens = self._am_tokens(morpheme_norm) if morpheme_norm is not None else []
        cm_tokens = self._cm_tokens(morpheme_norm) if morpheme_norm is not None else []
        fm_tokens = self._fm_tokens(morpheme_norm) if morpheme_norm is not None else []
        ae_tokens = self._ae_tokens(eojeol_norm) if eojeol_norm is not None else []

        am = self._bundle(am_tokens, mattr_window, hdd_samples, mtld_mn, mtld_ttr)
        cm = self._bundle(cm_tokens, mattr_window, hdd_samples, mtld_mn, mtld_ttr)
        fm = self._bundle(fm_tokens, mattr_window, hdd_samples, mtld_mn, mtld_ttr)
        ae = self._bundle(ae_tokens, mattr_window, hdd_samples, mtld_mn, mtld_ttr)

        self.AM_MATTR = am["MATTR"] if morpheme_norm is not None else None
        self.AM_HDD = am["HDD"] if morpheme_norm is not None else None
        self.AM_MTLD = am["MTLD"] if morpheme_norm is not None else None
        self.CM_MATTR = cm["MATTR"] if morpheme_norm is not None else None
        self.CM_HDD = cm["HDD"] if morpheme_norm is not None else None
        self.CM_MTLD = cm["MTLD"] if morpheme_norm is not None else None
        self.FM_MATTR = fm["MATTR"] if morpheme_norm is not None else None
        self.FM_HDD = fm["HDD"] if morpheme_norm is not None else None
        self.FM_MTLD = fm["MTLD"] if morpheme_norm is not None else None
        self.AE_MATTR = ae["MATTR"] if eojeol_norm is not None else None
        self.AE_HDD = ae["HDD"] if eojeol_norm is not None else None
        self.AE_MTLD = ae["MTLD"] if eojeol_norm is not None else None

        self.vald = {
            "AM_MATTR": self.AM_MATTR,
            "AM_HDD": self.AM_HDD,
            "AM_MTLD": self.AM_MTLD,
            "CM_MATTR": self.CM_MATTR,
            "CM_HDD": self.CM_HDD,
            "CM_MTLD": self.CM_MTLD,
            "FM_MATTR": self.FM_MATTR,
            "FM_HDD": self.FM_HDD,
            "FM_MTLD": self.FM_MTLD,
            "AE_MATTR": self.AE_MATTR,
            "AE_HDD": self.AE_HDD,
            "AE_MTLD": self.AE_MTLD,
        }

###############################
### Sophistication ############
###############################

class SophisticationIndices:
    """
    Rarity (log-frequency) indices using corpus counts from pickles.

    Indices:
      - AM_logFreq:   mean log relative frequency of all morphemes
      - CM_logFreq:   mean log relative frequency of content morphemes
      - FM_logFreq:   mean log relative frequency of functional morphemes
      - BiAM_logFreq: mean bigram log frequency of continuous morphemes
      - AE_logFreq:   mean log relative frequency of eojeols
      - BiAE_logFreq: mean bigram log frequency of continuous eojeols
    """
    def _mean(self, values):
        return stat.mean(values) if values else 0.0

    # -------------------------
    # Token selection
    # -------------------------
    def _all_morpheme_tokobjs(self, morpheme_norm):
        return [
            tok for tok in getattr(morpheme_norm, "toksto", [])
            if getattr(tok, "postype", None) in {"cw", "fw"}
            and not getattr(tok, "inposignore", False)
        ]

    def _content_morpheme_tokobjs(self, morpheme_norm):
        return [
            tok for tok in getattr(morpheme_norm, "toksto", [])
            if getattr(tok, "postype", None) == "cw"
            and not getattr(tok, "inposignore", False)
        ]

    def _functional_morpheme_tokobjs(self, morpheme_norm):
        return [
            tok for tok in getattr(morpheme_norm, "toksto", [])
            if getattr(tok, "postype", None) == "fw"
            and not getattr(tok, "inposignore", False)
        ]

    def _all_eojeol_tokobjs(self, eojeol_norm):
        return [
            tok for tok in getattr(eojeol_norm, "toksto", [])
            if not getattr(tok, "ispunct", False)
            and not getattr(tok, "isspace", False)
        ]

    # -------------------------
    # Generic DB lookup means (skip missing)
    # -------------------------
    def _mean_unigram_metric(self, tokobjs, unigram_dict, debug=False, label=""):
        vals = []
        skipped = 0
        for tok in tokobjs:
            key = getattr(tok, "rwl_key", "") or ""
            v = unigram_dict.get(key)
            if v is None:
                skipped += 1
                continue
            vals.append(v)

        if debug:
            total = len(tokobjs)
            print(f"[{label}] unigram total={total} used={len(vals)} skipped={skipped}")

        return self._mean(vals)

    def _mean_bigram_metric(self, tokobjs, bigram_dict, debug=False, label=""):
        if len(tokobjs) < 2:
            if debug:
                print(f"[{label}] bigram total=0 used=0 skipped=0")
            return 0.0

        vals = []
        skipped = 0
        for left, right in zip(tokobjs, tokobjs[1:]):
            lk = getattr(left, "rwl_key", "") or ""
            rk = getattr(right, "rwl_key", "") or ""
            if not lk or not rk:
                skipped += 1
                continue

            bigram_key = f"{lk}+{rk}"
            v = bigram_dict.get(bigram_key)
            if v is None:
                skipped += 1
                continue
            vals.append(v)

        if debug:
            total_pairs = len(tokobjs) - 1
            print(f"[{label}] bigram total={total_pairs} used={len(vals)} skipped={skipped}")

        return self._mean(vals)

    def _resolve_metric_sources(self, db, label):
        if isinstance(db, dict):
            return (
                db["unigram"]["log_perN"],
                db["bigram"]["log_perN"],
                db["unigram"]["range_prop"],
                db["bigram"]["range_prop"],
            )

        if hasattr(db, "metric_lookup"):
            return (
                db.metric_lookup("unigram", "log_perN"),
                db.metric_lookup("bigram", "log_perN"),
                db.metric_lookup("unigram", "range_prop"),
                db.metric_lookup("bigram", "range_prop"),
            )

        raise TypeError(f"Unsupported {label} DB source. Expected SQLiteTokenStatsDB.")

    # -------------------------
    # Constructor
    # -------------------------
    def __init__(
        self,
        morpheme_norm=None,
        eojeol_norm=None,
        mo_db=None,
        eo_db=None,
        debug=False,
    ):
        # --- Load DBs (precomputed) if not provided ---
        if mo_db is None:
            mo_db = load_morpheme_sqlite_db()
            if mo_db is None:
                raise FileNotFoundError("Missing SQLite dependency file: dep_files/morpheme_db.sqlite")
        if eo_db is None:
            eo_db = load_eojeol_sqlite_db()
            if eo_db is None:
                raise FileNotFoundError("Missing SQLite dependency file: dep_files/eojeol_db.sqlite")

        self.mo_db = mo_db
        self.eo_db = eo_db

        # Pull the actual lookup dicts we need (log frequency)
        (
            self.mo_uni_log,
            self.mo_bi_log,
            self.mo_uni_range,
            self.mo_bi_range,
        ) = self._resolve_metric_sources(self.mo_db, "morpheme")
        (
            self.eo_uni_log,
            self.eo_bi_log,
            self.eo_uni_range,
            self.eo_bi_range,
        ) = self._resolve_metric_sources(self.eo_db, "eojeol")

        # Default outputs
        self.AM_logFreq = None
        self.CM_logFreq = None
        self.FM_logFreq = None
        self.BiAM_logFreq = None
        self.AE_logFreq = None
        self.BiAE_logFreq = None

        # New: range indices
        self.AM_range = None
        self.CM_range = None
        self.FM_range = None
        self.BiAM_range = None
        self.AE_range = None
        self.BiAE_range = None

        if morpheme_norm is None and eojeol_norm is None:
            self.vald = {
                "AM_logFreq": None,
                "CM_logFreq": None,
                "FM_logFreq": None,
                "BiAM_logFreq": None,
                "AE_logFreq": None,
                "BiAE_logFreq": None,
                "AM_range": None,
                "CM_range": None,
                "FM_range": None,
                "BiAM_range": None,
                "AE_range": None,
                "BiAE_range": None,
            }
            return

        # --- Token selectors ---
        am_toks = self._all_morpheme_tokobjs(morpheme_norm) if morpheme_norm else []
        cm_toks = self._content_morpheme_tokobjs(morpheme_norm) if morpheme_norm else []
        fm_toks = self._functional_morpheme_tokobjs(morpheme_norm) if morpheme_norm else []
        ae_toks = self._all_eojeol_tokobjs(eojeol_norm) if eojeol_norm else []

        # --- Compute indices using DB lookups only (skip missing) ---
        if morpheme_norm is not None:
            # log frequency
            self.AM_logFreq   = self._mean_unigram_metric(am_toks, self.mo_uni_log, debug=debug, label="AM_logFreq")
            self.CM_logFreq   = self._mean_unigram_metric(cm_toks, self.mo_uni_log, debug=debug, label="CM_logFreq")
            self.FM_logFreq   = self._mean_unigram_metric(fm_toks, self.mo_uni_log, debug=debug, label="FM_logFreq")
            self.BiAM_logFreq = self._mean_bigram_metric(am_toks, self.mo_bi_log,  debug=debug, label="BiAM_logFreq")

            # range proportion
            self.AM_range   = self._mean_unigram_metric(am_toks, self.mo_uni_range, debug=debug, label="AM_range")
            self.CM_range   = self._mean_unigram_metric(cm_toks, self.mo_uni_range, debug=debug, label="CM_range")
            self.FM_range   = self._mean_unigram_metric(fm_toks, self.mo_uni_range, debug=debug, label="FM_range")
            self.BiAM_range = self._mean_bigram_metric(am_toks, self.mo_bi_range,  debug=debug, label="BiAM_range")

        if eojeol_norm is not None:
            # log frequency
            self.AE_logFreq   = self._mean_unigram_metric(ae_toks, self.eo_uni_log, debug=debug, label="AE_logFreq")
            self.BiAE_logFreq = self._mean_bigram_metric(ae_toks, self.eo_bi_log,  debug=debug, label="BiAE_logFreq")

            # range proportion
            self.AE_range   = self._mean_unigram_metric(ae_toks, self.eo_uni_range, debug=debug, label="AE_range")
            self.BiAE_range = self._mean_bigram_metric(ae_toks, self.eo_bi_range,  debug=debug, label="BiAE_range")

        self.vald = {
            "AM_logFreq": self.AM_logFreq,
            "CM_logFreq": self.CM_logFreq,
            "FM_logFreq": self.FM_logFreq,
            "BiAM_logFreq": self.BiAM_logFreq,
            "AE_logFreq": self.AE_logFreq,
            "BiAE_logFreq": self.BiAE_logFreq,
            "AM_range": self.AM_range,
            "CM_range": self.CM_range,
            "FM_range": self.FM_range,
            "BiAM_range": self.BiAM_range,
            "AE_range": self.AE_range,
            "BiAE_range": self.BiAE_range,
        }

###############################
### SOA #######################
###############################

class SOAIndices:
    """
    SOA indices using precomputed association measures from JSON DBs.

    Indices:
      - BiAM_MI:           Mean MI of continuous morpheme bigrams
      - BiAM_deltaPleft:   Mean deltaPleft of continuous morpheme bigrams
      - BiAM_deltaPright:  Mean deltaPright of continuous morpheme bigrams
      - BiAE_MI:           Mean MI of continuous eojeol bigrams
      - BiAE_deltaPleft:   Mean deltaPleft of continuous eojeol bigrams
      - BiAE_deltaPright:  Mean deltaPright of continuous eojeol bigrams
    """

    def _mean(self, values):
        return stat.mean(values) if values else 0.0

    # -------------------------
    # Token selection
    # -------------------------
    def _all_morpheme_tokobjs(self, morpheme_norm):
        return [
            tok for tok in getattr(morpheme_norm, "toksto", [])
            if getattr(tok, "postype", None) in {"cw", "fw"}
            and not getattr(tok, "inposignore", False)
        ]

    def _all_eojeol_tokobjs(self, eojeol_norm):
        return [
            tok for tok in getattr(eojeol_norm, "toksto", [])
            if not getattr(tok, "ispunct", False)
            and not getattr(tok, "isspace", False)
        ]

    # -------------------------
    # Generic DB lookup means (skip missing)
    # -------------------------
    def _mean_bigram_metric(self, tokobjs, bigram_metric_dict, debug=False, label=""):
        """
        Build bigram key as f"{left}+{right}" using tok.rwl_key.
        If key is missing from the DB, skip it (do NOT approximate).
        """
        if len(tokobjs) < 2:
            if debug:
                print(f"[{label}] bigram total=0 used=0 skipped=0")
            return 0.0

        vals = []
        skipped = 0

        for left, right in zip(tokobjs, tokobjs[1:]):
            lk = getattr(left, "rwl_key", "") or ""
            rk = getattr(right, "rwl_key", "") or ""
            if not lk or not rk:
                skipped += 1
                continue

            k = f"{lk}+{rk}"
            v = bigram_metric_dict.get(k)
            if v is None:
                skipped += 1
                continue

            vals.append(v)

        if debug:
            total_pairs = len(tokobjs) - 1
            print(f"[{label}] bigram total={total_pairs} used={len(vals)} skipped={skipped}")

        return self._mean(vals)

    # -------------------------
    # Constructor
    # -------------------------
    def __init__(
        self,
        morpheme_norm=None,
        eojeol_norm=None,
        mo_db=None,
        eo_db=None,
        debug=False,
    ):
        # --- Load DBs if not provided ---
        # NOTE: these are JSON files, so use json.load (NOT pickle.load)
        if mo_db is None:
            with open(dep_file_path("dep_files/soa_morph.json"), "r", encoding="utf-8") as f:
                mo_db = json.load(f)
        if eo_db is None:
            with open(dep_file_path("dep_files/soa_eojeol.json"), "r", encoding="utf-8") as f:
                eo_db = json.load(f)

        self.soa_mo_db = mo_db
        self.soa_eo_db = eo_db

        # Pull the actual lookup dicts we need
        self.mo_MI = self.soa_mo_db.get("mi", {})
        self.mo_deltaPleft = self.soa_mo_db.get("deltap_left", {})
        self.mo_deltaPright = self.soa_mo_db.get("deltap_right", {})

        self.eo_MI = self.soa_eo_db.get("mi", {})
        self.eo_deltaPleft = self.soa_eo_db.get("deltap_left", {})
        self.eo_deltaPright = self.soa_eo_db.get("deltap_right", {})

        # Default outputs
        self.BiAM_MI = None
        self.BiAM_deltaPleft = None
        self.BiAM_deltaPright = None
        self.BiAE_MI = None
        self.BiAE_deltaPleft = None
        self.BiAE_deltaPright = None

        if morpheme_norm is None and eojeol_norm is None:
            self.vald = {
                "BiAM_MI": None,
                "BiAM_deltaPleft": None,
                "BiAM_deltaPright": None,
                "BiAE_MI": None,
                "BiAE_deltaPleft": None,
                "BiAE_deltaPright": None,
            }
            return

        # --- Token selectors ---
        am_toks = self._all_morpheme_tokobjs(morpheme_norm) if morpheme_norm else []
        ae_toks = self._all_eojeol_tokobjs(eojeol_norm) if eojeol_norm else []

        # --- Compute indices (true bigram lookup only; skip missing) ---
        if morpheme_norm is not None:
            self.BiAM_MI = self._mean_bigram_metric(am_toks, self.mo_MI, debug=debug, label="BiAM_MI")
            self.BiAM_deltaPleft = self._mean_bigram_metric(am_toks, self.mo_deltaPleft, debug=debug, label="BiAM_deltaPleft")
            self.BiAM_deltaPright = self._mean_bigram_metric(am_toks, self.mo_deltaPright, debug=debug, label="BiAM_deltaPright")

        if eojeol_norm is not None:
            self.BiAE_MI = self._mean_bigram_metric(ae_toks, self.eo_MI, debug=debug, label="BiAE_MI")
            self.BiAE_deltaPleft = self._mean_bigram_metric(ae_toks, self.eo_deltaPleft, debug=debug, label="BiAE_deltaPleft")
            self.BiAE_deltaPright = self._mean_bigram_metric(ae_toks, self.eo_deltaPright, debug=debug, label="BiAE_deltaPright")

        self.vald = {
            "BiAM_MI": self.BiAM_MI,
            "BiAM_deltaPleft": self.BiAM_deltaPleft,
            "BiAM_deltaPright": self.BiAM_deltaPright,
            "BiAE_MI": self.BiAE_MI,
            "BiAE_deltaPleft": self.BiAE_deltaPleft,
            "BiAE_deltaPright": self.BiAE_deltaPright,
        }


#####################################
### Compositionality Indices ########
#####################################

class CompositionalityIndices:
    """
    This part mirrors InflectionalDiversityIndices:
      - Iterate eojeols from doc.sentences -> token
      - Convert each eojeol token into morpheme TokObjects via Normalize._expand_morpheme_units
      - Use TokObject.tag_ (XPOS) to:
          * detect head type (noun vs verb) by tags starting with N or V
          * count functional morphemes (J*, E*)
          * count derivational suffixes (XSA, XSN, XSV)

    Indices (Eojeol-level means):
      Segmental compositionality
        - AE_MC : total morphemes / total eojeols
        - NE_MC   : total morphemes in noun-headed eojeols / # noun-headed eojeols
        - VE_MC   : total morphemes in verb-headed eojeols / # verb-headed eojeols

      Functional morpheme compositionality
        - AE_FMC : total (J* + E*) / total eojeols
        - NE_FMC   : total J* in noun-headed eojeols / # noun-headed eojeols
        - VE_FMC   : total E* in verb-headed eojeols / # verb-headed eojeols

    """

    def safe_divide(self, numerator, denominator):
        if denominator == 0 or denominator == 0.0:
            return 0.0
        return numerator / denominator

    def _mean(self, values):
        return stat.mean(values) if values else 0.0

    def _eojeol_tokobjects(self, eojeol_token, params):
        """
        Reuse Normalize/TokObject token treatment so compositionality analysis
        follows the same lemma+XPOS handling as other indices.
        """
        norm_helper = Normalize(text=None, params=params)
        morph_units = norm_helper._expand_morpheme_units(eojeol_token, params)  # ["lemma+XPOS", ...]
        return [TokObject(unit, counter=0, params=params) for unit in morph_units]

    def _head_type(self, xpos_seq):
        """
        Head type for the eojeol:
          - 'N' if there is any tag that starts with 'N'
          - 'V' if there is any tag that starts with 'V'
        If both exist, choose the earliest occurring one (in morpheme order).
        If neither, return None.
        """
        first_n = None
        first_v = None
        for i, x in enumerate(xpos_seq):
            if first_n is None and x.startswith("N"):
                first_n = i
            if first_v is None and x.startswith("V"):
                first_v = i
            if first_n is not None and first_v is not None:
                break

        if first_n is None and first_v is None:
            return None
        if first_n is None:
            return "V"
        if first_v is None:
            return "N"
        return "N" if first_n < first_v else "V"

    def _collect_counts(self, text, params, debug=False, debug_n=20):
        if params is None or getattr(params, "nlp", None) is None:
            raise ValueError("CompositionalityIndices requires params.nlp with POS/XPOS enabled.")
        if getattr(params, "unit", None) != "morpheme":
            raise ValueError("CompositionalityIndices requires params.unit == 'morpheme'.")

        doc = params.nlp(text)

        # Denominators
        all_eojeol_n = 0
        n_eojeol_n = 0
        v_eojeol_n = 0

        # Segmental totals (morpheme counts)
        all_morph_total = 0
        n_morph_total = 0
        v_morph_total = 0

        # Inflection totals
        all_inf_total = 0          # J* + E*
        n_inf_total = 0            # J* in N-headed
        v_inf_total = 0            # E* in V-headed

        eojeol_morph_info = []
        ambiguous = 0
        ambiguous_examples = []

        for sent in doc.sentences:
            for token in sent.tokens:  # eojeol
                tokobjs = self._eojeol_tokobjects(token, params)
                if not tokobjs:
                    continue

                all_eojeol_n += 1

                xpos_seq = [str(getattr(tok, "tag_", "UNK")) for tok in tokobjs]
                morph_n = len(tokobjs)

                # head type
                has_n = any(x.startswith("N") for x in xpos_seq)
                has_v = any(x.startswith("V") for x in xpos_seq)
                head = self._head_type(xpos_seq)

                if has_n and has_v:
                    ambiguous += 1
                    if debug and len(ambiguous_examples) < debug_n:
                        ambiguous_examples.append({
                            "eojeol": getattr(token, "text", ""),
                            "xpos_seq": xpos_seq,
                        })

                # Segmental
                all_morph_total += morph_n

                # Inflection counts
                j_count = sum(x.startswith("J") for x in xpos_seq)
                e_count = sum(x.startswith("E") for x in xpos_seq)
                all_inf_total += (j_count + e_count)

                # Split by head
                if head == "N":
                    n_eojeol_n += 1
                    n_morph_total += morph_n
                    n_inf_total += j_count
                elif head == "V":
                    v_eojeol_n += 1
                    v_morph_total += morph_n
                    v_inf_total += e_count

                eojeol_morph_info.append({
                    "eojeol": getattr(token, "text", ""),
                    "morph_n": morph_n,
                    "xpos_seq": xpos_seq,
                    "head": head,
                    "J_count": j_count,
                    "E_count": e_count,
                })

        if debug:
            print(f"[Compositionality] eojeols={all_eojeol_n}  N_head={n_eojeol_n}  V_head={v_eojeol_n}  neither={all_eojeol_n - n_eojeol_n - v_eojeol_n}")
            if ambiguous:
                print(f"[Compositionality] ambiguous(N+V in same eojeol)={ambiguous}")
                for ex in ambiguous_examples:
                    print("[Compositionality] ambiguous example:", ex)

        return {
            "all_eojeol_n": all_eojeol_n,
            "n_eojeol_n": n_eojeol_n,
            "v_eojeol_n": v_eojeol_n,
            "all_morph_total": all_morph_total,
            "n_morph_total": n_morph_total,
            "v_morph_total": v_morph_total,
            "all_inf_total": all_inf_total,
            "n_inf_total": n_inf_total,
            "v_inf_total": v_inf_total,
            "eojeol_morph_info": eojeol_morph_info,
        }

    def __init__(self, text=None, params=None, debug=True, debug_n=20):
        if text is None:
            self.AE_MC = None
            self.NE_MC = None
            self.VE_MC = None
            self.AE_FMC = None
            self.NE_FMC = None
            self.VE_FMC = None
            self.eojeol_morph_info = []
            self.vald = {
                "AE_MC": None, "NE_MC": None, "VE_MC": None,
                "AE_FMC": None, "NE_FMC": None, "VE_FMC": None,
            }
            return

        counts = self._collect_counts(text, params, debug=debug, debug_n=debug_n)
        all_e = counts["all_eojeol_n"]
        n_e = counts["n_eojeol_n"]
        v_e = counts["v_eojeol_n"]

        # Segmental compositionality
        self.AE_MC = self.safe_divide(counts["all_morph_total"], all_e)
        self.NE_MC = self.safe_divide(counts["n_morph_total"], n_e)
        self.VE_MC = self.safe_divide(counts["v_morph_total"], v_e)

        # Inflectional compositionality
        self.AE_FMC = self.safe_divide(counts["all_inf_total"], all_e)
        self.NE_FMC = self.safe_divide(counts["n_inf_total"], n_e)
        self.VE_FMC = self.safe_divide(counts["v_inf_total"], v_e)

        self.eojeol_morph_info = counts["eojeol_morph_info"]

        self.vald = {
            "AE_MC": self.AE_MC,
            "NE_MC": self.NE_MC,
            "VE_MC": self.VE_MC,
            "AE_FMC": self.AE_FMC,
            "NE_FMC": self.NE_FMC,
            "VE_FMC": self.VE_FMC,
            # optional denominators for diagnostics
            # "eojeols_n": all_e,
            # "N_head_eojeols_n": n_e,
            # "V_head_eojeols_n": v_e,
        }



################################
###  VocabGrade Indices ########
################################

class VocabularyGradeTypeIndices:
    """
    Vocabulary Grade + Type indices using:

      gradeLevel.sqlite:
        Level | Form | Type (NW/SW/MX/FL)

    Matching strategy:
      - morpheme_norm only (params.unit == "morpheme")
      - eojeol -> TokObject list
      - heuristic lemma candidate generation
      - DB Form lookup (NO POS used in DB)

    Outputs:
      - Vocab_meanLevel
      - Vocab_cov
      - Type_NW_prop / SW / MX / FL
      - Level_k_prop (for observed levels)
    """

    # --- XPOS groups (Sejong-style) ---
    VERB_BASE_TAGS = {"VV", "VX", "VCP", "VCN"}
    ADJ_BASE_TAGS  = {"VA"}
    DERIV_STEMS_PREFERRED_PREFIX = ("N", "XR")

    LIGHT_VERBS = ("하다", "되다", "시키다")
    LIGHT_ADJS  = ("스럽다",)

    def safe_divide(self, n, d):
        return n / d if d else 0.0

    def _mean(self, vals):
        return stat.mean(vals) if vals else 0.0

    # -------------------------------------
    # eojeol -> morpheme TokObjects
    # -------------------------------------
    def _eojeol_tokobjects(self, eojeol_token, params):
        norm_helper = Normalize(text=None, params=params)
        morph_units = norm_helper._expand_morpheme_units(eojeol_token, params)
        return [TokObject(unit, counter=0, params=params) for unit in morph_units]

    # -------------------------------------
    # Candidate generation (coverage booster)
    # -------------------------------------
    def _lemma_candidates_from_tokobjs(self, tokobjs, max_candidates=8):

        xpos_seq = [str(getattr(t, "tag_", "") or "") for t in tokobjs]
        lemmas   = [str(getattr(t, "lemma_", "") or "") for t in tokobjs]

        def first_by_prefix(prefixes):
            for lm, xp in zip(lemmas, xpos_seq):
                if lm and any(xp.startswith(p) for p in prefixes):
                    return lm
            return None

        def first_by_tagset(tagset):
            for lm, xp in zip(lemmas, xpos_seq):
                if xp in tagset and lm:
                    return lm
            return None

        def add_candidate(lst, s):
            s = (s or "").strip()
            if s and s not in seen:
                seen.add(s)
                lst.append(s)

        seen = set()
        cands = []

        stem = first_by_prefix(self.DERIV_STEMS_PREFERRED_PREFIX)
        if not stem:
            stem = first_by_prefix(("V",)) or first_by_prefix(("A",))

        has_deriv = any(x in {"XSV","XSA","XSN"} for x in xpos_seq)
        has_ha = any(lm in {"하","하다"} for lm in lemmas)

        # Productive light verb generation
        if stem and (has_deriv or has_ha):
            for lv in self.LIGHT_VERBS:
                add_candidate(cands, stem + lv)
            for la in self.LIGHT_ADJS:
                add_candidate(cands, stem + la)

        # Plain verb/adjective
        vbase = first_by_tagset(self.VERB_BASE_TAGS)
        abase = first_by_tagset(self.ADJ_BASE_TAGS)

        if vbase:
            add_candidate(cands, vbase if vbase.endswith("다") else vbase + "다")
        if abase:
            add_candidate(cands, abase if abase.endswith("다") else abase + "다")

        # Noun fallback
        nhead = first_by_prefix(("N",))
        if nhead:
            add_candidate(cands, nhead)

        if stem:
            add_candidate(cands, stem)

        return cands[:max_candidates]

    # -------------------------------------
    # Main matching
    # -------------------------------------
    def _collect_matches(self, text, params, debug=False):

        if params.unit != "morpheme":
            raise ValueError("Must use morpheme_norm (params.unit='morpheme').")

        doc = params.nlp(text)

        matched_levels = []
        matched_types = []

        matched_n = 0
        candidate_n = 0

        for sent in doc.sentences:
            for token in sent.tokens:

                tokobjs = self._eojeol_tokobjects(token, params)
                if not tokobjs:
                    continue

                cands = self._lemma_candidates_from_tokobjs(tokobjs)
                if not cands:
                    continue

                candidate_n += 1

                hit = None
                for cand in cands:
                    if cand in self.level_map:
                        hit = cand
                        break

                if hit:
                    matched_n += 1
                    matched_levels.append(self.level_map[hit])
                    matched_types.append(self.type_map.get(hit, ""))

        return matched_levels, matched_types, matched_n, candidate_n

    # -------------------------------------
    # Constructor
    # -------------------------------------
    def __init__(self, text=None, params=None, lex_db_path=None, lex_db_rows=None):

        if text is None:
            self.vald = {}
            return

        # ---- Load DB ----
        if lex_db_rows is None:
            if lex_db_path is not None:
                conn = sqlite3.connect(lex_db_path)
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT Form, Level, Type FROM grade_level")
                lex_db_rows = [dict(row) for row in cur.fetchall()]
                cur.close()
                conn.close()
            else:
                lex_db_rows = load_grade_level_rows()

        level_map = {}
        type_map = {}

        for r in lex_db_rows:
            form = str(r["Form"]).strip()
            lvl = int(r["Level"])
            typ = str(r.get("Type", "")).strip()

            if form not in level_map:
                level_map[form] = lvl
            else:
                level_map[form] = min(level_map[form], lvl)

            if form not in type_map:
                type_map[form] = typ

        self.level_map = level_map
        self.type_map = type_map

        # ---- Match ----
        matched_levels, matched_types, matched_n, cand_n = \
            self._collect_matches(text, params)

        # ---- Compute indices ----
        self.Vocab_meanLevel = self._mean(matched_levels)

        type_counts = Counter(matched_types)
        denom = sum(type_counts.values())

        self.Type_NW_prop = self.safe_divide(type_counts.get("NW", 0), denom)
        self.Type_SW_prop = self.safe_divide(type_counts.get("SW", 0), denom)
        self.Type_MX_prop = self.safe_divide(type_counts.get("MX", 0), denom)
        self.Type_FL_prop = self.safe_divide(type_counts.get("FL", 0), denom)

        level_counts = Counter(matched_levels)
        level_denom = sum(level_counts.values())
        all_levels = sorted(set(self.level_map.values()))

        self.Level_props = {
            f"Level_{k}_prop": self.safe_divide(level_counts.get(k, 0), level_denom)
            for k in all_levels
        }

        self.vald = {
            "Vocab_meanLevel": self.Vocab_meanLevel,
            "Type_NW_prop": self.Type_NW_prop,
            "Type_SW_prop": self.Type_SW_prop,
            "Type_MX_prop": self.Type_MX_prop,
            "Type_FL_prop": self.Type_FL_prop,
            **self.Level_props,
        }
