"""KoNLPy-compatible Korean morphological analysis."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import gzip
from pathlib import Path
import re
from threading import Lock

import numpy as np

from ._lib import address, library


POS_NAMES = (
    "Noun", "Verb", "Adjective", "Adverb", "Determiner", "Exclamation",
    "Josa", "Eomi", "PreEomi", "Conjunction", "Modifier", "VerbPrefix", "Suffix",
    "Unknown", "Korean", "Foreign", "Number", "KoreanParticle", "Alpha",
    "Punctuation", "Hashtag", "ScreenName", "Email", "URL", "CashTag", "Space",
    "Others", "ProperNoun",
)
POS_IDS = {name: index for index, name in enumerate(POS_NAMES)}
DATA = Path(__file__).with_name("data")
_LEXICON_LOCK = Lock()


def _fnv1a(text: str) -> int:
    value = 1469598103934665603
    for char in text:
        value ^= ord(char)
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


class _Lexicon:
    def __init__(self) -> None:
        with np.load(DATA / "lexicon.npz") as archive:
            self.keys = archive["keys"]
            self.masks = archive["masks"]
            self.freqs = archive["freqs"]
        expected = (
            ("keys", self.keys, np.dtype(np.uint64)),
            ("masks", self.masks, np.dtype(np.int64)),
            ("freqs", self.freqs, np.dtype(np.float64)),
        )
        for name, array, dtype in expected:
            if array.dtype != dtype or array.ndim != 1 or not array.flags.c_contiguous:
                raise RuntimeError(
                    f"invalid lexicon {name}: expected contiguous 1-D {dtype}"
                )
        self.capacity = self.keys.size
        if (
            self.capacity == 0
            or self.capacity & (self.capacity - 1)
            or self.masks.size != self.capacity
            or self.freqs.size != self.capacity
        ):
            raise RuntimeError("invalid lexicon table dimensions")
        self.keys_addr = address(self.keys)
        self.masks_addr = address(self.masks)
        self.freqs_addr = address(self.freqs)
        self._stems: dict[tuple[int, str], str] | None = None
        self._typos: list[tuple[str, str]] | None = None

    @lru_cache(maxsize=16_384)
    def slot(self, word: str) -> int:
        key = _fnv1a(word)
        slot = key & (self.capacity - 1)
        while self.masks[slot]:
            if int(self.keys[slot]) == key:
                return slot
            slot = (slot + 1) & (self.capacity - 1)
        return -1

    def has(self, word: str, pos: int) -> bool:
        slot = self.slot(word)
        return slot >= 0 and bool(int(self.masks[slot]) & (1 << pos))

    def stem(self, word: str, pos: int) -> str:
        if self._stems is None:
            stems: dict[tuple[int, str], str] = {}
            with gzip.open(DATA / "stems.tsv.gz", "rt", encoding="utf-8") as stream:
                for line in stream:
                    pos_text, surface, stem = line.rstrip("\n").split("\t")
                    stems[(int(pos_text), surface)] = stem
            self._stems = stems
        return self._stems.get((pos, word), word)

    @property
    def typos(self) -> list[tuple[str, str]]:
        if self._typos is None:
            pairs = []
            for line in (DATA / "typos.txt").read_text(encoding="utf-8").splitlines():
                fields = line.strip().split()
                if len(fields) == 2:
                    pairs.append((fields[0], fields[1]))
            self._typos = sorted(pairs, key=lambda pair: len(pair[0]))
        return self._typos


_lexicon: _Lexicon | None = None


def _get_lexicon() -> _Lexicon:
    global _lexicon
    if _lexicon is None:
        with _LEXICON_LOCK:
            if _lexicon is None:
                _lexicon = _Lexicon()
    return _lexicon


@dataclass
class _Token:
    text: str
    pos: int
    offset: int
    unknown: bool = False


@dataclass
class _Phrase:
    tokens: list[_Token]
    pos: int

    @property
    def text(self) -> str:
        return "".join(token.text for token in self.tokens)

    @property
    def length(self) -> int:
        return sum(len(token.text) for token in self.tokens)


_CHUNK_PATTERNS = (
    ("URL", re.compile(r"(?:https?://|www\.)[^\s]+")),
    ("Email", re.compile(r"[\w.\-_]+@[\w.]+")),
    ("ScreenName", re.compile(r"@[\w가-힣_]+")),
    ("Hashtag", re.compile(r"#[\w가-힣_]+")),
    ("CashTag", re.compile(r"\$[A-Za-z][A-Za-z0-9_]*")),
    ("Number", re.compile(
        r"\$?\d+(?:,\d{3})*(?:[/~:.\-]\d+)?(?:천|만|억|조)*"
        r"(?:%|원|달러|위안|옌|엔|유로|등|년|월|일|회|시간|시|분|초)?"
    )),
    ("Korean", re.compile(r"[가-힣]+")),
    ("KoreanParticle", re.compile(r"[ㄱ-ㅣ]+")),
    ("Alpha", re.compile(r"[A-Za-z]+")),
    ("Punctuation", re.compile(r"[!-/:-@\[-`{-~·…’]+")),
)


def _chunks(text: str, keep_space: bool = False) -> list[_Token]:
    result: list[_Token] = []

    for segment in re.finditer(r"\s+|\S+", text):
        surface = segment.group()
        base = segment.start()
        if surface.isspace():
            if keep_space:
                result.append(_Token(surface, POS_IDS["Space"], base))
            continue
        accepted: list[tuple[int, int, str]] = []
        matched_length = 0
        for name, pattern in _CHUNK_PATTERNS:
            if matched_length >= len(surface):
                break
            for match in pattern.finditer(surface):
                start, end = match.span()
                disjoint = all(end <= old_start or start >= old_end for old_start, old_end, _ in accepted)
                if disjoint:
                    accepted.append((start, end, name))
                    matched_length += end - start
        accepted.sort()
        cursor = 0
        for start, end, name in accepted:
            if start > cursor:
                result.append(_Token(surface[cursor:start], POS_IDS["Foreign"], base + cursor))
            result.append(_Token(surface[start:end], POS_IDS[name], base + start))
            cursor = end
        if cursor < len(surface):
            result.append(_Token(surface[cursor:], POS_IDS["Foreign"], base + cursor))
    return result


def _collapse_nouns(tokens: list[_Token]) -> list[_Token]:
    result: list[_Token] = []
    collapsing = False
    for token in tokens:
        if token.pos == POS_IDS["Noun"] and len(token.text) == 1 and collapsing:
            previous = result[-1]
            result[-1] = _Token(
                previous.text + token.text, previous.pos, previous.offset, True
            )
        else:
            result.append(token)
            collapsing = token.pos == POS_IDS["Noun"] and len(token.text) == 1
    return result


@lru_cache(maxsize=16_384)
def _analyze_korean_layout(text: str) -> tuple[tuple[str, int, int, bool], ...]:
    lexicon = _get_lexicon()
    chars = np.fromiter(map(ord, text), dtype=np.int64, count=len(text))
    n = len(text)
    starts = np.empty(n, dtype=np.int64)
    ends = np.empty(n, dtype=np.int64)
    poses = np.empty(n, dtype=np.int64)
    unknowns = np.empty(n, dtype=np.int64)
    slots = (n + 1) * 5
    scratch_i = np.empty(slots * 21, dtype=np.int64)
    scratch_f = np.empty(slots * 2, dtype=np.float64)
    count = int(library().mkl_analyze(
        address(chars), n, lexicon.keys_addr, lexicon.masks_addr,
        lexicon.freqs_addr, lexicon.capacity, address(starts), address(ends),
        address(poses), address(unknowns), n, address(scratch_i), scratch_i.size,
        address(scratch_f), scratch_f.size,
    ))
    # Keep every NumPy owner alive until ctypes returns; the local references above
    # are deliberately not inlined into the call. The native function returns a
    # negative status for a rejected boundary contract.
    if count < 0:
        raise RuntimeError(f"Mojo morphology kernel rejected its buffers ({count})")
    if count > n:
        raise RuntimeError(
            f"Mojo morphology kernel returned {count} tokens for {n} characters"
        )
    for i in range(count):
        start = int(starts[i])
        end = int(ends[i])
        pos = int(poses[i])
        unknown = int(unknowns[i])
        if not (
            0 <= start < end <= n
            and 0 <= pos < len(POS_NAMES)
            and unknown in (0, 1)
        ):
            raise RuntimeError("Mojo morphology kernel returned invalid token metadata")
    tokens = [
        _Token(
            text[int(starts[i]):int(ends[i])],
            int(poses[i]),
            int(starts[i]),
            bool(unknowns[i]),
        )
        for i in range(count)
    ]
    return tuple(
        (token.text, token.pos, token.offset, token.unknown)
        for token in _collapse_nouns(tokens)
    )


def _analyze_korean(text: str, offset: int) -> list[_Token]:
    return [
        _Token(surface, pos, offset + relative_offset, unknown)
        for surface, pos, relative_offset, unknown in _analyze_korean_layout(text)
    ]


def _stem_tokens(tokens: list[_Token], use_stem: bool) -> list[_Token]:
    lexicon = _get_lexicon()
    result: list[_Token] = []
    predicates = {POS_IDS["Verb"], POS_IDS["Adjective"]}
    endings = {POS_IDS["Eomi"], POS_IDS["PreEomi"]}
    for token in tokens:
        if token.pos in endings and result and result[-1].pos in predicates:
            previous = result[-1]
            if not use_stem:
                result[-1] = _Token(
                    previous.text + token.text, previous.pos, previous.offset, previous.unknown
                )
        else:
            if use_stem and token.pos in predicates:
                token = _Token(
                    lexicon.stem(token.text, token.pos),
                    token.pos,
                    token.offset,
                    token.unknown,
                )
            result.append(token)
    return result


_ONSETS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_VOWELS = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_CODAS = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
_CODA_N_EXCEPTIONS = frozenset("은는운인텐근른픈닌든던")
_NORMALIZABLE = re.compile(r"[ㄱ-ㅣ가-힣]+")
_EMOTION = re.compile(r"([가-힣]+)(ㅋ+|ㅎ+|[ㅠㅜ]+)")
_REPEATED_CHAR = re.compile(r"(.)\1{3,}|[ㅠㅜ]{3,}")
_REPEATED_PAIR = re.compile(r"(..)(?:\1){2,}")
_WHITESPACE = re.compile(r"\s+")


def _decompose(char: str) -> tuple[str, str, str]:
    value = ord(char) - 0xAC00
    return _ONSETS[value // 588], _VOWELS[(value % 588) // 28], _CODAS[value % 28]


def _compose(onset: str, vowel: str, coda: str = " ") -> str:
    return chr(0xAC00 + _ONSETS.index(onset) * 588 + _VOWELS.index(vowel) * 28 + _CODAS.index(coda))


def _normalize_emotion(match: re.Match[str]) -> str:
    chunk, emotion = match.group(1), match.group(2)
    lexicon = _get_lexicon()
    if (
        lexicon.has(chunk, POS_IDS["Noun"])
        or lexicon.has(chunk[-1:], POS_IDS["Eomi"])
        or lexicon.has(chunk[-2:], POS_IDS["Eomi"])
    ):
        return chunk + emotion
    onset, vowel, coda = _decompose(chunk[-1])
    if coda in {"ㅋ", "ㅎ"}:
        chunk = chunk[:-1] + _compose(onset, vowel)
    elif len(chunk) > 1:
        prev_onset, prev_vowel, prev_coda = _decompose(chunk[-2])
        if prev_coda == " " and vowel == emotion[0] and onset in _CODAS:
            chunk = chunk[:-2] + _compose(prev_onset, prev_vowel, onset)
    return chunk + emotion


def _normalize_coda_n(chunk: str) -> str:
    if len(chunk) < 2:
        return chunk
    lexicon = _get_lexicon()
    head, last = chunk[-2], chunk[-1]
    if (
        lexicon.has(chunk, POS_IDS["Noun"])
        or lexicon.has(chunk, POS_IDS["Conjunction"])
        or lexicon.has(chunk, POS_IDS["Adverb"])
        or lexicon.has(chunk[-2:], POS_IDS["Noun"])
        or not ("가" <= head <= "힣")
        or head in _CODA_N_EXCEPTIONS
    ):
        return chunk
    onset, vowel, coda = _decompose(head)
    new_head = chunk[:-2] + _compose(onset, vowel)
    if coda == "ㄴ" and last in {"데", "가", "지"} and lexicon.has(new_head, POS_IDS["Noun"]):
        return new_head + "인" + last
    return chunk


@lru_cache(maxsize=4_096)
def _normalize_korean_chunk(chunk: str) -> str:
    value = _EMOTION.sub(_normalize_emotion, chunk)
    value = _REPEATED_CHAR.sub(lambda match: match.group()[:3], value)
    value = _REPEATED_PAIR.sub(lambda match: match.group()[:4], value)
    value = _normalize_coda_n(value)
    for typo, correction in _get_lexicon().typos:
        if typo in value:
            value = value.replace(typo, correction)
    return _WHITESPACE.sub(" ", value)


def _collapse_pos(tokens: list[_Token]) -> list[_Phrase]:
    result: list[_Phrase] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.pos in {POS_IDS["Determiner"], POS_IDS["Modifier"], POS_IDS["Noun"]}:
            end = index
            if tokens[end].pos == POS_IDS["Determiner"]:
                end += 1
            while end < len(tokens) and tokens[end].pos == POS_IDS["Modifier"]:
                end += 1
            if end < len(tokens) and tokens[end].pos == POS_IDS["Noun"]:
                end += 1
                if end < len(tokens) and tokens[end].pos == POS_IDS["Suffix"]:
                    end += 1
                result.append(_Phrase(tokens[index:end], POS_IDS["Noun"]))
                index = end
                continue
        if token.pos in {POS_IDS["Number"], POS_IDS["Alpha"]}:
            end = index + 1
            while end < len(tokens) and tokens[end].pos in {
                POS_IDS["Number"], POS_IDS["Alpha"],
            }:
                end += 1
            result.append(_Phrase(tokens[index:end], POS_IDS["Noun"]))
            index = end
            continue
        if token.pos == POS_IDS["VerbPrefix"]:
            end = index
            while end < len(tokens) and tokens[end].pos == POS_IDS["VerbPrefix"]:
                end += 1
            if end < len(tokens) and tokens[end].pos in {
                POS_IDS["Verb"], POS_IDS["Adjective"],
            }:
                result.append(_Phrase(tokens[index:end + 1], tokens[end].pos))
                index = end + 1
                continue
        result.append(_Phrase([token], token.pos))
        index += 1
    return result


def _trim_phrase_chunk(chunk: list[_Phrase]) -> list[_Phrase]:
    heads = {
        POS_IDS["Verb"], POS_IDS["Adjective"], POS_IDS["Noun"], POS_IDS["ProperNoun"],
        POS_IDS["Alpha"], POS_IDS["Number"],
    }
    tails = {
        POS_IDS["Noun"], POS_IDS["ProperNoun"], POS_IDS["Alpha"], POS_IDS["Number"],
    }
    while chunk and chunk[0].pos not in heads:
        chunk = chunk[1:]
    while chunk and chunk[-1].pos not in tails:
        chunk = chunk[:-1]
    while chunk and chunk[0].pos == POS_IDS["Space"]:
        chunk = chunk[1:]
    while chunk and chunk[-1].pos == POS_IDS["Space"]:
        chunk = chunk[:-1]
    return chunk


def _phrase_text(chunk: list[_Phrase]) -> str:
    return "".join(phrase.text for phrase in chunk).strip()


def _extract_phrases(tokens: list[_Token]) -> list[str]:
    phrases = _collapse_pos(tokens)
    noun_collapsed: list[_Phrase] = []
    noun_buffer: list[_Phrase] = []
    for phrase in phrases:
        if phrase.pos in {POS_IDS["Noun"], POS_IDS["ProperNoun"]}:
            noun_buffer.append(phrase)
        else:
            if noun_buffer:
                noun_collapsed.append(_Phrase(
                    [token for item in noun_buffer for token in item.tokens],
                    POS_IDS["Noun"],
                ))
                noun_buffer = []
            noun_collapsed.append(phrase)
    if noun_buffer:
        noun_collapsed.append(_Phrase(
            [token for item in noun_buffer for token in item.tokens], POS_IDS["Noun"]
        ))

    phrase_tokens = {POS_IDS["Noun"], POS_IDS["ProperNoun"], POS_IDS["Space"]}
    conjunction_josa = {"와", "과", "의"}
    candidates: list[list[_Phrase]] = []
    buffer: list[_Phrase] = []
    for phrase in noun_collapsed:
        is_modifying = False
        if phrase.pos in {POS_IDS["Verb"], POS_IDS["Adjective"]} and phrase.text:
            last = phrase.text[-1]
            if "가" <= last <= "힣":
                is_modifying = _decompose(last)[2] in {"ㄹ", "ㄴ"} and last != "만"
        is_non_noun = (
            is_modifying
            or (phrase.pos == POS_IDS["Josa"] and phrase.text in conjunction_josa)
            or phrase.pos in {POS_IDS["Alpha"], POS_IDS["Number"]}
        )
        if phrase.pos in phrase_tokens:
            buffer.append(phrase)
            if phrase.pos in {POS_IDS["Noun"], POS_IDS["ProperNoun"]}:
                candidates.append(list(buffer))
        elif is_non_noun:
            buffer.append(phrase)
        else:
            if buffer:
                candidates.append(list(buffer))
            buffer = []
    if buffer:
        candidates.append(list(buffer))

    singles = [
        [phrase] for phrase in phrases
        if phrase.pos in {POS_IDS["Noun"], POS_IDS["ProperNoun"]}
        and phrase.length >= 2
    ]
    candidates.extend(singles)
    hashtags = [
        token.text for token in tokens
        if token.pos in {POS_IDS["Hashtag"], POS_IDS["CashTag"]}
    ]

    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = _trim_phrase_chunk(candidate)
        without_spaces = [item for item in candidate if item.pos != POS_IDS["Space"]]
        if not without_spaces:
            continue
        if len(without_spaces) > 8 or sum(item.length for item in without_spaces) > 30:
            continue
        if (
            len(without_spaces) < 3
            and sum(item.length for item in without_spaces) < 2
        ):
            continue
        if not any(item.length > 1 for item in without_spaces):
            continue
        if candidate[-1].tokens[-1].pos == POS_IDS["Suffix"] and candidate[-1].tokens[-1].text == "적":
            continue
        text = _phrase_text(candidate)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    for text in hashtags:
        if text not in seen:
            result.append(text)
            seen.add(text)
    return result


class Okt:
    """Standalone implementation of KoNLPy's ``konlpy.tag.Okt`` API."""

    def __init__(self, jvmpath=None, max_heap_size=1024):
        del jvmpath, max_heap_size
        _get_lexicon()
        self.tagset = {
            "Adjective": "형용사", "Adverb": "부사", "Alpha": "알파벳",
            "Conjunction": "접속사", "Determiner": "관형사", "Eomi": "어미",
            "Exclamation": "감탄사", "Foreign": "외국어, 한자 및 기타기호",
            "Hashtag": "트위터 해쉬태그", "Josa": "조사",
            "KoreanParticle": "(ex: ㅋㅋ)", "Noun": "명사", "Number": "숫자",
            "PreEomi": "선어말어미", "Punctuation": "구두점",
            "ScreenName": "트위터 아이디", "Suffix": "접미사",
            "Unknown": "미등록어", "Verb": "동사",
        }

    def _tokenize(self, phrase: str, norm: bool, stem: bool, keep_space: bool = False):
        assert isinstance(phrase, str), "phrase input should be string, not %s" % type(phrase)
        if not phrase:
            return []
        text = self.normalize(phrase) if norm else phrase
        tokens: list[_Token] = []
        for chunk in _chunks(text, keep_space=keep_space):
            if chunk.pos == POS_IDS["Korean"]:
                tokens.extend(_analyze_korean(chunk.text, chunk.offset))
            else:
                tokens.append(chunk)
        return _stem_tokens(tokens, stem)

    def pos(self, phrase, norm=False, stem=False, join=False):
        tokens = self._tokenize(phrase, norm=norm, stem=stem)
        if join:
            return [f"{token.text}/{POS_NAMES[token.pos]}" for token in tokens]
        return [(token.text, POS_NAMES[token.pos]) for token in tokens]

    def nouns(self, phrase):
        return [surface for surface, pos in self.pos(phrase) if pos == "Noun"]

    def morphs(self, phrase, norm=False, stem=False):
        return [surface for surface, _ in self.pos(phrase, norm=norm, stem=stem)]

    def phrases(self, phrase):
        tokens = self._tokenize(phrase, norm=False, stem=False, keep_space=True)
        return _extract_phrases(tokens)

    def normalize(self, phrase):
        if not isinstance(phrase, str):
            raise TypeError("phrase input should be string, not %s" % type(phrase))
        return _NORMALIZABLE.sub(
            lambda match: _normalize_korean_chunk(match.group()), phrase
        )
