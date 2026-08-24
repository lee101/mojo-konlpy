"""Behavioral parity with KoNLPy's Open Korean Text wrapper."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from mojo_konlpy._lib import address, library
from mojo_konlpy.tag import Okt, _analyze_korean_layout, _normalize_korean_chunk

konlpy_tag = pytest.importorskip("konlpy.tag")


@pytest.fixture(scope="module")
def analyzers():
    return Okt(), konlpy_tag.Okt()


PUBLISHED_TEXTS = [
    "단독입찰보다 복수입찰의 경우",
    "유일하게 항공기 체계 종합개발 경험을 갖고 있는 KAI는",
    "날카로운 분석과 신뢰감 있는 진행으로",
    "이것도 되나욬ㅋㅋ",
]


@pytest.mark.parametrize("text", PUBLISHED_TEXTS)
def test_pos_published_examples(analyzers, text):
    ours, upstream = analyzers
    assert ours.pos(text) == upstream.pos(text)


@pytest.mark.parametrize("text", PUBLISHED_TEXTS)
def test_morphs_published_examples(analyzers, text):
    ours, upstream = analyzers
    assert ours.morphs(text) == upstream.morphs(text)


@pytest.mark.parametrize("text", PUBLISHED_TEXTS)
def test_nouns_published_examples(analyzers, text):
    ours, upstream = analyzers
    assert ours.nouns(text) == upstream.nouns(text)


@pytest.mark.parametrize("text", PUBLISHED_TEXTS)
def test_phrases_published_examples(analyzers, text):
    ours, upstream = analyzers
    assert ours.phrases(text) == upstream.phrases(text)


@pytest.mark.parametrize(
    "text",
    [
        "이것도 되나욬ㅋㅋ",
        "한국어를 처리하는 예시입니다.",
        "새로운 스테밍을 추가했었다.",
        "영등포구청역에 있는 맛집 좀 알려주세요.",
    ],
)
def test_norm_and_stem_parity(analyzers, text):
    ours, upstream = analyzers
    assert ours.pos(text, norm=True, stem=True) == upstream.pos(
        text, norm=True, stem=True
    )


@pytest.mark.parametrize(
    "text",
    [
        "안됔ㅋㅋㅋ 머구뮤ㅠㅠㅠ 하즤",
        "ㅋㅋㅋㅋㅋㅋ 재밌다아아아!!!",
        "훌쩍훌쩍훌쩍훌쩍",
        "소린가",
    ],
)
def test_normalize_parity(analyzers, text):
    ours, upstream = analyzers
    assert ours.normalize(text) == upstream.normalize(text)


def test_normalize_reuses_repeated_chunks(analyzers):
    ours, upstream = analyzers
    text = "안됔ㅋㅋㅋ 안됔ㅋㅋㅋ 안됔ㅋㅋㅋ"
    _normalize_korean_chunk.cache_clear()
    assert ours.normalize(text) == upstream.normalize(text)
    cache = _normalize_korean_chunk.cache_info()
    assert cache.misses == 1
    assert cache.hits == 2


def test_morphology_reuses_repeated_chunks(analyzers):
    ours, upstream = analyzers
    text = "대한민국 대한민국 대한민국"
    _analyze_korean_layout.cache_clear()
    assert ours.pos(text) == upstream.pos(text)
    cache = _analyze_korean_layout.cache_info()
    assert cache.misses == 1
    assert cache.hits == 2


@pytest.mark.parametrize("length", range(1, 11))
def test_simd_initialization_vector_and_tail_parity(analyzers, length):
    ours, upstream = analyzers
    text = "가" * length
    _analyze_korean_layout.cache_clear()
    assert ours.pos(text) == upstream.pos(text)


def test_join_parity(analyzers):
    ours, upstream = analyzers
    text = "자연주의 쇼핑몰은 어떤 곳인가?"
    assert ours.pos(text, join=True) == upstream.pos(text, join=True)


def test_non_korean_chunk_parity(analyzers):
    ours, upstream = analyzers
    text = "abc 123,000원 #한국 @user test@example.com https://example.com/a"
    assert ours.pos(text) == upstream.pos(text)


def test_number_overlap_quirks_match(analyzers):
    ours, upstream = analyzers
    text = "제30조제2항 및 12일에"
    assert ours.pos(text) == upstream.pos(text)


def test_foreign_and_punctuation_parity(analyzers):
    ours, upstream = analyzers
    text = "유지․발전, 정의·인도"
    assert ours.pos(text) == upstream.pos(text)


def test_long_korean_chunk_parity(analyzers):
    ours, upstream = analyzers
    text = "가나다라마바사아자차카타파하입니다"
    assert ours.pos(text) == upstream.pos(text)


CORPUS_LINES = [
    "대한민국은 민주공화국이다.",
    "모든 국민은 법 앞에 평등하다.",
    "국가는 사회보장·사회복지의 증진에 노력할 의무를 진다.",
    "헌법에 의하여 체결·공포된 조약과 일반적으로 승인된 국제법규는 국내법과 같은 효력을 가진다.",
    "정당의 설립은 자유이며, 복수정당제는 보장된다.",
    "누구든지 체포 또는 구속을 당한 때에는 즉시 변호인의 조력을 받을 권리를 가진다.",
    "국회의원은 국가이익을 우선하여 양심에 따라 직무를 행한다.",
    "대통령은 국무회의의 의장이 되고, 국무총리는 부의장이 된다.",
]


@pytest.mark.parametrize("text", CORPUS_LINES)
def test_real_corpus_line_parity(analyzers, text):
    ours, upstream = analyzers
    assert ours.pos(text) == upstream.pos(text)


def test_tagset_parity(analyzers):
    ours, upstream = analyzers
    assert ours.tagset == upstream.tagset


def test_constructor_compatibility_arguments():
    assert Okt(jvmpath="/ignored/for/compatibility", max_heap_size=1).pos("한국")


def test_empty_input_parity(analyzers):
    ours, upstream = analyzers
    assert ours.pos("") == upstream.pos("") == []
    assert ours.morphs("") == upstream.morphs("") == []
    assert ours.nouns("") == upstream.nouns("") == []


@pytest.mark.parametrize("value", [None, 42, ["한국어"]])
def test_invalid_input_parity(analyzers, value):
    ours, upstream = analyzers
    with pytest.raises(AssertionError):
        upstream.pos(value)
    with pytest.raises(AssertionError):
        ours.pos(value)


def test_shared_library_exports_kernel():
    root = Path(__file__).resolve().parents[1]
    library = ctypes.CDLL(str(root / "dist" / "libmojo-konlpy.so"))
    assert library.mkl_analyze is not None


def test_ffi_rejects_null_pointers_before_dereference():
    assert library().mkl_analyze(*([0] * 15)) == -1


def test_ffi_rejects_short_scratch_buffers():
    buffers = [np.ones(1, dtype=np.int64) for _ in range(8)]
    frequency = np.ones(1, dtype=np.float64)
    scratch_float = np.ones(1, dtype=np.float64)
    result = library().mkl_analyze(
        address(buffers[0]), 1,
        address(buffers[1].astype(np.uint64)), address(buffers[2]),
        address(frequency), 1,
        address(buffers[3]), address(buffers[4]), address(buffers[5]),
        address(buffers[6]), 1, address(buffers[7]), 1,
        address(scratch_float), 1,
    )
    assert result == -2


def test_address_rejects_wrong_layout_and_type():
    with pytest.raises(TypeError):
        address([1, 2])
    with pytest.raises(ValueError):
        address(np.ones((2, 2), dtype=np.int64))
