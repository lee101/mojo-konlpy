"""Where the morphology kernel spends its time, by input shape.

Run with `pixi run python bench/analysis.py`. These are the numbers behind the
README's parallelism section: how long one mkl_analyze call takes per chunk
length, and how big the chunks real callers actually pass are.

The host is shared. Every cell reports the quartile of a batch of calls
(median and lower quartile), so a loaded host is visible rather than averaged
into the number; a run is only worth recording when the two lines agree.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from mojo_konlpy._lib import address, library
from mojo_konlpy.tag import Okt, _chunks, _get_lexicon, POS_IDS


LEXICON = _get_lexicon()
OKT = Okt()

SENTENCE = (
    "유구한 역사와 전통에 빛나는 우리 대한국민은 민주공화국을 세우고 "
    "모든 국민의 자유와 권리를 보장한다"
)
PROSE_WORDS = [
    "유구한", "역사와", "전통에", "빛나는", "우리", "대한국민은",
    "민주공화국을", "세우고", "모든", "국민의", "자유와", "권리를", "보장한다",
]
NOISY = "이것도 되나욬ㅋㅋ 안됔ㅋㅋㅋ 머구뮤ㅠㅠㅠ 하즤 " * 100
SHORT_BATCH = [
    "단독입찰보다 복수입찰의 경우",
    "자연주의 쇼핑몰은 어떤 곳인가?",
    "영등포구청역에 있는 맛집 좀 알려주세요.",
    "새로운 스테밍을 추가했었다.",
] * 100


def kernel_call(text: str) -> int:
    """One direct mkl_analyze call over `text`; returns the token count."""
    chars = np.fromiter(map(ord, text), dtype=np.int64, count=len(text))
    n = len(text)
    starts, ends, poses, unknowns = (np.empty(n, dtype=np.int64) for _ in range(4))
    slots = (n + 1) * 5
    scratch_i = np.empty(slots * 21, dtype=np.int64)
    scratch_f = np.empty(slots * 2, dtype=np.float64)
    return int(library().mkl_analyze(
        address(chars), n, LEXICON.keys_addr, LEXICON.masks_addr,
        LEXICON.freqs_addr, LEXICON.capacity, address(starts), address(ends),
        address(poses), address(unknowns), n, address(scratch_i), scratch_i.size,
        address(scratch_f), scratch_f.size,
    ))


def quartiles(function, repeat: int = 21) -> tuple[float, float]:
    """(median, lower quartile) in microseconds."""
    function()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    ordered = sorted(samples)
    return statistics.median(samples) * 1e6, ordered[len(ordered) // 4] * 1e6


def table(rows, header) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print("| " + " | ".join(str(cell) for cell in row) + " |")


def section_kernel_by_length() -> None:
    print("### Direct mkl_analyze cost by chunk length (us)\n")
    rows = []
    for word in PROSE_WORDS[:6] + ["단독입찰보다"]:
        median, q1 = quartiles(lambda w=word: kernel_call(w))
        rows.append((word, len(word), f"{median:.1f}", f"{q1:.1f}"))
    for n in (8, 16, 32, 64, 112, 200, 400):
        median, q1 = quartiles(lambda n=n: kernel_call("가" * n))
        rows.append((f"'가' x {n}", n, f"{median:.1f}", f"{q1:.1f}"))
    table(rows, ("input", "chars", "median", "q1"))


def section_cache_hit_sizes() -> None:
    print("\n### Chunk sizes as real callers pass them\n")
    rows = []
    for label, text in (
        ("bench short batch x400", " ".join(SHORT_BATCH)),
        ("bench paragraph", (SENTENCE + " ") * 100),
        ("bench noisy", NOISY),
    ):
        chunks = [t.text for t in _chunks(text) if t.pos == POS_IDS["Korean"]]
        sizes = sorted(len(c) for c in chunks)
        over = sum(1 for s in sizes if s >= 128)
        rows.append((label, len(chunks), sizes[len(sizes) // 2] if sizes else 0,
                     sizes[-1] if sizes else 0, over))
    table(rows, ("workload", "korean chunks", "median chars", "max chars",
                 "chunks >= 128"))
    print("\nWhole method, cold cache (ms):")
    rows = []
    for label, function in (
        ("pos, 400 short sentences", lambda: [Okt().pos(t) for t in SHORT_BATCH]),
        ("pos, 100-sentence paragraph", lambda: Okt().pos((SENTENCE + " ") * 100)),
        ("pos(norm=True, stem=True), 400", lambda: [
            Okt().pos(t, norm=True, stem=True) for t in SHORT_BATCH]),
        ("normalize, 3,100 noisy", lambda: Okt().normalize(NOISY)),
        ("phrases, 400", lambda: [Okt().phrases(t) for t in SHORT_BATCH]),
    ):
        rows.append((label, f"{quartiles(function, repeat=7)[0] / 1e3:.2f}"))
    table(rows, ("workload", "ms"))


def main() -> None:
    section_kernel_by_length()
    section_cache_hit_sizes()


if __name__ == "__main__":
    main()
