"""Does the JVM-side normalize cache in this KoNLPy build, and by how much?

Run with: pixi run python bench/verify_normalize_probe.py

The README's normalize row times one large string that consists of five
distinct Korean chunks repeated 100 times. Both the port and (apparently) the
JVM memoise per chunk, so the timed call may be mostly cache replay. This
probe measures the same normalizer on (a) the benchmark string and (b) the
same chunks addressed as distinct strings, to expose the cache's effect.

It asserts nothing about timing; it prints numbers only.
"""

from __future__ import annotations

import statistics
import time

from konlpy.tag import Okt as UpstreamOkt

from mojo_konlpy import tag as mojo_tag
from mojo_konlpy.tag import Okt as MojoOkt

NOISY = "이것도 되나욬ㅋㅋ 안됔ㅋㅋㅋ 머구뮤ㅠㅠㅠ 하즤 " * 100
ROUNDS = 21


def time_it(function: "callable", rounds: int = ROUNDS, clear: "callable | None" = None):
    samples = []
    for _ in range(rounds):
        if clear is not None:
            clear()
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(samples)
    return min(ordered), statistics.median(ordered)


def main() -> None:
    ours = MojoOkt()
    upstream = UpstreamOkt()
    chunk_count = len(ours.normalize(NOISY))
    assert ours.normalize(NOISY) == upstream.normalize(NOISY)

    # Forty distinct strings, each of the benchmark's shape: if a side caches
    # per chunk, distinct first-characters evade it.
    distractors = [f"아{i:02d} 이것도 되나욬ㅋㅋ 안됔ㅋㅋㅋ 머구뮤ㅠㅠㅠ 하즤" for i in range(40)]

    def upstream_distinct() -> None:
        for text in distractors:
            upstream.normalize(text)

    def mojo_distinct() -> None:
        for text in distractors:
            ours.normalize(text)

    clears = lambda: (
        mojo_tag._normalize_korean_chunk.cache_clear(),
        mojo_tag._get_lexicon().slot.cache_clear(),
    )

    rows = [
        ("upstream, benchmark string", lambda: upstream.normalize(NOISY), None),
        ("mojo, benchmark string, cache cleared", lambda: ours.normalize(NOISY), clears),
        ("mojo, benchmark string, warm cache", lambda: ours.normalize(NOISY), None),
        ("upstream, 40 distinct strings", upstream_distinct, None),
        ("mojo, 40 distinct strings, cache cleared", mojo_distinct, clears),
    ]
    print(f"normalized output length: {chunk_count} chars (input {len(NOISY)} chars, "
          f"5 distinct Korean chunks)")
    for label, function, clear in rows:
        best, median = time_it(function, clear=clear)
        print(f"{label}: best {best:.2f} ms, median {median:.2f} ms")


if __name__ == "__main__":
    main()
