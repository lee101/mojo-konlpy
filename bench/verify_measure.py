"""VERIFY-phase independent re-measurement of the README benchmark table.

Run with: pixi run python bench/verify_measure.py

Differences from bench/bench.py, deliberately, so the numbers are an
independent check rather than a rerun of the same code path:

* both callables receive the *same* argument objects (the input texts are
  built once, outside the timed call); each side only differs in the library
  that consumes them;
* caches on the port side are cleared before every timed run, so a row cannot
  win by replaying a memoised result;
* each side is timed `rounds` times and the full range, median and best are
  reported, instead of a single best-of-five;
* parity (complete output equality) is asserted for every round's inputs.

It prints one machine-readable block that bench/verify.md quotes.
"""

from __future__ import annotations

import json
import math
import platform
import statistics
import time

from konlpy.tag import Okt as UpstreamOkt

from mojo_konlpy import tag as mojo_tag
from mojo_konlpy.tag import Okt as MojoOkt

ROUNDS = 9

SENTENCE = (
    "유구한 역사와 전통에 빛나는 우리 대한국민은 민주공화국을 세우고 "
    "모든 국민의 자유와 권리를 보장한다."
)
SHORT_BATCH = tuple([
    "단독입찰보다 복수입찰의 경우",
    "자연주의 쇼핑몰은 어떤 곳인가?",
    "영등포구청역에 있는 맛집 좀 알려주세요.",
    "새로운 스테밍을 추가했었다.",
] * 100)
NOISY = "이것도 되나욬ㅋㅋ 안됔ㅋㅋㅋ 머구뮤ㅠㅠㅠ 하즤 " * 100
PARAGRAPH = (SENTENCE + " ") * 100


def clear_caches() -> None:
    mojo_tag._analyze_korean_layout.cache_clear()
    mojo_tag._normalize_korean_chunk.cache_clear()
    mojo_tag._get_lexicon().slot.cache_clear()


def timed(function, clears: bool) -> list[float]:
    samples = []
    for _ in range(ROUNDS):
        if clears:
            clear_caches()
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def summarise(samples: list[float]) -> dict[str, float]:
    return {
        "best": min(samples),
        "median": statistics.median(samples),
        "worst": max(samples),
    }


def main() -> None:
    ours = MojoOkt()
    upstream = UpstreamOkt()

    cases = [
        (
            "pos, 400 short sentences",
            lambda: [ours.pos(text) for text in SHORT_BATCH],
            lambda: [upstream.pos(text) for text in SHORT_BATCH],
        ),
        (
            "pos, 100-sentence paragraph",
            lambda: ours.pos(PARAGRAPH),
            lambda: upstream.pos(PARAGRAPH),
        ),
        (
            "pos(norm=True, stem=True), 400 calls",
            lambda: [ours.pos(text, norm=True, stem=True) for text in SHORT_BATCH],
            lambda: [upstream.pos(text, norm=True, stem=True) for text in SHORT_BATCH],
        ),
        (
            "normalize, 3,100 noisy characters",
            lambda: ours.normalize(NOISY),
            lambda: upstream.normalize(NOISY),
        ),
        (
            "phrases, 400 calls",
            lambda: [ours.phrases(text) for text in SHORT_BATCH],
            lambda: [upstream.phrases(text) for text in SHORT_BATCH],
        ),
    ]

    print(f"machine: {platform.processor() or platform.machine()}")
    print(f"python: {platform.python_version()}")
    print(f"rounds per side: {ROUNDS} (cache cleared before each port round)")
    results = {}
    for name, mojo_call, upstream_call in cases:
        clear_caches()
        expected = upstream_call()
        actual = mojo_call()
        assert actual == expected, f"parity failed: {name}"
        # Warm-up on both sides (JVM JIT, import-time lazy state).
        clear_caches()
        mojo_call()
        upstream_call()
        mojo_samples = timed(mojo_call, clears=True)
        upstream_samples = timed(upstream_call, clears=False)
        mojo_stats = summarise(mojo_samples)
        upstream_stats = summarise(upstream_samples)
        ratio_best = upstream_stats["best"] / mojo_stats["best"]
        ratio_median = upstream_stats["median"] / mojo_stats["median"]
        ratio_worst = upstream_stats["worst"] / mojo_stats["worst"]
        results[name] = {
            "mojo": mojo_stats,
            "upstream": upstream_stats,
            "ratio_best": ratio_best,
            "ratio_median": ratio_median,
            "ratio_worst": ratio_worst,
            "parity": True,
        }
        print()
        print(f"## {name}")
        print(
            f"mojo     best={mojo_stats['best']:.2f} median={mojo_stats['median']:.2f} "
            f"worst={mojo_stats['worst']:.2f} ms"
        )
        print(
            f"upstream best={upstream_stats['best']:.2f} median={upstream_stats['median']:.2f} "
            f"worst={upstream_stats['worst']:.2f} ms"
        )
        print(
            f"ratio    best={ratio_best:.2f}x median={ratio_median:.2f}x "
            f"worst={ratio_worst:.2f}x"
        )
    print()
    print("JSON " + json.dumps(results))


if __name__ == "__main__":
    main()
