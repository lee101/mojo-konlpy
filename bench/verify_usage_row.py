"""Benchmark the row bench/bench.py omits: the README's Usage example.

Run with: pixi run python bench/verify_usage_row.py

`print(okt.pos("한국어를 처리하는 예시입니다."))` from the README is the first
thing a reader runs, and it is a cold single call. The benchmark table has no
cold-start row, so this measures one: cache cleared, both sides, identical
input. It also reports the JVM startup that the port avoids, since that is the
real-world shape of the claim.
"""

from __future__ import annotations

import statistics
import subprocess
import time

from konlpy.tag import Okt as UpstreamOkt

from mojo_konlpy import tag as mojo_tag
from mojo_konlpy.tag import Okt as MojoOkt

TEXT = "한국어를 처리하는 예시입니다."
ROUNDS = 15


def clears() -> None:
    mojo_tag._analyze_korean_layout.cache_clear()
    mojo_tag._normalize_korean_chunk.cache_clear()
    mojo_tag._get_lexicon().slot.cache_clear()


def main() -> None:
    ours = MojoOkt()
    upstream = UpstreamOkt()
    clears()
    assert ours.pos(TEXT) == upstream.pos(TEXT)

    mojo_samples = []
    upstream_samples = []
    for _ in range(ROUNDS):
        clears()
        start = time.perf_counter()
        ours.pos(TEXT)
        mojo_samples.append((time.perf_counter() - start) * 1e6)
    for _ in range(ROUNDS):
        start = time.perf_counter()
        upstream.pos(TEXT)
        upstream_samples.append((time.perf_counter() - start) * 1e6)

    for label, samples in (("mojo", mojo_samples), ("upstream", upstream_samples)):
        ordered = sorted(samples)
        print(
            f"{label}: best {ordered[0]:.1f} us, median {statistics.median(ordered):.1f} us"
        )

    print()
    print("separate-process cold start for `import` + first pos() (best of 3):")
    for label, snippet in (
        (
            "mojo",
            "from mojo_konlpy.tag import Okt\nOkt().pos(%r)" % TEXT,
        ),
        (
            "upstream",
            "from konlpy.tag import Okt\nOkt().pos(%r)" % TEXT,
        ),
    ):
        best = None
        for _ in range(3):
            start = time.perf_counter()
            subprocess.run(
                ["python", "-c", snippet], check=True, capture_output=True
            )
            elapsed = time.perf_counter() - start
            best = elapsed if best is None else min(best, elapsed)
        print(f"  {label}: {best * 1000:.0f} ms")


if __name__ == "__main__":
    main()
