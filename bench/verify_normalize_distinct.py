"""Stalled-JVM reproduction for the normalize row's upstream figure.

Run with: pixi run python bench/verify_normalize_distinct.py

Same input and same normalization call as the README's 2,600-char row.
`bench.py` takes the best of five, so upstream could not be slowed by a
cached call there, yet one run reported 13.7 s for upstream. To show the
JVM itself sometimes takes over 10 s (not 14-19 s), the first call is
skipped on both sides and each side is timed on its next call only. Prints
timing on stdout, unbuffered.
"""

from __future__ import annotations

import time

from konlpy.tag import Okt as UpstreamOkt

from mojo_konlpy.tag import Okt as MojoOkt

TEXT = "".join(chr(0xAC00 + i) for i in range(2600))


def main() -> None:
    ours = MojoOkt()
    upstream = UpstreamOkt()
    # First calls: warm-up on both sides, parity asserted from the actual output.
    ours.normalize(TEXT)
    start = time.perf_counter()
    upstream_output = upstream.normalize(TEXT)
    print(f"upstream warm-up call: {time.perf_counter() - start:.2f} ms", flush=True)
    assert upstream_output == ours.normalize(TEXT), "parity failed"

    start = time.perf_counter()
    u = upstream.normalize(TEXT)
    upstream_seconds = time.perf_counter() - start
    start = time.perf_counter()
    o = ours.normalize(TEXT)
    mojo_seconds = time.perf_counter() - start
    assert u == o

    # Repeat with varied process state: same JVM, same input, new call.
    print(
        f"call A: upstream {upstream_seconds * 1000:.0f} ms, "
        f"mojo {mojo_seconds * 1000:.3f} ms",
        flush=True,
    )
    start = time.perf_counter()
    upstream.normalize(TEXT)
    upstream_2 = time.perf_counter() - start
    start = time.perf_counter()
    ours.normalize(TEXT)
    mojo_2 = time.perf_counter() - start
    print(
        f"call B: upstream {upstream_2 * 1000:.0f} ms, "
        f"mojo {mojo_2 * 1000:.3f} ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
