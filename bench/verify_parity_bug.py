"""Minimal reproduction of a stable port/upstream parity divergence.

Run with: pixi run python bench/verify_parity_bug.py

Found while re-measuring: `Okt.pos("하지 겟", norm=True)` and
`Okt.pos("하지 겟")` disagree with KoNLPy on the token types, while
`"하지겟"` (no space) agrees on both sides. The port may also miss the
`하지 + 겟` split entirely. This script prints both sides, with the port's
memoised layouts cleared first, so a cached result cannot hide the bug.
"""

from __future__ import annotations

from konlpy.tag import Okt as UpstreamOkt

from mojo_konlpy import tag as mojo_tag
from mojo_konlpy.tag import Okt as MojoOkt

CASES = [
    ("하지 겟", {}),
    ("하지 겟", {"norm": True}),
    ("하지겟", {}),
    ("하지 겟", {"stem": True}),
    ("먹지 겟", {}),
]


def main() -> None:
    ours = MojoOkt()
    upstream = UpstreamOkt()
    failures = 0
    for text, kwargs in CASES:
        values = []
        for analyzer in (ours, upstream):
            mojo_tag._analyze_korean_layout.cache_clear()
            mojo_tag._normalize_korean_chunk.cache_clear()
            values.append(analyzer.pos(text, **kwargs))
        same = values[0] == values[1]
        failures += not same
        print(f"{text!r} {kwargs}: parity={same}")
        print(f"  mojo    : {values[0]}")
        print(f"  upstream: {values[1]}")
    print()
    print(f"divergent cases: {failures}/{len(CASES)}")


if __name__ == "__main__":
    main()
