"""Measure mojo-konlpy and upstream KoNLPy on identical Korean text."""

from __future__ import annotations

import math
import platform
import subprocess
import time

from konlpy.tag import Okt as UpstreamOkt
from mojo_konlpy.tag import Okt as MojoOkt


def best_time(function, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def machine() -> str:
    try:
        model = subprocess.check_output(
            ["lscpu"], text=True
        )
        for line in model.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    ours = MojoOkt()
    upstream = UpstreamOkt()
    sentence = (
        "유구한 역사와 전통에 빛나는 우리 대한국민은 민주공화국을 세우고 "
        "모든 국민의 자유와 권리를 보장한다."
    )
    noisy = "이것도 되나욬ㅋㅋ 안됔ㅋㅋㅋ 머구뮤ㅠㅠㅠ 하즤 " * 100
    short_batch = [
        "단독입찰보다 복수입찰의 경우",
        "자연주의 쇼핑몰은 어떤 곳인가?",
        "영등포구청역에 있는 맛집 좀 알려주세요.",
        "새로운 스테밍을 추가했었다.",
    ] * 100
    paragraph = (sentence + " ") * 100

    cases = [
        (
            "pos, 400 short sentences",
            lambda: [ours.pos(text) for text in short_batch],
            lambda: [upstream.pos(text) for text in short_batch],
        ),
        (
            "pos, 100-sentence paragraph",
            lambda: ours.pos(paragraph),
            lambda: upstream.pos(paragraph),
        ),
        (
            "pos(norm=True, stem=True), 400 calls",
            lambda: [ours.pos(text, norm=True, stem=True) for text in short_batch],
            lambda: [upstream.pos(text, norm=True, stem=True) for text in short_batch],
        ),
        (
            "normalize, 3,100 noisy characters",
            lambda: ours.normalize(noisy),
            lambda: upstream.normalize(noisy),
        ),
        (
            "phrases, 400 calls",
            lambda: [ours.phrases(text) for text in short_batch],
            lambda: [upstream.phrases(text) for text in short_batch],
        ),
    ]

    print(f"Machine: {machine()}")
    print(f"Python: {platform.python_version()}")
    print()
    print("| workload | mojo-konlpy | KoNLPy | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, mojo_call, upstream_call in cases:
        expected = upstream_call()
        actual = mojo_call()
        if actual != expected:
            raise AssertionError(f"benchmark parity failed for {name}")
        mojo_call()
        upstream_call()
        mojo_seconds = best_time(mojo_call)
        upstream_seconds = best_time(upstream_call)
        ratio = upstream_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{upstream_seconds * 1000:.2f} ms | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()
