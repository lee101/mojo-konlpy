"""Count how many differential (port vs upstream) assertions the suite makes.

Run with: pixi run python bench/verify_test_coverage.py

Used by bench/verify.md to state, rather than guess, how much of the test
suite is real parity checking. It instantiates the two analyzers, instruments
them so every call is recorded, and reports:
  * the number and names of tests that never call an upstream analyzer;
  * the number of distinct (method, arguments) comparisons that touch upstream;
  * which method is exercised least.
It does not modify the tests.
"""

from __future__ import annotations

import collections
import importlib
import sys
from pathlib import Path

import konlpy.tag as konlpy_tag

import mojo_konlpy.tag as mojo_tag

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
test_parity = importlib.import_module("test_parity")

UPSTREAM_METHODS = ("pos", "morphs", "nouns", "phrases", "normalize", "tagset")


class Recorder:
    def __init__(self, inner):
        self.inner = inner
        self.calls: collections.Counter[str] = collections.Counter()
        self.keys: set[tuple] = set()

    def __getattr__(self, name):
        if name == "tagset":
            self.calls["tagset"] += 1
            self.keys.add(("tagset",))
            return self.inner.tagset
        if name not in UPSTREAM_METHODS:
            return getattr(self.inner, name)
        target = getattr(self.inner, name)

        def call(*args, **kwargs):
            self.calls[name] += 1
            self.keys.add((name, repr(args), repr(sorted(kwargs.items()))))
            return target(*args, **kwargs)

        return call


def main() -> None:
    ours = Recorder(mojo_tag.Okt())
    upstream = Recorder(konlpy_tag.Okt())

    def analyzers():
        return ours, upstream
    names = [name for name in dir(test_parity) if name.startswith("test_")]
    assert names, "no tests found"
    compared = 0
    skipped = []
    ids = []
    for name in names:
        function = getattr(test_parity, name)
        takes_analyzers = "analyzers" in function.__code__.co_varnames[
            : function.__code__.co_argcount
        ]
        marks = getattr(function, "pytestmark", [])
        cases = [()]
        for mark in marks:
            if mark.name == "parametrize":
                _, values = mark.args
                cases = [(value,) for value in values]
        for case in cases:
            before = sum(upstream.calls.values())
            function(*(analyzers(), *case) if takes_analyzers else case)
            if sum(upstream.calls.values()) > before:
                compared += 1
            else:
                skipped.append(name)
            ids.append(name)

    print(f"test functions: {len(names)}")
    print(f"test cases executed (parametrised expanded): {len(ids)}")
    print(f"cases that called upstream at least once: {compared}")
    print(f"cases that never called upstream: {len(skipped)}")
    for name in sorted(set(skipped)):
        print(f"  - {name}")
    print("upstream comparisons by method:")
    for method, count in sorted(upstream.calls.items()):
        print(f"  {method}: {count}")
    print(f"distinct upstream comparisons (method, args): {len(upstream.keys)}")
    uncompared = [m for m in UPSTREAM_METHODS if m not in upstream.calls]
    print(f"public methods never compared: {uncompared or 'none'}")


if __name__ == "__main__":
    main()
