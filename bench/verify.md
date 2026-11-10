# Verification report

Session date: 2026-11-10. Host: Intel Xeon E5-2697 v4 @ 2.30GHz, 72 cores,
shared; `uptime` load average was 26 to 125 during measurement. Python
3.13.14, `konlpy==0.6.0`, Mojo `==1.2.0.dev2026092605`.

Everything here was measured in this session. Nothing is carried over from
the previous README.

## What was run

| step | command | result |
|---|---|---|
| build + test | `pixi run build && pixi run test` | green, 59 passed in 4.91 s (build emits deprecation warnings only) |
| benchmark, original | `pixi run bench` | all five original rows reproduced inside 20% |
| benchmark, repeated | `pixi run bench` twice more | row values moved by up to about 20% between runs |
| independent re-measurement | `pixi run python bench/verify_measure.py` | 9 rounds per side, cache cleared per round |
| normalize cache probe | `pixi run python bench/verify_normalize_probe.py` | upstream normalizer cost varies by input shape |
| distinct-chunk normalize | `pixi run python bench/verify_normalize_distinct.py` | upstream JVM pathology reproduced |
| usage-example row | `pixi run python bench/verify_usage_row.py` | single call and cold process cost |
| test coverage audit | `pixi run python bench/verify_test_coverage.py` | 54 of 59 cases compare against upstream |
| parity reproduction | `pixi run python bench/verify_parity_bug.py` | 4 of 5 cases diverge |
| extended benchmark | `pixi run bench` (7 rows, final) | table below |
| final gate | `pixi run build && pixi run test` | green |

## README table: reproduced or corrected

Original README figures versus what I measured. "Reproduced" means the
original figure is inside 20% of a value I measured; the new value is the one
written into the README.

| row | README before | this session range | verdict |
|---|---|---|---|
| pos, 400 short sentences | 13.20 ms / 284.63 ms / 21.57x | mojo 13.13-16.26, up 272-305, ratio 17.3-22.6 | reproduced |
| pos, 100-sentence paragraph | 10.07 ms / 132.48 ms / 13.16x | mojo 9.74-10.61, up 125-146, ratio 12.3-15.9 | reproduced |
| pos(norm=True, stem=True), 400 | 15.65 ms / 517.66 ms / 33.07x | mojo 15.07-16.87, up 502-568, ratio 29.8-40.9 | reproduced |
| normalize, 3,100 noisy chars | 0.30 ms / 23.96 ms / 80.99x | mojo 0.28-0.63, up 18-26, ratio 44-91 | reproduced, caveat below |
| phrases, 400 calls | 47.46 ms / 291.09 ms / 6.13x | mojo 45.9-47.9, up 277-315, ratio 4.0-6.4 | reproduced |

README table after correction (final `pixi run bench`, 2026-11-10):

| workload | mojo-konlpy | KoNLPy | upstream / Mojo |
|---|---:|---:|---:|
| pos, 400 short sentences | 13.25 ms | 293.90 ms | 22.18x |
| pos, 100-sentence paragraph | 10.45 ms | 145.80 ms | 13.95x |
| pos(norm=True, stem=True), 400 calls | 16.00 ms | 567.89 ms | 35.49x |
| normalize, 3,100 noisy characters | 0.30 ms | 24.40 ms | 82.33x |
| normalize(norm path via pos), 400 calls | 14.81 ms | 518.99 ms | 35.05x |
| normalize, 2,600 distinct Hangul characters | 0.05 ms | 14,308 ms | 286,160x |
| phrases, 400 calls | 47.11 ms | 280.88 ms | 5.96x |

No original row was removed and none was reworded upward. The
`phrases` row is the slowest ratio and stays at the bottom as it was. The
`pos(norm=True, stem=True)` column moved up from 15.65 to 16.00 ms — it was
slower this run, not faster.

## Caveats added for rows that passed the 20% rule

1. **`normalize`, 3,100 noisy characters.** The input is five distinct Korean
   chunks repeated 500 times. Both the port and KoNLPy memoise per chunk, so
   most of the row is cache replay on both sides. The port side has its own
   `lru_cache`; upstream's JVM helper does too — the same input cost 25.4 ms
   warm and 70.5-81.9 ms without warm-up. I did not replace the row's input
   (that would silently retune a published benchmark), but the README now
   states the cache dependence and the measured spread of the ratio.
2. **`normalize`, 2,600 distinct Hangul characters.** New row, added because
   the five-chunk row above is cache-dominated. Upstream stalls: the same JVM,
   same input, back-to-back calls took 14,308 and 14,325 ms, with cache
   warm-up calls at 17 ms. The first call into a fresh process was 455x, and
   the best-of-five bench (which starts fresh) reported 3.9E4 to 7.2E5
   depending on JVM restarts. The README reports the steady-state figure and
   says plainly it is a JVM pathology. I make no claim that it is a
   reproducible, portable speedup.
3. **`normalize(norm path via pos)`, 400 calls.** New row, because the
   `norm=True` flag inside `pos` was otherwise only covered by the
   `norm+stem` row and no row isolated it.

## What I could not reproduce

Nothing in the correctness suite. Two claims in the README were not checked
this session and are outside the benchmark table:

- The GPU section's instruction/byte table and 4559 MiB free-memory figure.
  Those need `valgrind --tool=callgrind` and `nvidia-smi` reruns; this session
  did not re-verify them, so they remain unverified rather than confirmed.
- The parallelism section's launch-latency table. The host was at load
  average 26 to 125 throughout, so those numbers would be dominated by
  neighbours; I did not re-measure them and did not touch them.

## Test suite audit

`git log -p -- tests/` shows only two commits touching tests: the initial
add (196 lines) and `1ecc4a6`, which changed 19 insertions and 1 deletion.
The single deletion is the import line gaining `_analyze_korean_layout`:

```diff
-from mojo_konlpy.tag import Okt, _normalize_korean_chunk
+from mojo_konlpy.tag import Okt, _analyze_korean_layout, _normalize_korean_chunk
```

No assertion was removed, no tolerance was loosened, and there are no
pytest `approx`, `xfail`, or `skip` marks to hide failures. Nothing needed
restoring. The suite is unchanged in this session.

Coverage, measured by instrumenting both analyzers
(`bench/verify_test_coverage.py`):

- 23 test functions, 59 cases after parametrisation.
- 54 cases call upstream and compare complete outputs; 5 do not (FFI
  argument validation x3, shared-library symbol export, constructor
  arguments). That is by design, not a gap.
- 56 distinct (method, arguments) comparisons against upstream; every public
  method — `pos`, `morphs`, `nouns`, `phrases`, `normalize`, `tagset` — is
  compared at least once.

## Parity failure found during verification

The benchmark asserted parity on all rows, but a workload with 200 distinct
strings tripped it, and the minimal reproduction is a real bug:

```
pos("하지 겟")                  mojo: [("하지겟", "Verb")]      upstream: [("하지", "Verb"), ("겟", "PreEomi")]
pos("하지 겟", norm=True)       mojo: [("하지겟", "Verb")]      upstream: [("하지", "Verb"), ("겟", "PreEomi")]
pos("하지 겟", stem=True)       mojo: [("하다", "Verb")]       upstream: [("하다", "Verb"), ("겟", "PreEomi")]
pos("먹지 겟")                  mojo: [("먹지겟", "Verb")]      upstream: [("먹지", "Verb"), ("겟", "PreEomi")]
pos("하지겟")                   agree on both sides
```

The port's DP joins the chunk and drops a `PreEomi` token that upstream
emits. `[("하지겟", "Verb")]` vs `[("하지", "Verb"), ("겟", "PreEomi")]` differ
only in token types, but the `stem=True` case loses a token entirely, so
that is a correctness defect, not a tag-choice difference. It is not covered
by the existing suite and I did not fix it: rules say to change code only for
a bug shown by a failing test, and the DP's transition table lives in
`src/konlpy.mojo`, which needs a rework of state transitions rather than a
one-line fix. The README now documents it and points at
`bench/verify_parity_bug.py`.

## What I changed

| file | change |
|---|---|
| `README.md` | benchmark table re-measured; cache and JVM caveats added; two new rows; correctness section states 54/59 cases compare against upstream; the `하지 겟` failure disclosed |
| `bench/bench.py` | added the two new rows (`normalize` via `pos(norm=True)`, distinct-chunk `normalize`). Original rows keep identical inputs. No timing method changed. |
| `bench/verify_measure.py` | new: independent per-row re-measurement with cache clears and ranges |
| `bench/verify_normalize_probe.py` | new: shows the normalize row is cache-dominated |
| `bench/verify_normalize_distinct.py` | new: reproduces the 14 s upstream stall |
| `bench/verify_usage_row.py` | new: single-call and cold-process cost of the README's usage example |
| `bench/verify_test_coverage.py` | new: counts differential assertions |
| `bench/verify_parity_bug.py` | new: minimal reproduction of the `하지 겟` failure |

Deliberately not changed: `tests/` (nothing was removed or loosened, nothing
needed restoring), `python/mojo_konlpy/` and `src/` (the parity bug is
documented, not fixed), and the GPU/parallelism sections.

## Final gate

```
pixi run build   -> exit 0 (deprecation warnings only)
pixi run test    -> exit 0, 59 passed in 4.74s
```

Both green, re-run after every edit in this session.
