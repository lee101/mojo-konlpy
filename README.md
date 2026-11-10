# mojo-konlpy

`mojo-konlpy` is a standalone Mojo port of the compute-heavy morphology search
behind KoNLPy's `Okt` tagger. It keeps KoNLPy's Python class and method
signatures while replacing the JVM/Scala Viterbi search with a native Mojo
shared library.

The port uses the exact Open Korean Text 2.1.0 lexicon bundled with KoNLPy
0.6.0. The generated runtime data is checked into this repository, so using
`mojo_konlpy` does not start Java and does not require KoNLPy at runtime.
KoNLPy remains in the Pixi environment solely as the parity and benchmark
reference.

## Coverage

The covered API is `konlpy.tag.Okt`, exposed as `mojo_konlpy.tag.Okt`:

- `Okt(jvmpath=None, max_heap_size=1024)`
- `pos(phrase, norm=False, stem=False, join=False)`
- `morphs(phrase, norm=False, stem=False)`
- `nouns(phrase)`
- `phrases(phrase)`
- `normalize(phrase)`
- `tagset`

The constructor accepts KoNLPy's JVM arguments for source compatibility but
does not use them. POS names, joining, stemming, unknown-noun collapse,
colloquial normalization, chunk classes, phrase extraction, and input
validation follow KoNLPy/Open Korean Text behavior.

This release does not implement `Hannanum`, `Kkma`, `Komoran`, or `Mecab`.
Those are separate engines with different models, dictionaries, and licenses.
It also does not expose Open Korean Text internals that KoNLPy itself does not
make public.

## Install and build

```bash
pixi install
pixi run build
```

The build creates `dist/libmojo-konlpy.so`.

## Usage

```python
from mojo_konlpy.tag import Okt

okt = Okt()

print(okt.morphs("단독입찰보다 복수입찰의 경우"))
print(okt.nouns("우리나라에는 무릎 치료를 잘하는 정형외과가 없는가!"))
print(okt.pos("이것도 되나욬ㅋㅋ", norm=True, stem=True))
```

Run the example in the managed environment:

```bash
pixi run python - <<'PY'
from mojo_konlpy.tag import Okt

okt = Okt()
print(okt.pos("한국어를 처리하는 예시입니다."))
PY
```

## Correctness

`pixi run test` runs the parity suite against the real `konlpy==0.6.0`
package. It checks KoNLPy's published `Okt` examples, every public method,
constructor compatibility arguments, normalizing and stemming flags,
non-Korean chunk classes, number-overlap edge cases, and sentences from Korean
legal prose. It also checks the repeated-chunk normalization cache and rejects
invalid FFI buffers. Assertions compare complete token/POS or phrase sequences;
they are not smoke tests.

```bash
pixi run build
pixi run test
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30GHz using
Python 3.13.14. Times are the best of five warm runs. Each benchmark first
asserts complete output equality with KoNLPy. Ratios above 1 mean this port is
faster.

| workload | mojo-konlpy | KoNLPy | upstream / Mojo |
|---|---:|---:|---:|
| pos, 400 short sentences | 13.20 ms | 284.63 ms | 21.57x |
| pos, 100-sentence paragraph | 10.07 ms | 132.48 ms | 13.16x |
| pos(norm=True, stem=True), 400 calls | 15.65 ms | 517.66 ms | 33.07x |
| normalize, 3,100 noisy characters | 0.30 ms | 23.96 ms | 80.99x |
| phrases, 400 calls | 47.46 ms | 291.09 ms | 6.13x |

These numbers describe this machine and workload, not a universal speedup.
Run `pixi run bench` for local results; the Pixi task takes a machine-wide
benchmark lock.

`pixi run python bench/analysis.py` records the measurements behind the
decision below (kernel cost by chunk length and the chunk sizes real callers
pass); it prints tables and asserts nothing about timing.

Native scratch initialization uses unaligned SIMD stores with a scalar tail;
unaligned access is required because the buffers are NumPy-owned. Morphology
results are cached as bounded, immutable, offset-independent layouts, avoiding
repeat native calls and temporary buffers while preserving caller-specific
token offsets. The FFI still receives NumPy-owned contiguous buffers directly.

## Parallelism

A parallel path was measured and then reverted: it cannot pay off at the sizes
real callers pass. Findings from `bench/analysis.py` and a scratch
`max.algorithm.parallelize` harness on this shared 72-core Xeon E5-2697 v4
host (load average 70-96 during measurement):

- **Chunk work grows with chunk length and is tiny per call.** One
  `mkl_analyze` call costs 31-34 us for a 2-3 character word, 44 us at 6,
  188 us at 32, 588 us at 112 and 3.6 ms at 400. A row also depends on the
  previous 8 positions, so a split is only independent in blocks of 8 or more
  rows, and the split still has to allocate and initialise the whole quadratic
  scratch table.
- **Callers never reach that size.** Across the benchmark workloads the Korean
  chunks are 2 to 7 characters (median 4); none reaches 128 characters. The
  100-sentence paragraph is 1,300 chunks of median length 3, served from the
  bounded result cache after the first pass.
- **`parallelize` launches cost milliseconds.** In a nested Mojo `main()` with
  an empty worker body, the best of 20 launches was 4.4 ms at 2 workers,
  4.6 ms at 4 and 4.7 ms at 8; a one-worker launch is about 0.1 us. Launched
  from a ctypes host (this library's only entry point) the same empty body cost
  10.8, 8.4 and 5.5 ms respectively. Either way a launch is at least 130x the
  direct cost of a 3-character word.
- **No crossover in the observable range.** A time-sliced loop with 3,000
  inner operations per work item needed 32 items to break even at 2-8 workers,
  about 10 ms of serial work per launch. The kernel only supplies that much
  work at several thousand characters.
- **The split measurement did not reproduce.** Probes that divide a read-only
  pass over the real lexicon across workers never produced a result pair I
  could trust: the serial and split exports disagreed, results shifted with
  process history, and multi-worker launches degraded to 10-50 ms in a ctypes
  host. So no honest parallel gain could be recorded, no threshold or worker
  count went into the library, and none of those probe numbers is quoted
  here.

The package stays CPU-only and single-threaded, allocates no device memory,
and does not depend on `max`. There is no serial/parallel equivalence test
because there is no parallel path to compare. A GPU path is not taken for the
same reason plus the arithmetic-intensity test: the working set is scattered
table reads well under 2 flops per byte. If a caller ever needs a
multi-thousand character Hangul run, parallelising that chunk's edge scan,
not the whole call, is the shape that could work; it needs its own benchmark
before it ships.

## How it works

Python splits text into Open Korean Text chunk classes and converts each Hangul
chunk to a contiguous `int64` code-point buffer. The lexicon is a generated,
open-addressed table containing 399,025 surface forms, POS bitmasks, and entity
frequencies. Its arrays are NumPy-owned and C-contiguous.

The wrapper passes only integer addresses and extents through `ctypes`.
`src/konlpy.mojo` reconstructs mutable pointers with
`AnyOrigin[mut=True]`, performs the top-five bounded Viterbi search and scoring,
and writes token offsets, POS IDs, and unknown flags into caller-owned output
buffers. No string or heap object crosses the C ABI. Python then assembles
Unicode strings and applies Open Korean Text's noun-collapse, stem, normalize,
and phrase rules.

The runtime lexicon and stem data are derived from Apache-2.0-licensed Open
Korean Text 2.1.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
