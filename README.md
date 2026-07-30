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
| pos, 400 short sentences | 146.74 ms | 311.27 ms | 2.12x |
| pos, 100-sentence paragraph | 228.74 ms | 277.41 ms | 1.21x |
| pos(norm=True, stem=True), 400 calls | 154.44 ms | 589.29 ms | 3.82x |
| normalize, 3,100 noisy characters | 0.49 ms | 46.27 ms | 93.59x |
| phrases, 400 calls | 191.24 ms | 418.70 ms | 2.19x |

These numbers describe this machine and workload, not a universal speedup.
Run `pixi run bench` for local results; the Pixi task takes a machine-wide
benchmark lock.

There is no SIMD, threaded, or GPU normalization path. Its hot work is
variable-length Unicode matching, data-dependent substitution, and irregular
hash-table probing rather than a contiguous numeric loop. Splitting the small
chunks across threads costs more than the cached work, while converting them to
numeric device buffers would add allocation, copies, and transfer overhead to a
low-arithmetic-intensity kernel. The morphology search is likewise
branch-heavy, state-dependent, and irregular. A GPU path would lose, so the
package remains CPU-only and does not allocate device memory or depend on
`max`.

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
