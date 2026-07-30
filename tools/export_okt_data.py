"""Build the standalone lexicon from KoNLPy's Open Korean Text 2.1.0 jar."""

from __future__ import annotations

import gzip
from pathlib import Path
import zipfile

import jpype
import numpy as np
from konlpy.tag import Okt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "python" / "mojo_konlpy" / "data"
POS_NAMES = (
    "Noun", "Verb", "Adjective", "Adverb", "Determiner", "Exclamation",
    "Josa", "Eomi", "PreEomi", "Conjunction", "Modifier", "VerbPrefix", "Suffix",
)


def fnv1a(text: str) -> int:
    value = 1469598103934665603
    for char in text:
        value ^= ord(char)
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def scala_items(mapping):
    iterator = mapping.iterator()
    while iterator.hasNext():
        pair = iterator.next()
        yield pair._1(), pair._2()


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    Okt()
    provider_class = jpype.JClass(
        "org.openkoreantext.processor.util.KoreanDictionaryProvider$"
    )
    provider = getattr(provider_class, "MODULE$")
    dictionaries = provider.koreanDictionary()

    words: dict[int, tuple[str, int]] = {}
    pos_by_name = {str(pos): pos for pos in dictionaries.keySet()}
    for pos_id, name in enumerate(POS_NAMES):
        iterator = dictionaries.get(pos_by_name[name]).iterator()
        while iterator.hasNext():
            word = str(iterator.next())
            key = fnv1a(word)
            if key in words and words[key][0] != word:
                raise RuntimeError(f"FNV-1a collision: {word!r} and {words[key][0]!r}")
            old_word, mask = words.get(key, (word, 0))
            words[key] = (old_word, mask | (1 << pos_id))

    freq_by_hash = {
        fnv1a(str(entry.getKey())): float(entry.getValue())
        for entry in provider.koreanEntityFreq().entrySet()
    }
    capacity = 1
    while capacity < len(words) * 2:
        capacity *= 2
    keys = np.zeros(capacity, dtype=np.uint64)
    masks = np.zeros(capacity, dtype=np.int64)
    freqs = np.zeros(capacity, dtype=np.float64)
    for key, (_, mask) in words.items():
        slot = key & (capacity - 1)
        while masks[slot]:
            slot = (slot + 1) & (capacity - 1)
        keys[slot] = key
        masks[slot] = mask
        freqs[slot] = freq_by_hash.get(key, 0.0)
    np.savez_compressed(DATA / "lexicon.npz", keys=keys, masks=masks, freqs=freqs)

    with gzip.open(DATA / "stems.tsv.gz", "wt", encoding="utf-8", newline="\n") as stream:
        for pos, mapping in scala_items(provider.predicateStems()):
            pos_id = POS_NAMES.index(str(pos))
            for surface, stem in scala_items(mapping):
                stream.write(f"{pos_id}\t{surface}\t{stem}\n")

    jar = Path(jpype.JClass("org.openkoreantext.processor.OpenKoreanTextProcessor")
               .class_.getProtectionDomain().getCodeSource().getLocation().toURI().getPath())
    with zipfile.ZipFile(jar) as archive:
        typo_path = "org/openkoreantext/processor/util/typos/typos.txt"
        (DATA / "typos.txt").write_bytes(archive.read(typo_path))

    print(f"exported {len(words):,} surfaces into {capacity:,} hash slots")


if __name__ == "__main__":
    main()
