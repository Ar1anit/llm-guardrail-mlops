"""Near-duplicate detection with MinHash LSH over word shingles.

Exact deduplication misses prompts that are one template with a changed name, word, or prefix.
When such variants land in different splits, the golden set measures memorisation instead of
generalisation. Two texts are near-duplicates when the Jaccard similarity of their word
``SHINGLE_SIZE``-grams reaches ``NEAR_DUPLICATE_THRESHOLD``. MinHash LSH proposes candidate
pairs and every candidate is confirmed with the exact Jaccard similarity, so the result does
not depend on hash collisions beyond the candidate search. Hashing is seeded and
deterministic.
"""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

SHINGLE_SIZE = 3
# A one-word edit in a ten-word prompt already lowers word 3-gram similarity to about 0.5.
NEAR_DUPLICATE_THRESHOLD = 0.5
# 42 bands of 3 rows find a pair at similarity 0.5 with probability above 0.99.
BANDS = 42
BAND_ROWS = 3
_MERSENNE_PRIME = (1 << 61) - 1
_SHINGLE_MODULUS = (1 << 31) - 1
_HASH_SEED = 42
_WORD = re.compile(r"\w+")

Shingles = frozenset[tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class NearDuplicatePair:
    """Indices ``first < second`` of two texts and their exact shingle Jaccard similarity."""

    first: int
    second: int
    similarity: float


def shingles(text: str, size: int = SHINGLE_SIZE) -> Shingles:
    """Case-folded word ``size``-grams; a text shorter than ``size`` words is one shingle."""
    words = _WORD.findall(text.casefold())
    if len(words) < size:
        return frozenset([tuple(words)])
    return frozenset(tuple(words[start : start + size]) for start in range(len(words) - size + 1))


def jaccard(first: Shingles, second: Shingles) -> float:
    union = len(first | second)
    return len(first & second) / union if union else 1.0


def _shingle_hash(shingle: tuple[str, ...]) -> int:
    digest = hashlib.blake2b(" ".join(shingle).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % _SHINGLE_MODULUS


def _signatures(documents: Sequence[Shingles]) -> npt.NDArray[np.uint64]:
    rng = np.random.default_rng(_HASH_SEED)
    permutations = BANDS * BAND_ROWS
    # Universal hashing (a * x + b) mod p with x < 2**31 and a, b < 2**30, so a * x + b stays
    # below 2**62 and uint64 arithmetic is exact.
    a = rng.integers(1, 1 << 30, size=permutations, dtype=np.uint64)
    b = rng.integers(0, 1 << 30, size=permutations, dtype=np.uint64)
    signatures = np.empty((len(documents), permutations), dtype=np.uint64)
    for row, document in enumerate(documents):
        values = np.array([_shingle_hash(s) for s in document], dtype=np.uint64)
        hashed = (np.outer(values, a) + b) % np.uint64(_MERSENNE_PRIME)
        signatures[row] = hashed.min(axis=0)
    return signatures


def near_duplicate_pairs(
    texts: Sequence[str],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[NearDuplicatePair]:
    """Every pair of texts whose shingle Jaccard similarity is at least ``threshold``."""
    documents = [shingles(text) for text in texts]
    if len(documents) < 2:
        return []
    signatures = _signatures(documents)
    candidates: set[tuple[int, int]] = set()
    for band in range(BANDS):
        buckets: dict[bytes, list[int]] = {}
        columns = signatures[:, band * BAND_ROWS : (band + 1) * BAND_ROWS]
        for index, key in enumerate(columns):
            buckets.setdefault(key.tobytes(), []).append(index)
        for members in buckets.values():
            candidates.update(
                (members[i], members[j])
                for i in range(len(members))
                for j in range(i + 1, len(members))
            )
    pairs = []
    for first, second in sorted(candidates):
        similarity = jaccard(documents[first], documents[second])
        if similarity >= threshold:
            pairs.append(NearDuplicatePair(first, second, similarity))
    return pairs


def near_duplicate_clusters(
    texts: Sequence[str],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[int]:
    """A cluster id per text: the transitive closure of the near-duplicate relation.

    Ids are the smallest index in each cluster, so they are stable for a given input order.
    """
    parent = list(range(len(texts)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for pair in near_duplicate_pairs(texts, threshold):
        first, second = root(pair.first), root(pair.second)
        if first != second:
            parent[max(first, second)] = min(first, second)
    return [root(index) for index in range(len(texts))]
