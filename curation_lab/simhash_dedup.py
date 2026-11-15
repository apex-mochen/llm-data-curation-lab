"""SimHash near-duplicate dedup with pigeonhole banding.

If two 64-bit fingerprints differ in at most ``max_hamming`` bits and the
fingerprint is split into ``max_hamming + 1`` blocks, at least one block is
identical. That gives an exact candidate filter while only comparing Hamming
distance against documents that share a block.
"""

from typing import List, Set, Tuple

from .simhash import SimHash


def _block_specs(blocks: int) -> List[Tuple[int, int]]:
    base, extra = divmod(64, blocks)
    specs = []
    shift = 64
    for index in range(blocks):
        size = base + (1 if index < extra else 0)
        shift -= size
        specs.append((shift, (1 << size) - 1))
    return specs


class SimHashLSHDedup:
    def __init__(self, ngram: int = 5, mode: str = "word", max_hamming: int = 3) -> None:
        if max_hamming < 0:
            raise ValueError("max_hamming must be non-negative")
        if max_hamming >= 64:
            raise ValueError("max_hamming must be less than 64")
        self.hasher = SimHash(ngram=ngram, mode=mode)
        self.max_hamming = max_hamming
        self.blocks = max_hamming + 1
        self._specs = _block_specs(self.blocks)
        self._index: List[dict] = [{} for _ in range(self.blocks)]
        self._fingerprints: List[int] = []
        self.kept = 0

    @property
    def bucket_count(self) -> int:
        return sum(len(index) for index in self._index)

    def _keys(self, fingerprint: int) -> List[Tuple[int, int]]:
        return [(block, (fingerprint >> shift) & mask) for block, (shift, mask) in enumerate(self._specs)]

    def is_duplicate(self, text: str) -> bool:
        fingerprint = int(self.hasher.fingerprint(text))
        candidates: Set[int] = set()
        for block, key in self._keys(fingerprint):
            candidates.update(self._index[block].get(key, ()))
        for index in candidates:
            if (fingerprint ^ self._fingerprints[index]).bit_count() <= self.max_hamming:
                return True

        index = len(self._fingerprints)
        self._fingerprints.append(fingerprint)
        for block, key in self._keys(fingerprint):
            self._index[block].setdefault(key, []).append(index)
        self.kept += 1
        return False

    def check_and_add(self, text: str) -> bool:
        """Pipeline-compatible alias for :meth:`is_duplicate`."""
        return self.is_duplicate(text)
