"""SearchIndex: BM25 over unique chunks, filtered to one snapshot at query time."""
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .text import Chunk, tokenize

# --------------------------------------------------------------------------------------
# 4. SearchIndex: BM25 over unique chunks, filtered by snapshot
# --------------------------------------------------------------------------------------

class SearchIndex:
    K1, B = 1.4, 0.75

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.lengths: List[int] = []
        self.members: List[int] = []          # bitmask of snapshots containing each chunk
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # term -> (chunk, tf)
        self._by_key: Dict[Tuple[str, str, str], int] = {}
        self._snap_bit: Dict[str, int] = {}

    def _bit(self, snapshot_id: str) -> int:
        if snapshot_id not in self._snap_bit:
            self._snap_bit[snapshot_id] = 1 << len(self._snap_bit)
        return self._snap_bit[snapshot_id]

    def add(self, snapshot_id: str, chunk: Chunk) -> None:
        """Index a chunk once; later snapshots that repeat it only set their membership bit."""
        bit = self._bit(snapshot_id)
        key = (chunk.url, chunk.heading, chunk.text)
        i = self._by_key.get(key)
        if i is not None:
            self.members[i] |= bit
            return
        i = len(self.chunks)
        self._by_key[key] = i
        self.chunks.append(chunk)
        self.members.append(bit)
        terms = tokenize(chunk.heading) + tokenize(chunk.text)
        self.lengths.append(len(terms))
        for term, tf in Counter(terms).items():
            self.postings[term].append((i, tf))

    def search(self, query: str, snapshot_id: str, k: int = 5) -> List[Tuple[float, Chunk]]:
        return [(s, self.chunks[i]) for i, s in self.search_ids(query, snapshot_id, k)]

    def search_ids(self, query: str, snapshot_id: str, k: int = 5) -> List[Tuple[int, float]]:
        """(chunk index, BM25 score), best first, over chunks live in `snapshot_id`."""
        bit = self._snap_bit.get(snapshot_id)
        if bit is None or not self.chunks:
            return []
        n = len(self.chunks)
        avg = sum(self.lengths) / n
        scores: Dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                if self.members[i] & bit:
                    norm = tf + self.K1 * (1 - self.B + self.B * self.lengths[i] / avg)
                    scores[i] += idf * tf * (self.K1 + 1) / norm
        # Sort by score, then chunk index, so equal scores always come back in the same order.
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def idf(self, term: str) -> float:
        """Inverse document frequency; 0 for terms the site never uses."""
        df = len(self.postings.get(term, ()))
        if not df:
            return 0.0
        n = len(self.chunks)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))
