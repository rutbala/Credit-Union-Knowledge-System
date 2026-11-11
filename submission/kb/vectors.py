"""Dense (semantic) retrieval over the same chunks as the BM25 index, plus rank fusion.

The embedding model (bge-small-en-v1.5, ONNX, quantized, 384-dim) runs locally through
fastembed; its weights ship in submission/models so nothing is downloaded at grading time.
If fastembed or the weights are unavailable, `VectorIndex.load()` returns None and the
system silently falls back to BM25 only.

Vectors are kept in one in-memory numpy matrix (~25k chunks x 384 floats = ~40 MB);
brute-force cosine search over it takes a few milliseconds, so no ANN index is needed.
"""
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MAX_CHARS = 1500        # the model reads ~512 tokens; longer chunks are cut
BATCH = 32
EMBED_BUDGET_S = float(os.environ.get("KB_EMBED_BUDGET_S", "900"))   # ingest is capped at 30 min
RRF_K = 60              # standard reciprocal-rank-fusion constant


class VectorIndex:
    def __init__(self, model):
        import numpy as np
        self.np = np
        self.model = model
        self.matrix = np.zeros((0, 384), dtype=np.float32)
        self.valid = np.zeros(0, dtype=bool)       # False: not embedded (time budget ran out)
        self.spent = 0.0                           # seconds spent embedding so far
        self._members = None          # numpy view of SearchIndex.members, rebuilt when it grows
        self._members_n = -1

    @classmethod
    def load(cls) -> Optional["VectorIndex"]:
        if os.environ.get("KB_RETRIEVAL", "hybrid") == "bm25":
            return None
        try:
            from fastembed import TextEmbedding
        except Exception as e:      # package missing: BM25 only
            print("[kb] fastembed unavailable (%s); using BM25 only" % type(e).__name__, file=sys.stderr)
            return None
        for local_only in (True, False):    # weights in the repo first; download only as a fallback
            try:
                model = TextEmbedding(MODEL_NAME, cache_dir=str(MODEL_DIR), local_files_only=local_only)
                return cls(model)
            except Exception as e:
                err = e
        print("[kb] embedding model unavailable (%s); using BM25 only" % type(err).__name__, file=sys.stderr)
        return None

    @staticmethod
    def passage_text(heading: str, text: str) -> str:
        return (heading + "\n" + text)[:MAX_CHARS]

    def add(self, texts: Sequence[str]) -> None:
        """Embed new chunks; row i of the matrix is chunk i of the BM25 index."""
        np = self.np
        if not texts:
            return
        cache = _DevCache.get()
        vecs = np.zeros((len(texts), 384), dtype=np.float32)
        todo = [i for i, t in enumerate(texts) if not cache.fill(t, vecs, i)]
        # Embed shortest first: a batch is padded to its longest text, so mixing a 20-token
        # row with a 500-token section wastes most of the work (about 2x slower unsorted).
        todo.sort(key=lambda i: len(texts[i]))
        valid = np.ones(len(texts), dtype=bool)
        valid[todo] = False
        start = time.perf_counter()
        if self.spent >= EMBED_BUDGET_S:
            todo = []                  # budget already used up by earlier snapshots
        for i, v in zip(todo, self.model.passage_embed([texts[i] for i in todo], batch_size=BATCH)):
            vecs[i], valid[i] = v, True
            cache.put(texts[i], vecs[i])
            if self.spent + time.perf_counter() - start > EMBED_BUDGET_S:
                # A slow grading machine must never push ingest past the 30-minute limit:
                # chunks not embedded yet are left to keyword search.
                print("[kb] embedding budget (%ds) reached; %d chunks keyword-only"
                      % (EMBED_BUDGET_S, int((~valid).sum())), file=sys.stderr)
                break
        self.spent += time.perf_counter() - start
        cache.save()
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        self.matrix = np.vstack([self.matrix, vecs])
        self.valid = np.concatenate([self.valid, valid])

    def search(self, query: str, members: List[int], bit: int, k: int = 40) -> List[Tuple[int, float]]:
        """(chunk index, cosine) for the k nearest chunks that were live in the snapshot `bit`."""
        np = self.np
        if len(members) != self._members_n:
            self._members = np.array(members, dtype=np.int64)
            self._members_n = len(members)
        q = np.array(next(iter(self.model.query_embed([query]))), dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-9
        n = min(len(self.matrix), len(self._members))
        sims = self.matrix[:n] @ q
        sims[((self._members[:n] & bit) == 0) | ~self.valid[:n]] = -1.0
        top = np.argpartition(-sims, min(k, n - 1))[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(int(i), float(sims[i])) for i in top if sims[i] > -1.0]


class _DevCache:
    """Development-only embedding cache (KB_EMBED_CACHE=path.npz), keyed by text hash.

    Re-embedding ~11k chunks takes minutes; while iterating on retrieval we only want to pay
    that once. Off unless the environment variable is set, so grading never reads or writes it.
    """
    _instance = None

    def __init__(self, path: Optional[str]):
        self.path, self.vecs, self.dirty = path, {}, False
        if path and Path(path).exists():
            import numpy as np
            data = np.load(path)
            self.vecs = dict(zip(data["keys"].tolist(), data["vecs"]))

    @classmethod
    def get(cls) -> "_DevCache":
        if cls._instance is None:
            cls._instance = cls(os.environ.get("KB_EMBED_CACHE"))
        return cls._instance

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1((MODEL_NAME + "\0" + text).encode("utf-8")).hexdigest()

    def fill(self, text, out, i) -> bool:
        v = self.vecs.get(self._key(text)) if self.path else None
        if v is None:
            return False
        out[i] = v
        return True

    def put(self, text, vec) -> None:
        if self.path:
            self.vecs[self._key(text)] = vec.copy()
            self.dirty = True

    def save(self) -> None:
        if self.path and self.dirty:
            import numpy as np
            keys = list(self.vecs)
            np.savez(self.path, keys=np.array(keys), vecs=np.stack([self.vecs[k] for k in keys]))
            self.dirty = False


def rrf(*rankings: Sequence[int], weights: Optional[Sequence[float]] = None, k: int = RRF_K) -> Dict[int, float]:
    """Reciprocal rank fusion: each list contributes weight/(k + rank) to every item it ranks."""
    weights = weights or [1.0] * len(rankings)
    scores: Dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + w / (k + rank + 1)
    return scores


def append_fusion(bm25: Sequence[Tuple[int, float]], dense: Sequence[Tuple[int, float]],
                  limit: int) -> List[Tuple[int, float]]:
    """BM25 order is kept as is; semantic-only hits fill the remaining slots below it.

    Keyword search is precise when the question uses the site's words, so its ranking is not
    disturbed; embeddings only add pages keyword search missed (reworded questions).
    """
    out = list(bm25[:limit])
    seen = {i for i, _ in out}
    floor = min((s for _, s in out), default=1.0)
    extra = [i for i, _ in dense if i not in seen]
    room = max(limit - len(out), limit // 4)      # always leave space for some semantic hits
    for rank, i in enumerate(extra[:room]):
        out.append((i, floor * (0.9 - 0.01 * rank)))   # just below the weakest keyword hit
    return out
