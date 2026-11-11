"""Optional semantic reranker: a cross-encoder reads the question and each candidate passage
together and scores how well the passage answers it. Its score is added to the rule score of the
top candidates, so it can lift the right sentence out of fine print; the rules still validate
the winner (value present, tier/term match, stale tables, refusal). Off unless KB_RERANK=on.

  KB_RERANK         off | on (default off)
  KB_RERANK_MODEL   fastembed cross-encoder name (default Xenova/ms-marco-MiniLM-L-6-v2, 86 MB)
  KB_RERANK_WEIGHT  how much the reranker's opinion counts against the rule score (default 2.0)
"""
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MAX_CHARS = 512


class Reranker:
    def __init__(self, model, weight: float):
        self.model, self.weight = model, weight

    @classmethod
    def load(cls) -> Optional["Reranker"]:
        if os.environ.get("KB_RERANK", "off").lower() != "on":
            return None
        name = os.environ.get("KB_RERANK_MODEL", DEFAULT_MODEL)
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except Exception as e:
            print("[kb] reranker unavailable (%s)" % type(e).__name__, file=sys.stderr)
            return None
        local = MODEL_DIR / name.split("/")[-1]
        attempts = [lambda: TextCrossEncoder(name, specific_model_path=str(local))] if local.is_dir() else []
        attempts.append(lambda: TextCrossEncoder(name, cache_dir=os.environ.get("KB_RERANK_CACHE") or None))
        for attempt in attempts:
            try:
                return cls(attempt(), float(os.environ.get("KB_RERANK_WEIGHT", "2.0")))
            except Exception as e:
                err = e
        print("[kb] reranker model unavailable (%s)" % type(err).__name__, file=sys.stderr)
        return None

    def scores(self, query: str, passages: Sequence[str]) -> List[float]:
        """Relevance in (0, 1) for each passage (sigmoid of the cross-encoder logit)."""
        if not passages:
            return []
        raw = list(self.model.rerank(query, [p[:MAX_CHARS] for p in passages]))
        return [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, r)))) for r in raw]
