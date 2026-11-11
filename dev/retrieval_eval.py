#!/usr/bin/env python3
"""Retrieval-only evaluation: is the right page in the top k?

    python dev/retrieval_eval.py                    # every registered retriever
    python dev/retrieval_eval.py --only baseline -v # one retriever, list misses

Separates "search didn't find the page" from "composer picked the wrong row", so retrieval
changes (BM25 tweaks, embeddings, fusion) can be judged on their own.

A retriever is any object with:
    ingest(snapshot) -> None
    search(question, as_of=None, k=10) -> list[str]   # page URLs, best first, de-duplicated
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
import run_eval  # noqa: E402

KS = (1, 3, 5, 10)


def url_key(u):
    """Same path-level match the scorer uses for sources (no host prefix, query, fragment, slash)."""
    p = urlparse(u if "//" in u else "https://" + u)
    return p.netloc.lower().replace("www.", "", 1) + p.path.rstrip("/").lower()


# --------------------------------------------------------------------------- retrievers

class BaselineRetriever:
    """The current submission's BM25 + authority rerank (KnowledgeBase._retrieve)."""

    def __init__(self):
        self.kb = run_eval.load_submission(ROOT / "submission")

    def ingest(self, snapshot):
        self.kb.ingest(snapshot)

    def search(self, question, as_of=None, k=10):
        sid = self.kb.store.snapshot_for(as_of)
        hits = self.kb._retrieve(question, sid, k=k * 4)
        return list(dict.fromkeys(c.url for _, c in hits))[:k]


def _registry():
    reg = {"baseline": BaselineRetriever}
    try:   # the new hybrid index, once it exists
        sys.path.insert(0, str(ROOT / "submission"))
        from kb.retriever import eval_retrievers
        reg.update(eval_retrievers())
    except ImportError:
        pass
    return reg


# --------------------------------------------------------------------------- scoring

def evaluate(retriever, items, k_max=10):
    rows, lat = [], []
    for it in items:
        t = time.perf_counter()
        urls = retriever.search(it["question"], as_of=it.get("as_of"), k=k_max)
        lat.append(time.perf_counter() - t)
        rel = {url_key(u) for u in it["relevant"]}
        rank = next((i + 1 for i, u in enumerate(urls) if url_key(u) in rel), None)
        rows.append({"id": it["id"], "kind": it["kind"], "rank": rank, "top": urls[:3]})
    return rows, sorted(lat)


def summarize(rows):
    def block(rs):
        n = len(rs) or 1
        out = {"n": len(rs)}
        for k in KS:
            out["hit@%d" % k] = round(sum(1 for r in rs if r["rank"] and r["rank"] <= k) / n, 3)
        out["mrr"] = round(sum(1.0 / r["rank"] for r in rs if r["rank"]) / n, 3)
        return out
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    return {"all": block(rows), **{k: block(v) for k, v in sorted(by.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=ROOT / "dev" / "retrieval_eval.jsonl")
    ap.add_argument("--only", help="comma-separated retriever names")
    ap.add_argument("-v", "--verbose", action="store_true", help="list questions missed at k=5")
    args = ap.parse_args()

    items = [json.loads(l) for l in args.eval.read_text().splitlines() if l.strip()]
    snaps = run_eval.load_snapshots(ROOT / "snapshots")
    reg = _registry()
    names = args.only.split(",") if args.only else list(reg)

    table = {}
    for name in names:
        r = reg[name]()
        t = time.perf_counter()
        for s in snaps:
            r.ingest(s)
        ingest_s = time.perf_counter() - t
        rows, lat = evaluate(r, items)
        table[name] = summarize(rows)
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        print("\n== %s  (ingest %.1fs, search p95 %.0f ms)" % (name, ingest_s, p95 * 1000))
        for kind, m in table[name].items():
            print("  %-9s n=%-3d hit@1 %.2f  hit@3 %.2f  hit@5 %.2f  hit@10 %.2f  mrr %.2f" % (
                kind, m["n"], m["hit@1"], m["hit@3"], m["hit@5"], m["hit@10"], m["mrr"]))
        if args.verbose:
            for row in rows:
                if not row["rank"] or row["rank"] > 5:
                    top = [urlparse(u).path or "/" for u in row["top"]]
                    print("    MISS %-16s rank=%-4s top=%s" % (row["id"], row["rank"], top))
    out = ROOT / "results" / "retrieval_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
