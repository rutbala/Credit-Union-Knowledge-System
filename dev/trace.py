#!/usr/bin/env python3
"""Trace one question through the pipeline: parsed query, retrieved chunks, top passages.
    python dev/trace.py "question" [as_of]          (KB_RETRIEVAL=bm25 for a fast run)"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
import run_eval  # noqa: E402

ks = run_eval.load_submission(ROOT / "submission")
for s in run_eval.load_snapshots(ROOT / "snapshots"):
    ks.ingest(s)
import types
mod = types.SimpleNamespace(**type(ks).__init__.__globals__)   # harness does not register the module
for question in sys.argv[1].split("||"):
    as_of = sys.argv[2] if len(sys.argv) > 2 else None
    sid = ks.store.snapshot_for(as_of)
    q = mod.parse_query(question)
    print("\n=== %s  [%s]" % (question, sid))
    print("kind=%s focus=%s names=%s numbers=%s terms=%s list=%s variants=%s" % (
        q.kind, q.focus, q.names, q.numbers, q.term_numbers, q.wants_list, set(q.variants)))
    hits = ks._retrieve(" ".join([q.text] + q.expansion), sid, k=mod.TOP_CHUNKS)
    for s, c in hits[:6]:
        print("  hit %.3f %s | %s" % (s, c.url.replace("https://www.dupaco.com", ""), c.heading[-60:]))
    weights = {c.url: (ks.store.version(sid, c.url).weight(True) if ks.store.version(sid, c.url) else 0) for _, c in hits}
    units = ks.composer.score(q, hits, weights)
    for u in units[:5]:
        print("  unit %.2f cov=%.2f val=%s qual=%s contra=%s refuse=%s | %s" % (
            u.score, u.coverage, u.has_value, u.qualified, u.contradicts, ks.composer.refuse(u), u.text[:90].replace("\n", " ")))
    print("  ANSWER:", ks.answer(question, as_of).text[:200].replace("\n", " | "))
