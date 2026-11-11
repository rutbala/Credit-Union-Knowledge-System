#!/usr/bin/env python3
"""Score a submission's conflicts() against dev/conflicts_gold.json and show what matched.

    python dev/conflict_report.py [--submission submission] [--min-confidence medium]

Uses the harness's own score_conflicts(), then explains each gold entry (which submitted
conflict claimed it, and whether the resolution was right) and lists unmatched submissions,
which count against precision.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
import run_eval  # noqa: E402
from scoring import _norm, contains, score_conflicts  # noqa: E402

LEVELS = {"low": 0, "medium": 1, "high": 2}


def match(gold, submitted):
    """Replays score_conflicts' greedy matching so we can see who claimed which gold entry."""
    used, out = set(), {}
    for g in gold:
        for i, c in enumerate(submitted):
            if i in used:
                continue
            vals = " | ".join(c.values or [])
            vhits = sum(contains(vals, v) for v in g["values"])
            khit = any(_norm(k) in _norm(c.topic or "") for k in g.get("topic_keywords", []))
            if vhits >= 2 or (vhits >= 1 and khit):
                used.add(i)
                out[g["id"]] = (i, bool(c.resolution and contains(c.resolution, g["correct_value"])))
                break
    return out, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, default=ROOT / "submission")
    ap.add_argument("--gold", type=Path, default=ROOT / "dev" / "conflicts_gold.json")
    ap.add_argument("--min-confidence", choices=LEVELS, default="low")
    args = ap.parse_args()

    gold = [g for g in json.loads(args.gold.read_text())
            if LEVELS[g.get("confidence", "high")] >= LEVELS[args.min_confidence]]
    ks = run_eval.load_submission(args.submission)
    for s in run_eval.load_snapshots(ROOT / "snapshots"):
        ks.ingest(s)
    submitted = ks.conflicts()

    report = score_conflicts(gold, submitted)
    pillar = 0.7 * report["f1"] + 0.3 * (report["resolution_accuracy"] or 0)
    matched, used = match(gold, submitted)

    print("\nGOLD (%d entries, confidence >= %s)" % (len(gold), args.min_confidence))
    for g in gold:
        if g["id"] in matched:
            i, ok = matched[g["id"]]
            c = submitted[i]
            print("  %-4s %-22s <- #%d %s  [resolution %s]" % (
                "OK" if ok else "WIN?", g["id"], i, c.topic[:60], c.resolution))
        else:
            print("  MISS %-22s values %s" % (g["id"], g["values"]))
    print("\nUNMATCHED SUBMISSIONS (hurt precision): %d of %d" % (len(submitted) - len(used), len(submitted)))
    for i, c in enumerate(submitted):
        if i not in used:
            print("  #%-3d %s  %s" % (i, c.topic[:70], c.values[:4]))
    print("\n" + json.dumps(report, indent=1))
    print("conflicts pillar = 0.7*F1 + 0.3*resolution = %.3f" % pillar)


if __name__ == "__main__":
    main()
