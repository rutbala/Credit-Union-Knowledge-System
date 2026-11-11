#!/usr/bin/env bash
# Regression check: public + validation evals (and conflicts estimate), one line each,
# plus every question whose score changed since the last saved run.
#   dev/check.sh [label] [python]       e.g. dev/check.sh vtest .venv/bin/python
set -uo pipefail
cd "$(dirname "$0")/.."
LABEL="${1:-latest}"; PY="${2:-python3}"
export PYTHONWARNINGS=ignore KB_EMBED_CACHE="${KB_EMBED_CACHE:-.cache/embeddings.npz}"
export KB_LLM="${KB_LLM:-off}"   # deterministic unless asked; .env may turn the LLM on
for set in public:eval/public_eval.jsonl validation:dev/validation_eval.jsonl; do
  name=${set%%:*}; file=${set#*:}; out=results/check/$name/$LABEL
  $PY harness/run_eval.py --submission submission --eval "$file" --out "$out" > "$out.log" 2>&1 || { echo "$name: RUN FAILED"; tail -5 "$out.log"; continue; }
  $PY - "$name" "$out" <<'PY'
import json, sys, glob, os
name, out = sys.argv[1], sys.argv[2]
rep = json.load(open(out + "/report.json"))
print("%-10s qa %.4f  final %.4f  traps %d  errors %d  p95 %.2fs  paraphrase %s" % (
    name, rep["qa_score"], rep["final_score"], rep["stale_trap_hits"], rep["errors"],
    rep["speed"]["p95_s"], rep["consistency"]["paraphrase_consistency"]))
prev = os.path.join(os.path.dirname(out), "last")
cur = {json.loads(l)["id"]: json.loads(l) for l in open(out + "/per_question.jsonl")}
if os.path.exists(prev + "/per_question.jsonl"):
    old = {json.loads(l)["id"]: json.loads(l) for l in open(prev + "/per_question.jsonl")}
    for i, r in cur.items():
        if i in old and abs(old[i]["score"] - r["score"]) > 1e-6:
            print("   %s %-14s %.2f -> %.2f  %s" % ("+" if r["score"] > old[i]["score"] else "-", i,
                  old[i]["score"], r["score"], r["answer"]["text"][:90].replace("\n", " | ")))
os.system("rm -rf '%s' && cp -R '%s' '%s'" % (prev, out, prev))
PY
done
