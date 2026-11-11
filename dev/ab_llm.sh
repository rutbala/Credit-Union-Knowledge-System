#!/usr/bin/env bash
# A/B the LLM routing on every non-sealed set: deterministic vs narrow vs prose routing.
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore KB_EMBED_CACHE=.cache/embeddings.npz
SETS="${SETS:-public:eval/public_eval.jsonl validation:dev/validation_eval.jsonl h1:dev/holdout_eval.jsonl h2:dev/holdout2_eval.jsonl h3:dev/holdout3_eval.jsonl}"
run() {
  local label=$1; shift; local line=$(printf "%-14s" "$label")
  for set in $SETS; do
    n=${set%%:*}; out=results/ab_llm/$label/$n
    env "$@" .venv/bin/python harness/run_eval.py --submission submission --eval "${set#*:}" --out "$out" >"$out.log" 2>&1
    line="$line $(python3 -c "import json;r=json.load(open('$out/report.json'));print('%s %.3f(p95 %.1fs)'%('$n',r['qa_score'],r['speed']['p95_s']))")"
  done
  echo "$line"
}
mkdir -p results/ab_llm/{off,narrow,prose}
run off KB_LLM=off
run narrow KB_LLM=on KB_LLM_ROUTE=narrow
run prose KB_LLM=on KB_LLM_ROUTE=prose
