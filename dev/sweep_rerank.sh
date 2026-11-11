#!/usr/bin/env bash
# A/B sweep of the reranker over models and weights on public, validation and holdout #1.
cd "$(dirname "$0")/.."
export KB_LLM=off PYTHONWARNINGS=ignore KB_EMBED_CACHE=.cache/embeddings.npz
export KB_RERANK_CACHE="${KB_RERANK_CACHE:-.cache/rerank}"
run() {  # label, env...
  local label=$1; shift
  local line="$label"
  for set in public:eval/public_eval.jsonl validation:dev/validation_eval.jsonl holdout1:dev/holdout_eval.jsonl; do
    name=${set%%:*}; out=results/sweep/$label/$name
    env "$@" .venv/bin/python harness/run_eval.py --submission submission --eval "${set#*:}" --out "$out" >/dev/null 2>&1
    line="$line  $name $(python3 -c "import json;r=json.load(open('$out/report.json'));print('%.3f(traps %d,p95 %.2fs)'%(r['qa_score'],r['stale_trap_hits'],r['speed']['p95_s']))")"
  done
  echo "$line"
}
[ -z "${SKIP_BASELINE:-}" ] && run baseline KB_RERANK=off
for m in ${MODELS:-Xenova/ms-marco-MiniLM-L-6-v2 Xenova/ms-marco-MiniLM-L-12-v2 jinaai/jina-reranker-v1-tiny-en}; do
  for w in ${WEIGHTS:-1 3}; do run "$(basename $m)-w$w" KB_RERANK=on KB_RERANK_MODEL=$m KB_RERANK_WEIGHT=$w; done
done
