# NOTES

A rule-based knowledge system with hybrid (keyword + semantic) search. Every answer is chosen by
deterministic rules over a dated, sourced history of the website; an LLM is optional and off by
default.

**Running:** `pip install -r submission/requirements.txt`, then
`python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl`.
**API keys:** none needed. **Paid APIs:** none by default (see *Optional LLM* below).

## How it works

**1. Keep history, not just the latest copy.** Every version of every page is stored with its
snapshot, crawl date and content hash. A page missing from one crawl is carried forward from the
previous one (each snapshot is "the whole site as we had it"), but discounted and never used as
evidence of a contradiction.

**2. "As of" means a snapshot.** A date maps to the newest snapshot taken on or before it; no date
means the newest snapshot (28 Sep 2026). Search, facts and answers all use only the pages live in
that snapshot.

**3. Clean the pages.** Pages are split into heading sections; each table becomes self-describing
rows ("Share Savings, $10,000 – $24,999.99: Dividend Rate 0.15%; APY 0.15%") labelled with the
product name written above it. Blocks repeated on 10+ pages (related-article teasers, notices) are
kept on one page only, because indexed everywhere they make unrelated pages look relevant.

**4. Find candidates with hybrid search.** BM25 over section chunks, plus a local embedding model
(bge-small-en-v1.5, ONNX via fastembed, weights shipped in `submission/models`, no download), fused
by weighted reciprocal-rank fusion with keyword ranking leading (semantic weight 0.5). Semantic
search fixes reworded questions ("ABA number" -> routing number); keeping BM25 in the lead keeps
exact product questions precise.

**5. Decide which pages to believe.** Each URL gets an authority tier:

| Tier | Pages | Weight |
|---|---|---|
| Official | `/rates`, `/business-rates`, product pages, `/contact`, home | highest |
| Promotion | `switch-and-earn`, `*-lp` landing pages, grant pages | lower: copy can lag the rate sheet |
| Tool | calculators, PDFs | lower: example rates baked in |
| Article | blog, newsroom | low for current facts (rates, hours, links); mild for history and awards; older articles discounted further |
| Junk | expired-offer, thank-you, privacy-modal copies | not indexed |

On top of the tiers, **stale tables are detected from the data**: for every table that repeats
rate-sheet rows, the share of rows that disagree with the rate sheet *in the same snapshot* is
measured. Tables that disagree on most rows (the auto table on `/switch-and-earn` at 5.34% vs 5.39%,
the mortgage table on `/homebuying-guidance` at 6.037%) are down-weighted and never shown to the
LLM. This is per table and per snapshot, so an up-to-date table on the same page is unaffected and
a page stops being stale once it is updated (`/personal/checking-accounts` in September).

**6. Pick the passage that answers.** The question is parsed into the kind of value it needs
(rate, fee, hours, address, phone, routing number, link, date, count), its key terms (filler and
relative-time words removed, light stemming, a few synonyms: CD -> certificate, car -> auto), tier
qualifiers ("60-month", "2025 model year", "$10,000") and named products ("Volunteer Time Off").
Candidate rows and paragraphs are scored on search rank, coverage of the question, presence of the
needed value, tier match (2025 falls inside "2024-27"; "48 Months" contradicts "60-month"), page
authority, and an exact-name bonus. Products the member did not ask about are penalised: business
versions, jumbo certificates, IRA versions.

**7. Resolve the current value of every fact.** Each table row (and a few prose statements:
branch count, total assets, routing number, Forbes recognitions) becomes a dated fact keyed by
product + tier + term + metric. For a snapshot, the rate sheet wins, then the most authoritative
tier, then the newest page; when equally official pages disagree (`/rates` updated,
`/business-rates` not yet) the value the rest of the site agrees with wins. Model-year tiers that
rolled forward (2023-26 -> 2024-27) map to their successor. For prose facts the most recently
*published* statement wins. Before an answer is returned, any row whose value differs from the
resolved fact is swapped for the authoritative row, and `conflict_note` says why.

**8. Be honest.** If the best passage covers too little of the question, lacks the needed value,
contradicts the asked term, or misses a proper name in the question ("Mullan Avenue"), the answer
is "I don't know. That isn't listed on the website, so please contact us for details." with no
numbers in it.

**9. Sources.** The page the answer came from, plus up to three official product pages in the same
snapshot that state the same key values (a rate-sheet answer also cites the product's own page).

**Conflicts.** `conflicts()` lists every fact the site has stated with more than one value, across
pages (still contradicting each other in the newest snapshot) or across snapshots (changed over
time), with the winner and the reason. One entry per row (headline metric only), tiers of one
product merged, entries with identical competing values (IRA specials mirroring certificate
specials) listed once, stale-page contradictions and prose facts first. Output order is
deterministic.

## Optional LLM (off by default)

`submission/kb/llm.py` can hand questions the rules are unsure about (refusals, claim checks such as
"do any tiers pay 0.35%?", weak prose matches) to an LLM, together with the top passages labelled by
page type and date. Guards: every number in its answer must appear in those passages or the
question; "not found" replies become the standard refusal; any error or timeout keeps the
deterministic answer. It never decides which value is current; stale tables are not shown to it.

Enable with `KB_LLM=on` and either `ANTHROPIC_API_KEY` (plus `pip install anthropic`; model
`ANTHROPIC_MODEL`, default `claude-haiku-5-5` for latency) or `AZURE_OPENAI_API_KEY` +
`AZURE_OPENAI_MODEL` (+ `AZURE_OPENAI_ENDPOINT`). Both are paid APIs. It is off by default because,
measured on our evaluation sets, it did not raise accuracy (the remaining misses are retrieval
problems or answers the practice set judges differently) and it added ~2 s of latency per routed
question.

## Results and runtime

| Set | Score (accuracy) | Notes |
|---|---|---|
| Public practice set (40) | 0.848 | speed 1.0 (p95 0.03 s), consistency 1.0, no stale values |
| Own validation set (30) | 1.000 | used for tuning, so optimistic |
| Own holdout set (25) | 0.84 | written before the last round of changes and not tuned on |
| Conflicts (own gold, 14) | pillar ~0.77 | recall 14/14, precision 0.58 |

Ingest of all five snapshots takes ~4-6 minutes, almost all of it embedding ~11k unique chunks on
CPU. Embedding stops after a 15-minute budget (remaining chunks stay keyword-only), so ingest stays
well under the 30-minute limit on slower machines. Without fastembed or the model files the system
falls back to keyword search automatically.

## Known limitations

- Facts written in the fine print under a table (a HELOC's maximum APR, floor rate, credit-score
  requirement) lose to the table row itself.
- Answers that need two pages combined (eligibility rules on `/about/join`, the three states on
  `/about`) are answered from one page only.
- Two practice questions expect an older value than the newest page states: "employees who used
  Volunteer Time Off last year" expects 294 (2023, from a 2024 release) while a 2025 release gives
  256 for 2024; "do any savings tiers pay 0.35%?" expects 0.20% (Share Savings) although the HSA
  and Money Market do pay 0.35% at higher balances. The system answers from the newest page.

## Development tools (`dev/`)

`check.sh` (public + validation regression run), `retrieval_eval.py` (recall@k on 65 labelled
questions), `conflict_report.py` + `conflicts_gold.json`, `trace.py` (one question through the
pipeline), `fresh_clone_check.sh` (clone, clean venv, offline run: the graders' setup).
