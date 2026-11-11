# NOTES

A knowledge system with hybrid (keyword + semantic) search over a dated, sourced history of the
website. Rules decide which value is current; an LLM (Azure OpenAI) answers descriptive and
fine-print questions from the retrieved passages, and the rules validate its answers.

**Running:** `pip install -r submission/requirements.txt`, then
`python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl`.

**API keys (paid API: Azure OpenAI).** Set these environment variables to enable the LLM:

| Variable | Value |
|---|---|
| `AZURE_OPENAI_API_KEY` | the Azure resource key |
| `AZURE_OPENAI_MODEL` | the deployment name |
| `AZURE_OPENAI_ENDPOINT` | the Responses API URL (default: `https://llm-model-test-hars.services.ai.azure.com/openai/v1/responses`) |

With them set, the LLM is on automatically (`KB_LLM=off` disables it). Without them the system
runs fully deterministic, with no API calls; it scores lower on descriptive questions (see
*Results*) but every other part works the same.

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
versions, jumbo certificates, IRA versions ("non-jumbo" / "not business" exclude them).

**6b. Validate the winner.** A passage about a different loan or term product than the question
names is not an answer: a "72-month certificate" or "96-month auto loan" question is never
answered with auto or boat rows of that term; if no candidate is about the asked product, the
system refuses. (Deposit accounts are not checked this way; their pages name each other too
often.)

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

## The LLM layer (on when a key is set)

`submission/kb/llm.py`. Rate and tier questions whose best candidate is a table stay with the rules
(fast and exact; that is where stale values live). The LLM answers:

- questions whose best rule candidate is a prose passage (descriptive questions, facts inside long
  pages, wording the page does not use: "who do I phone to buy insurance" -> "ext. 210");
- rate questions about an attribute written in fine print (maximum, floor, penalty, compounding,
  closing costs, credit score, fees);
- questions the rules refused, claim checks ("do any tiers pay 0.35%?") and yes/no questions about
  all/any/same.

It sees up to 12 passages from the retriever, spread over several sections, each labelled with its
page and page type (official rate sheet, product page, promotion, article with its date). The rules
then validate its answer: every number must appear in those passages or the question; the answer
must be about the product family asked; stale tables are never shown to it. A "not found" reply
becomes the standard refusal only if the rules had refused too; any error, timeout (3 s) or failed
check keeps the deterministic answer. The LLM never decides which value is current.

Switches: `KB_LLM=off`, `KB_LLM_ROUTE=narrow` (only when the rules are unsure), `KB_LLM_TIMEOUT`.
An Anthropic provider is also implemented (`ANTHROPIC_API_KEY`, needs `pip install anthropic`) but
untested; Azure is the provider we measured.

## Tried and not adopted

- **Cross-encoder reranker** (`kb/rerank.py`, `KB_RERANK=on`): a local fastembed cross-encoder
  added to the rule score when choosing between prose passages. It fixed one development question
  but left a fresh holdout unchanged (0.727 -> 0.727; with more weight 0.682), adds ~0.2 s per
  answer, and the best model (126 MB) exceeds GitHub's file limit. Off.
- **LLM only as a fallback when the rules are unsure** (`KB_LLM_ROUTE=narrow`): the rules are
  often confidently wrong on descriptive questions, so it rarely triggered where it would help.
  Routing all prose questions to it did better (below).
- **LLM rewriting every answer** (`KB_LLM_MODE=polish`): no accuracy change, only latency. Off.

## Results and runtime

Accuracy (`qa_score`). Holdout sets are our own questions, each written before the change it was
used to judge; #4 was written before the LLM routing and opened only after it was built.

| Set | Questions | Rules only (no key) | With the LLM (Azure) |
|---|---|---|---|
| Public practice set | 40 | 0.848 | 0.864 |
| Own validation set (tuned on) | 30 | 1.000 | 0.967 |
| Holdout #1 | 25 | 0.840 | 0.960 |
| Holdout #2 | 22 | 0.818 | 0.909 |
| Holdout #3 | 22 | 0.818 | 0.864 |
| Holdout #4 | 20 | 0.500 | 0.750 |
| **All holdouts** | 89 | **0.753** | **0.876** |

Public set, all four pillars (conflicts against our own 14-entry gold list): 0.837 without the
LLM, 0.845 with it. With the LLM, p95 answer latency is ~2.3 s (speed ~0.98) instead of 0.03 s;
consistency stays 1.0 and no stale value appears in any answer.

Ingest of all five snapshots takes ~4-6 minutes, almost all of it embedding ~11k unique chunks on
CPU. Embedding stops after a 15-minute budget (remaining chunks stay keyword-only), so ingest stays
well under the 30-minute limit on slower machines. Without fastembed or the model files the system
falls back to keyword search automatically.

## Known limitations

- Without the LLM, facts in the fine print under a table (a HELOC's maximum APR, floor rate,
  credit-score requirement) lose to the table row itself.
- With or without it, a passage the retriever does not find cannot be used ("how many employees
  were promoted in 2025" is on /careers but not retrieved); terms in years beyond seven
  ("9-year certificate") are not converted to months, so the term check cannot refuse them.
- Answers that need two pages combined (eligibility rules on `/about/join`, the three states on
  `/about`) are answered from one page only.
- Two practice questions expect an older value than the newest page states: "employees who used
  Volunteer Time Off last year" expects 294 (2023, from a 2024 release) while a 2025 release gives
  256 for 2024; "do any savings tiers pay 0.35%?" expects 0.20% (Share Savings) although the HSA
  and Money Market do pay 0.35% at higher balances. The system answers from the newest page.

## Development tools (`dev/`)

`check.sh` (public + validation regression run; forces the LLM off), `ab_llm.sh` (rules vs LLM
routing on all non-sealed sets), `retrieval_eval.py` (recall@k on 65 labelled questions), `conflict_report.py` + `conflicts_gold.json`, `trace.py` (one question through the
pipeline), `fresh_clone_check.sh` (clone, clean venv, offline run: the graders' setup).
