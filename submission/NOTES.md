# Notes

## How it works (Phase 1)

- **History, not just the latest copy.** `PageStore` keeps every version of every page with its
  snapshot id, snapshot date and crawl time. Identical page content is stored once.
- **What "as of" means.** A date maps to the newest snapshot taken on or before it. No date means
  the newest snapshot. A date before the first snapshot uses the first one.
- **Retrieval.** Each page is split into heading-delimited sections (tables stay with their
  heading). A BM25 index covers every unique section, and each section records which snapshots
  contained it, so a query only sees the website as it stood on the requested date.
- **Honesty.** If the best match is weak, the system says it doesn't know, with no numbers in the
  reply.

## Deciding what to believe (Phase 2)

Every URL gets an authority tier:

| Tier | Pages | Why |
|---|---|---|
| Product | `/rates`, `/business-rates`, `/personal/*`, `/business/*`, `/contact`, home, about | Maintained by the credit union; the source of truth for current values |
| Promo | `switch-and-earn`, `*-lp` landing pages, grant pages | Marketing copy; numbers can lag the rate sheet |
| Tool | `/calculators/*`, `hubfs/` PDFs and spreadsheets | Example rates baked in and rarely updated (e.g. 5.34% auto rate) |
| Article | `/learn/blog/*`, `/newsroom/*`, `/business/resources/*` | Dated: describe offers, counts and hours as they were when published |
| Listing | tag, author and index pages | Teasers of other pages |
| Junk | `offer-expired`, privacy-policy modal copies, thank-you pages | Not indexed |

Search results are reranked by tier weight. The weights depend on the question:
- **Current-fact questions** (rates, fees, hours, addresses, links, "now"): articles and
  calculators are pushed down hard.
- **Other questions** (awards, history, events): articles get only a mild discount, because a
  press release is often the only source.

Articles also lose a little weight with age (publish date read from the page), so the newest
release on a topic beats its older twins. The "Handpicked for You" related-article cards are
removed before indexing, and words are lightly stemmed so rewordings still match.

## Composing the answer (Phase 3)

**Tables become self-describing rows.** Rate pages name each product in a plain-text line just
above its table ("Credit Cards" / "Platinum Visa"), so every table is its own chunk labelled with
those lines, and each row is rendered with its column names:
`Share Savings, $10,000 – $24,999.99: Dividend Rate 0.15%; Annual Percentage Yield (APY) 0.15%; ...`
Footnote markers (`5.39% APR1`) and payment-example columns are dropped.

**The question is parsed** into:
- the kind of answer it needs (rate, fee, hours, address, phone, routing number, link, date or
  count), each with a pattern the answering text must contain;
- its key terms (generic words like "rate" or "Dupaco" are left out), plus a few synonyms
  (CD → certificate, mortgage → home loan);
- tier qualifiers ("60-month", "2025 model year", "$10,000").

**Passages compete.** The passages of the top 25 chunks (table rows, paragraphs, or runs of short
lines such as an hours block) are scored on:
- search rank;
- how much of the question's key terms they cover;
- whether they contain the needed value;
- tier matching (2025 falls inside "2024-27"; a "48 Months" row contradicts "60-month");
- page authority, with a bonus for rows from the official rate sheet (`/rates`, `/business-rates`).

**Writing the answer.**
- A specific question gets the matching row(s).
- A broad question gets the whole table.
- A list question ("what credit cards do you offer") also gets other strong tables, such as
  personal and business cards.
- Prose answers start with their page heading.
- Link questions return the best-matching link. Application links are only used when the
  question is about joining, applying or opening something.

**Refusing.** The system says "I don't know" (with no numbers) when:
- the best passage covers too little of the question;
- it lacks the needed value (e.g. there is no rate for construction loans);
- or a proper name in the question (e.g. "Mullan Avenue") never appears in it.

Answers are cached per question and snapshot, so asking again returns the same text.

## What is true now, and where the site contradicts itself (Phase 4)

**Facts.** Each table row becomes one fact per percentage. The fact key is the product label plus
the row's tier, with words stemmed and sorted, so "Special" + "10 Month Fixed Jumbo Certificate
Special" and "Special Certificates" + "10 Month Fixed Jumbo..." are the same fact. Product names
that differ between pages are unified ("Auto Loan Rates" on a promo, "Auto & Motorcycle Loans" on
the rate sheet). A few prose statements are facts too:
- branch count
- employee count
- total assets
- routing number
- how many times Forbes has recognized Dupaco

Every page version that states a fact is remembered.

**Deciding the current value** for a snapshot:
- **Table facts:** the official rate sheet (`/rates`, `/business-rates`) wins, then the most
  authoritative page tier, then the newest snapshot.
- **Model-year tiers that rolled forward:** if the rate sheet no longer lists a tier (2023-26) but
  lists a newer one for the same product and term (2024-27), the newer row is the current value.
  That is why the calculator's 5.34% loses to 5.39%.
- **Prose facts:** the most recently *published* statement wins. Undated articles and press-release
  footers count as oldest. Releases after the Hy-Vee closure say 22 branches; older ones say 23
  or 24.

**Using it in answers.**
- "How many branches" and "Forbes" questions are answered directly from the resolved fact.
- Table answers pass through a stale-value guard: if a more authoritative page states the same row
  with a different value, that page's row is used instead, and `conflict_note` says so.

**`conflicts()`** lists every fact the site has stated with more than one value, across pages
(still contradicting each other in the latest snapshot) or across snapshots (changed over time).
Each entry gives the competing values, a source for each, the winner, and the reason. To keep the
list useful:
- one metric per row (APY for deposits, APR for loans);
- tiers of one product and term merged into one entry (e.g. every loan-amount tier of the
  30-year purchase mortgage);
- live contradictions and prose facts ranked first;
- capped at 35 entries.

## Hardening (Phase 5)

- **Pages missing from a crawl carry forward.** Each snapshot is "every page we had", so a page a
  later crawl missed keeps its last copy. That copy is discounted, and it is never used as evidence
  of a contradiction.
- **Ended promotions are ignored** ("offer ended …", "no longer available").
- **Spelled-out terms are normalised**: "one-year certificate" becomes 12-month, and "five-year car
  loan" becomes 60-month. Mortgages keep their 15/20/30-year names.
- **Business products are discounted** unless the question asks about business products, or it is a
  broad list question ("what credit cards do you offer").
- **Refusing when the term doesn't match:** if the best passage only states other terms (6- and
  12-month when 60-month was asked), the system refuses.
- **No fixed search-score cut-off**, so small sites work too. Relevance comes from key-term
  coverage instead.

Checked against:
- the public set (0.83);
- 45 reworded and "as of" questions of our own (all correct, consistent within every reworded group);
- 16 questions on topics never tuned for (15 correct);
- the harness selftest (1.0).

The slowest answer takes about 0.01 s, and ingest of all five snapshots about 3 s.

## Running

From the kit root:

    python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl

Standard library only, so the clean Python 3.11 install the grader uses needs nothing from
requirements.txt. The code avoids features newer than Python 3.9.

## API keys

None. Standard library only.
