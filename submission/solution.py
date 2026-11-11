"""
Knowledge system that knows what's true *now*.

Phase 1: versioned page store + `as_of` snapshot resolution + BM25 retrieval over
section-level chunks of the snapshot that was live on the requested date.
Phase 2: page authority tiers + publish dates; retrieval is reranked so official
product/rate pages beat promos, calculators, and old blog posts or press releases.
Phase 3: tables become self-describing rows; answers are composed from the best row(s)
or passage for the question (answer kind, tier qualifiers, term coverage), and the system
refuses when nothing on the site covers the question.
Phase 4: every table row and a few prose statements become dated facts; the current value is
resolved per snapshot (rate sheet > maintained page > promo > tool > article; newest published
statement for counts), stale rows in answers are swapped for it, and conflicts() reports them.

Layout of this file (top to bottom):
  1. Text helpers   - tokenizing, cleaning markdown, splitting pages into chunks
  2. Authority      - classify a URL into a tier, read an article's publish date
  3. PageStore      - every version of every page, keyed by snapshot (history, not just latest)
  4. SearchIndex    - BM25 over unique chunks, filtered to one snapshot at query time
  5. Composer       - parse the question, score table rows / passages, write the answer
  6. FactStore      - dated facts, current-value resolution, contradiction reports
  7. KnowledgeBase  - the contract object: ingest / answer / conflicts

Note: no `from __future__ import annotations` here. The harness loads this file without
registering it in sys.modules, and string annotations make @dataclass crash in that case.
"""
import bisect
import dataclasses
import hashlib
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents if (d / "harness" / "contract.py").exists())))
from contract import Answer, Conflict, Snapshot, Source  # noqa: E402


# --------------------------------------------------------------------------------------
# 1. Text helpers
# --------------------------------------------------------------------------------------

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOPWORDS = frozenset(
    "a an the and or of to for in on at by is are was were be been it its this that these those "
    "what which who whom how when where why do does did can could would should will i you your we "
    "our us my me with from as about into any some there their they them get give tell please".split()
)

IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]*)[^)]*\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
# Phone-country pickers ("- 🇲🇾 Malaysia +60") are pure noise and full of numbers.
COUNTRY_LINE = re.compile(r"^\s*-\s+\S+\s+[A-Za-z].*\+\d+\s*$")
# Footnote markers glued to values: "5.39% APR1" -> "5.39% APR", "2.90%1" -> "2.90%".
FOOTNOTE = re.compile(r"(%|\bAPR|\bAPY|\))\d{1,2}\b")
TABLE_SEP = re.compile(r"^\|?[\s:|-]*-{3,}[\s:|-]*$")
# Table columns that hold the value (everything before the first one describes the tier).
VALUE_HEADER = re.compile(r"rate|apr|apy|yield|percentage|dividend|fee|min\.|minimum|price|cost", re.I)
# Columns that only add distracting numbers to an answer.
DROP_HEADER = re.compile(r"payment example|points", re.I)

MAX_CHUNK_CHARS = 1500
MAX_LABELS = 2          # plain-text label lines kept above a table ("Credit Cards", "Platinum Visa")
MAX_LABEL_CHARS = 60
MAX_TABLE_ROWS = 40


SUFFIXES = ("ibility", "ability", "ations", "ation", "ition", "izing", "ized", "izes", "ible", "able",
            "ings", "ing", "ies", "ied", "ers", "er", "ed", "ly")
ES_PLURAL = re.compile(r"(s|x|z|ch|sh)es$")


PLURAL_ACRONYMS = {"aprs": "apr", "apys": "apy", "cds": "cd", "atms": "atm", "iras": "ira", "hsas": "hsa"}


def stem(word: str) -> str:
    """Light suffix stripping so 'recognized' / 'recognition' or 'rates' / 'rate' meet."""
    if word in PLURAL_ACRONYMS:
        return PLURAL_ACRONYMS[word]
    if word[0].isdigit() or len(word) < 4:
        return word
    # Plurals first: "certificates" -> "certificate", "fees" -> "fee", "branches" -> "branch".
    if ES_PLURAL.search(word):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if len(word) <= 4:
        return word
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: -len(suf)]
    return word


def tokenize(text: str) -> List[str]:
    return [stem(t) for t in TOKEN.findall(text.lower()) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def clean_line(line: str) -> Optional[str]:
    """Drop images and noise, keep link text, strip markdown escapes and footnote markers."""
    if COUNTRY_LINE.match(line):
        return None
    line = LINK.sub(r"\1", IMAGE.sub("", line)).replace("\\", "").rstrip()
    if line.strip() in ("✕", "* * *"):
        return None
    return FOOTNOTE.sub(r"\1", line)


def line_links(line: str) -> List[Tuple[str, str]]:
    """(anchor, url) for links to the credit union's own sites."""
    out = []
    for m in LINK.finditer(IMAGE.sub("", line)):
        anchor, url = m.group(1).strip(), m.group(2)
        if "dupaco.com" in url and "/hubfs/" not in url and "/hs-fs/" not in url:
            out.append((anchor, url))
    return out


def is_link_only(line: str) -> bool:
    return bool(LINK.search(line)) and not LINK.sub("", IMAGE.sub("", line)).strip()


@dataclass(frozen=True)
class Chunk:
    url: str
    heading: str                    # heading path (+ table labels), e.g. "Rates > Credit Cards > Platinum Visa"
    text: str                       # prose, or the rendered table rows one per line
    rows: Tuple[str, ...] = ()      # table chunks only: one rendered line per row
    columns: str = ""               # table chunks only: the header cells ("Balance Tier | APY ...")
    links: Tuple[Tuple[str, str], ...] = ()   # (anchor, url) of links inside the chunk

    @property
    def is_table(self) -> bool:
        return bool(self.rows)


def split_cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render_rows(lines: List[str]) -> List[str]:
    """Turn markdown table lines into self-describing rows.

    "| Share Savings | $10,000 – $24,999.99 | 0.15% | ..." under its header becomes
    "Share Savings, $10,000 – $24,999.99: Dividend Rate 0.15%; Annual Percentage Yield (APY) 0.15%; ..."
    """
    header: List[str] = []
    body = lines
    if len(lines) >= 2 and TABLE_SEP.match(lines[1].strip()):
        header, body = split_cells(lines[0]), lines[2:]
    first_value = next((i for i, h in enumerate(header) if VALUE_HEADER.search(h)), None)
    if first_value == 0:
        first_value = 1   # the first column always names the row
    rows = []
    for line in body[:MAX_TABLE_ROWS]:
        cells = split_cells(line)
        if not any(cells):
            continue
        if not header or first_value is None:
            rows.append(" | ".join(c for c in cells if c))
            continue
        subject = ", ".join(c for c in cells[:first_value] if c and c != "—")
        values = []
        for h, c in zip(header[first_value:], cells[first_value:]):
            if c and c != "—" and not DROP_HEADER.search(h):
                values.append("%s %s" % (h, c))
        rows.append("%s: %s" % (subject, "; ".join(values)) if values else subject)
    return rows


def split_page(url: str, title: str, md: str) -> List[Chunk]:
    """Split a page into heading-delimited prose sections plus one chunk per table.

    A table chunk's heading carries the plain-text labels written just above it, because
    rate pages name products that way ("Credit Cards" / "Platinum Visa" / table).
    """
    chunks: List[Chunk] = []
    path: List[Tuple[int, str]] = []
    buf: List[str] = []
    buf_links: List[Tuple[str, str]] = []
    labels: List[str] = []
    table: List[str] = []
    skipping = False   # inside a "Handpicked for You" block of related-article cards

    def heading_for(extra: List[str]) -> str:
        return " > ".join([h for _, h in path] + extra) or title

    def flush():
        body = "\n".join(buf).strip()
        links = tuple(dict.fromkeys(buf_links))
        buf.clear()
        buf_links.clear()
        if len(body) < 20:
            return
        heading = heading_for([])
        # Oversized sections are cut on paragraph boundaries so one chunk stays focused.
        part: List[str] = []
        size = 0
        for para in body.split("\n\n"):
            if size + len(para) > MAX_CHUNK_CHARS and part:
                chunks.append(Chunk(url, heading, "\n\n".join(part), links=links))
                part, size = [], 0
            part.append(para)
            size += len(para) + 2
        if part:
            chunks.append(Chunk(url, heading, "\n\n".join(part), links=links))

    def flush_table():
        if not table:
            return
        rows = tuple(render_rows(table))
        columns = " | ".join(split_cells(table[0])) if len(table) > 1 and TABLE_SEP.match(table[1].strip()) else ""
        table.clear()
        if rows:
            chunks.append(Chunk(url, heading_for(labels), "\n".join(rows), rows=rows, columns=columns))
        if labels:
            labels.pop()   # that label named this table; the outer category stays

    for raw in md.splitlines():
        m = HEADING.match(raw)
        if m:
            level, text = len(m.group(1)), clean_line(m.group(2).strip()) or ""
            if skipping and level > 2:
                continue
            flush_table()
            flush()
            labels.clear()
            # Related-article cards are other pages' teasers (titles + dates); they only add noise.
            skipping = text.lower().startswith("handpicked for you")
            if skipping:
                continue
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, text))
            continue
        if skipping:
            continue
        line = clean_line(raw)
        if line is None:
            continue
        if line.lstrip().startswith("|"):
            table.append(line)
            continue
        flush_table()
        buf.append(line)
        buf_links.extend(line_links(raw))
        stripped = line.strip().strip("*").strip().lstrip("-•").strip()
        if stripped and len(stripped) <= MAX_LABEL_CHARS and not is_link_only(raw):
            labels.append(stripped)
            del labels[:-MAX_LABELS]
    flush_table()
    flush()
    return chunks


# --------------------------------------------------------------------------------------
# 2. Authority: which pages to believe
# --------------------------------------------------------------------------------------
# Lower tier = more authoritative. The official product and rate pages are what the credit
# union maintains; promos, calculators and dated articles are where stale numbers survive.

TIER_PRODUCT, TIER_PROMO, TIER_TOOL, TIER_ARTICLE, TIER_LISTING, TIER_JUNK = range(6)
# Two weight tables. "Current fact" questions (a rate, fee, hours, address, link) must come
# from maintained pages, so articles are pushed down hard. Other questions (awards, history,
# events) are often only answered by a press release, so articles are only mildly discounted.
TIER_WEIGHT_STRICT = {
    TIER_PRODUCT: 1.0,
    TIER_PROMO: 0.8,
    TIER_TOOL: 0.55,
    TIER_ARTICLE: 0.65,
    TIER_LISTING: 0.4,
}
TIER_WEIGHT_MILD = {
    TIER_PRODUCT: 1.0,
    TIER_PROMO: 0.9,
    TIER_TOOL: 0.7,
    TIER_ARTICLE: 0.85,
    TIER_LISTING: 0.5,
}
CURRENT_FACT = re.compile(
    r"\b(rates?|apr|apy|dividend|interest|fees?|cost|charge|minimum|hours?|open|address|located|"
    r"location|routing|phone|number|call|url|link|website|page|current(?:ly)?|now|today)\b", re.I)

PROMO_PATHS = ("switch-and-earn", "earn-more", "mortgage-refi", "money-makeover", "first-time-homebuyer-grant",
               "homebuying-guidance",
               "21-day-sprint-to-savings-challenge", "sprint-to-savings-challenge")
JUNK_PATHS = ("offer-expired", "subscription-preferences", "thankuse", "photo-release-form", "not-for-sale")
LISTING_PREFIXES = ("newsroom/tag", "newsroom/author", "learn/blog/tag", "learn/blog/author",
                    "business/resources/tag", "business/resources/author", "resources")
ARTICLE_PREFIXES = ("learn/blog/", "newsroom/", "business/resources/", "dupacodaily/")
TOOL_PREFIXES = ("calculators", "business-calculators", "hubfs/")

MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}
DATE_LINE = re.compile(r"^\s*(" + "|".join(MONTHS) + r")\s+(\d{1,2}),\s+(\d{4})\s*$", re.I)


def page_path(url: str) -> str:
    return re.sub(r"^https?://[^/]+", "", url).strip("/").lower()


def classify(url: str) -> int:
    path = page_path(url)
    first = path.split("/")[0].split("?")[0]
    if first in JUNK_PATHS or (first == "privacy-policy" and "?" in path):
        return TIER_JUNK
    if path.startswith(LISTING_PREFIXES) or path in ("newsroom", "learn/blog", "business/resources"):
        return TIER_LISTING
    if path.startswith(ARTICLE_PREFIXES) or re.match(r"20\d\d/\d\d/", path):
        return TIER_ARTICLE
    if path.startswith(TOOL_PREFIXES):
        return TIER_TOOL
    if first in PROMO_PATHS or first.endswith("-lp"):
        return TIER_PROMO
    return TIER_PRODUCT


def published_on(md: str) -> Optional[str]:
    """An article's publish date: a bare 'Month DD, YYYY' line in its first few lines."""
    for line in md.splitlines()[:12]:
        m = DATE_LINE.match(line)
        if m:
            return "%s-%02d-%02d" % (m.group(3), MONTHS[m.group(1).lower()], int(m.group(2)))
    return None


def authority(tier: int, published: Optional[str], taken_on: str, strict: bool = True) -> float:
    """Rerank weight for a page: its tier, and for articles how old they were at crawl time."""
    w = (TIER_WEIGHT_STRICT if strict else TIER_WEIGHT_MILD).get(tier, 0.0)
    if tier == TIER_ARTICLE and published:
        age_years = (int(taken_on[:4]) - int(published[:4])) + (int(taken_on[5:7]) - int(published[5:7])) / 12
        w *= max(0.75, 1.0 - 0.06 * max(age_years, 0.0))   # a 2026 release beats its 2021 twin
    return w


# --------------------------------------------------------------------------------------
# 3. PageStore: full history of every page
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PageVersion:
    url: str
    title: str
    snapshot_id: str
    taken_on: str      # snapshot date (ISO)
    fetched_at: str    # when this copy was actually crawled; can predate taken_on
    content_hash: str
    tier: int                     # authority tier from the URL (TIER_*)
    published: Optional[str]      # article publish date (ISO), None for undated pages
    carried: bool = False         # missing from this crawl; last seen copy carried forward

    def weight(self, strict: bool = True) -> float:
        w = authority(self.tier, self.published, self.taken_on, strict)
        return w * CARRIED_WEIGHT if self.carried else w


class PageStore:
    """Keeps every version of every page, plus which pages each snapshot contained."""

    def __init__(self):
        self.snapshot_ids: List[str] = []          # oldest -> newest
        self.snapshot_dates: List[str] = []        # parallel to snapshot_ids
        self.pages: Dict[str, Dict[str, PageVersion]] = {}   # snapshot_id -> url -> version
        self.content: Dict[str, str] = {}          # content_hash -> markdown (deduplicated)
        self.history: Dict[str, List[PageVersion]] = defaultdict(list)  # url -> versions, oldest first
        self.carried: Dict[str, List[PageVersion]] = {}   # snapshot_id -> versions carried into it

    def add(self, snapshot: Snapshot) -> List[PageVersion]:
        """Record a snapshot; returns the versions whose content was not seen before."""
        idx = bisect.bisect_right(self.snapshot_dates, snapshot.taken_on)
        self.snapshot_ids.insert(idx, snapshot.snapshot_id)
        self.snapshot_dates.insert(idx, snapshot.taken_on)

        fresh = []
        by_url = {}
        for p in snapshot.pages:
            h = hashlib.sha1(p.markdown.encode("utf-8")).hexdigest()
            v = PageVersion(p.url, p.title, snapshot.snapshot_id, snapshot.taken_on, p.fetched_at, h,
                            classify(p.url), published_on(p.markdown))
            by_url[p.url] = v
            self.history[p.url].append(v)
            if h not in self.content:
                self.content[h] = p.markdown
                fresh.append(v)
        # A snapshot is "every page we had": a page the crawl missed keeps its last known copy,
        # marked as carried so it never outranks a page actually seen in this crawl.
        prev = self.snapshot_ids[idx - 1] if idx > 0 else None
        self.carried[snapshot.snapshot_id] = []
        for url, old in (self.pages.get(prev, {}) if prev else {}).items():
            if url not in by_url:
                v = dataclasses.replace(old, snapshot_id=snapshot.snapshot_id, carried=True)
                by_url[url] = v
                self.carried[snapshot.snapshot_id].append(v)
        self.pages[snapshot.snapshot_id] = by_url
        return fresh

    def snapshot_for(self, as_of: Optional[str]) -> Optional[str]:
        """The snapshot that was live on `as_of` (newest taken on or before it).

        No date -> newest snapshot. A date before the first snapshot -> the first one,
        since it is the earliest view of the site we have.
        """
        if not self.snapshot_ids:
            return None
        if not as_of:
            return self.snapshot_ids[-1]
        i = bisect.bisect_right(self.snapshot_dates, as_of[:10]) - 1
        return self.snapshot_ids[max(i, 0)]

    def version(self, snapshot_id: str, url: str) -> Optional[PageVersion]:
        return self.pages.get(snapshot_id, {}).get(url)


# --------------------------------------------------------------------------------------
# 4. SearchIndex: BM25 over unique chunks, filtered by snapshot
# --------------------------------------------------------------------------------------

class SearchIndex:
    K1, B = 1.4, 0.75

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.lengths: List[int] = []
        self.members: List[int] = []          # bitmask of snapshots containing each chunk
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # term -> (chunk, tf)
        self._by_key: Dict[Tuple[str, str, str], int] = {}
        self._snap_bit: Dict[str, int] = {}

    def _bit(self, snapshot_id: str) -> int:
        if snapshot_id not in self._snap_bit:
            self._snap_bit[snapshot_id] = 1 << len(self._snap_bit)
        return self._snap_bit[snapshot_id]

    def add(self, snapshot_id: str, chunk: Chunk) -> None:
        """Index a chunk once; later snapshots that repeat it only set their membership bit."""
        bit = self._bit(snapshot_id)
        key = (chunk.url, chunk.heading, chunk.text)
        i = self._by_key.get(key)
        if i is not None:
            self.members[i] |= bit
            return
        i = len(self.chunks)
        self._by_key[key] = i
        self.chunks.append(chunk)
        self.members.append(bit)
        terms = tokenize(chunk.heading) + tokenize(chunk.text)
        self.lengths.append(len(terms))
        for term, tf in Counter(terms).items():
            self.postings[term].append((i, tf))

    def search(self, query: str, snapshot_id: str, k: int = 5) -> List[Tuple[float, Chunk]]:
        bit = self._snap_bit.get(snapshot_id)
        if bit is None or not self.chunks:
            return []
        n = len(self.chunks)
        avg = sum(self.lengths) / n
        scores: Dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                if self.members[i] & bit:
                    norm = tf + self.K1 * (1 - self.B + self.B * self.lengths[i] / avg)
                    scores[i] += idf * tf * (self.K1 + 1) / norm
        # Sort by score, then chunk index, so equal scores always come back in the same order.
        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(s, self.chunks[i]) for i, s in top]

    def idf(self, term: str) -> float:
        """Inverse document frequency; 0 for terms the site never uses."""
        df = len(self.postings.get(term, ()))
        if not df:
            return 0.0
        n = len(self.chunks)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))


# --------------------------------------------------------------------------------------
# 5. Answer composition: pick the table row or prose passage that answers the question
# --------------------------------------------------------------------------------------

MONTH_NAMES = "|".join(m.capitalize() for m in MONTHS)
# (kind, question pattern, pattern the answering text must contain). First match wins.
ANSWER_KINDS = [
    ("routing", r"\brouting\b|\baba\b", r"\b\d{9}\b"),
    ("url", r"\b(url|link|website|web ?page|online)\b", None),
    ("date", r"\bwhen\b|\bwhat date\b|\bstill open\b", r"\b(%s)\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b" % MONTH_NAMES),
    ("hours", r"\bhours?\b", r"\b\d{1,2}(:\d\d)?\s?[ap]\.?m\b|\bnoon\b"),
    # A rate is a percentage tied to a rate word ("APR 7.29%", "2.90% annual percentage rate"),
    # not any percentage ("borrow up to 80% of construction costs").
    ("rate", r"\b(rates?|aprs?|apys?|dividends?|interest|yield)\b",
     r"\d+(?:\.\d+)?\s?%\s*(apr|apy|annual percentage|dividend|fixed|variable|interest)"
     r"|\b(apr|apy|rate|yield|dividend)\)?\s*(\(as low as\)\s*)?(of|is|:)?\s*\d+(?:\.\d+)?\s?%"),
    ("fee", r"\b(fees?|cost|charge|price|how much)\b", r"\$\s?\d"),
    ("address", r"\b(address|located|location of|where is)\b", r"\b\d{2,6}\s+[A-Z][a-z]+"),
    ("phone", r"\b(phone|telephone|call)\b", r"\b\d{3}[-.)\s]+\d{3}[-.]\d{4}\b"),
    ("count", r"\bhow many\b", r"\b\d+\b"),
]
ANSWER_KINDS = [(k, re.compile(q, re.I), re.compile(v, re.I) if v else None) for k, q, v in ANSWER_KINDS]

# Questions about *doing* something, where an application link is the right answer.
ACTION = re.compile(r"\b(join|apply|open|sign ?up|enroll|log ?in|become a member)\b", re.I)
APP_HOST = re.compile(r"//(join|shine|homeloans|apply)\.", re.I)

LIST_INTENT = re.compile(r"\b(what are|which|list|all|options|offer|rates)\b", re.I)
# Words that say *what kind* of answer is wanted but not *which* product; never required.
GENERIC = frozenset(tokenize(
    "dupaco rate rates apr apy interest current currently today now offer offered provide know like want "
    "much many right still repeat see use need information info account accounts credit union "
    "between earn going have get any also there highest lowest best top maximum max"))
# Question-side synonyms (stemmed): a focus term counts as covered if any alternative appears.
SYNONYMS = {
    "cd": {"certificate"}, "cds": {"certificate"}, "certificate": {"cd"},
    "mortgage": {"home", "conventional"}, "card": {"visa"}, "join": {"member", "membership"},
    "locator": {"location", "locat"}, "wallet": {"pay"}, "university": {"college"}, "college": {"university"},
}

# Capitalised words that name products or the institution, not a specific place or person.
COMMON_CAPS = frozenset(tokenize(
    "Dupaco Visa Platinum Rewards Business Share Savings Checking Certificate Special Jumbo Fixed IRA Roth "
    "Traditional HSA Health Money Market Bright Launchpad Holiday Club Investor Choice High Yield Free First "
    "Rate Home Auto Personal Student Loan Shine Apple Google Samsung Pay ATM CD Can What How Is Do Does "
    "Iowa Illinois Wisconsin "
    + " ".join(MONTHS)))

NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")
TERM_NUM = re.compile(r"\b(\d{1,3})[\s-]*(month|mo|year|yr)s?\b", re.I)
RANGE = re.compile(r"(\$?\d[\d,]*(?:\.\d+)?)\s*(?:–|-|to)\s*(\$?\d[\d,]*(?:\.\d+)?)")
AT_LEAST = re.compile(r"(\$?\d[\d,]*(?:\.\d+)?)\s*\+")
AT_MOST = re.compile(r"(\$?\d[\d,]*(?:\.\d+)?)\s*(?:or less|and older|\(and older\))", re.I)

RATE_SHEETS = ("rates", "business-rates")
BUSINESS = re.compile(r"\bbusiness|\bcommercial\b", re.I)
BUSINESS_PENALTY = 0.75
MIN_RANGE_VALUE = 100    # question numbers below this are terms or counts, never tier amounts
MIN_COVERAGE = 0.5       # share of the question's focus (idf-weighted) the answer must cover
MIN_COVERAGE_QUALIFIED = 0.35   # ...lower when the row also matches the asked-for tier
TIE = 0.35               # rows scoring within this of the best are answered together
LIST_TIE = 1.0           # list questions also take other tables scoring within this
MAX_ROWS = 12
MAX_TABLES = 4
MAX_PROSE_CHARS = 600
SHORT_PARA = 100         # consecutive short paragraphs (hours, lists) form one passage
MAX_RUN = 10


def to_number(s: str) -> float:
    return float(s.replace("$", "").replace(",", ""))


@dataclass
class Query:
    text: str
    expansion: List[str]             # synonym stems added to the retrieval query
    focus: List[str]                 # distinctive tokens the answer has to be about
    names: List[str]                 # proper names (branch, street, partner); must appear in the answer
    kind: Optional[str]
    value_re: Optional[object]       # compiled pattern the answer text must contain
    numbers: List[float]
    term_numbers: List[Tuple[int, str]]   # (30, "year"), (60, "month")
    wants_list: bool
    business: bool                   # the question is about business products


NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten".split())}
SHORT_YEARS = re.compile(r"\b(\d{1,2})[\s-]*years?\b", re.I)


def normalize_terms(question: str) -> str:
    """'one-year certificate' -> '12-month certificate', 'five-year car loan' -> '60-month car loan'.

    Rate tables state certificate and consumer-loan terms in months; mortgages keep their
    15/20/30-year names, so only terms up to seven years are converted."""
    q = re.sub(r"\b(%s)\b(?=[\s-]*(?:years?|months?)\b)" % "|".join(NUMBER_WORDS),
               lambda m: NUMBER_WORDS[m.group(1).lower()], question, flags=re.I)
    if re.search(r"\bmortgage|home loan|refinanc", q, re.I):
        return q
    return SHORT_YEARS.sub(lambda m: "%d-month" % (int(m.group(1)) * 12) if int(m.group(1)) <= 7 else m.group(0), q)


def parse_query(question: str) -> Query:
    question = normalize_terms(question)
    toks = tokenize(question)
    focus = list(dict.fromkeys(t for t in toks if t not in GENERIC and not t[0].isdigit()))
    names = proper_names(question)
    expansion = [s for t in toks for s in sorted(SYNONYMS.get(t, ())) if s not in toks]
    kind, value_re = None, None
    for k, qre, vre in ANSWER_KINDS:
        if qre.search(question):
            kind, value_re = k, vre
            break
    # "7/1 ARM", "5/5" are product labels, not quantities.
    plain = re.sub(r"\b\d+/\d+\b", " ", question)
    numbers = [to_number(n) for n in NUM.findall(plain) if any(ch.isdigit() for ch in n)]
    term_numbers = [(int(n), u.lower()[:2]) for n, u in TERM_NUM.findall(question)]
    wants_list = bool(LIST_INTENT.search(question)) and not numbers
    business = bool(BUSINESS.search(question))
    return Query(question, expansion, focus, names, kind, value_re, numbers, term_numbers, wants_list, business)


def proper_names(question: str) -> List[str]:
    """Capitalised words after the first word that are not product vocabulary, stemmed.

    "Mullan Avenue branch" -> ["mullan", "avenue"]; if the best passage never mentions them,
    the site is not talking about that branch and the system should not answer from it.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", question)[1:]
    out = []
    for w in words:
        w = re.sub(r"['’]s$", "", w)
        if not w[0].isupper() or w.isupper():
            continue
        for t in tokenize(w):
            if t not in GENERIC and t not in COMMON_CAPS:
                out.append(t)
    return list(dict.fromkeys(out))


def ranges_in(text: str) -> List[Tuple[float, float]]:
    """Numeric tiers in a row: '$10,000 – $24,999.99', '2024-27', '$25,000+', '$832,750 or less'."""
    out = []
    for a, b in RANGE.findall(text):
        lo, hi = to_number(a), to_number(b)
        if hi < 100 <= lo and 1900 < lo < 2100:          # '2024-27' -> 2024..2027
            hi = lo // 100 * 100 + hi
        out.append((min(lo, hi), max(lo, hi)))
    out += [(to_number(a), float("inf")) for a in AT_LEAST.findall(text)]
    out += [(0.0, to_number(a)) for a in AT_MOST.findall(text)]
    return out


def qualifier_match(q: Query, text: str) -> Tuple[int, int]:
    """(qualifiers the text satisfies, qualifiers it contradicts) for tiered rows."""
    if not q.numbers and not q.term_numbers:
        return 0, 0
    present = {to_number(n) for n in NUM.findall(text) if any(ch.isdigit() for ch in n)}
    spans = ranges_in(text)
    # Range containment only makes sense for amounts and years; "7" is not "$832,750 or less".
    hits = sum(1 for n in q.numbers
               if n in present or (n >= MIN_RANGE_VALUE and any(lo <= n <= hi for lo, hi in spans)))
    row_terms = [(int(n), u.lower()[:2]) for n, u in TERM_NUM.findall(text)]
    conflicts = sum(1 for n, u in q.term_numbers
                    if any(ru == u for _, ru in row_terms) and (n, u) not in row_terms)
    return hits, conflicts


def passages(chunk: Chunk) -> List[str]:
    """Answer-sized units of a chunk: table rows, or paragraphs with short runs merged."""
    if chunk.is_table:
        return list(chunk.rows)
    out: List[str] = []
    run: List[str] = []

    def end_run():
        for i in range(0, len(run), MAX_RUN):
            out.append("\n".join(run[i:i + MAX_RUN]))
        run.clear()

    for para in (p.strip() for p in chunk.text.split("\n\n")):
        if not para:
            continue
        if len(para) <= SHORT_PARA:
            run.append(para)
            continue
        end_run()
        if len(para) <= MAX_PROSE_CHARS:
            out.append(para)
            continue
        piece = ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if piece and len(piece) + len(sent) > MAX_PROSE_CHARS // 2:
                out.append(piece)
                piece = ""
            piece = (piece + " " + sent).strip()
        if piece:
            out.append(piece)
    end_run()
    return out


@dataclass
class Scored:
    score: float
    coverage: float
    has_value: bool
    missing_rare: bool
    qualified: bool         # satisfies at least one tier qualifier from the question
    contradicts: bool       # states a different term than asked (6- and 12-month for "60-month") and no match
    chunk: Chunk
    text: str
    order: int


class Composer:
    """Scores candidate passages against a parsed question and writes the answer text."""

    def __init__(self, index: SearchIndex):
        self.index = index

    def covered(self, q: Query, terms: set) -> Tuple[float, bool]:
        total = covered = 0.0
        missing_rare = False
        for t in q.focus:
            w = self.index.idf(t)
            if not w:
                continue          # a word the site never uses tells us nothing
            total += w
            if t in terms or SYNONYMS.get(t, set()) & terms:
                covered += w
        missing_rare = any(n not in terms for n in q.names)
        return (covered / total if total else 1.0), missing_rare

    def score(self, q: Query, hits: List[Tuple[float, Chunk]], weights: Dict[str, float]) -> List[Scored]:
        if not hits:
            return []
        top = hits[0][0]
        out = []
        order = 0
        for rank_score, chunk in hits:
            ctx = set(tokenize(chunk.heading + " " + chunk.columns))
            for text in passages(chunk):
                if EXPIRED.search(text):
                    continue    # an ended promotion is history, not an answer
                # A table row is about the product its label names ("Rewards Visa" / "Standard").
                own = set(tokenize(text + (" " + chunk.heading.split(" > ")[-1] if chunk.is_table else "")))
                coverage, missing_rare = self.covered(q, own | ctx)
                inside, _ = self.covered(q, own)
                has_value = bool(q.value_re.search(text)) if q.value_re else True
                hits_n, conflicts = qualifier_match(q, text + " " + chunk.heading)
                s = (rank_score / top) + 2.0 * coverage + 0.5 * inside + 1.0 * hits_n - 1.5 * conflicts
                if q.value_re is not None:
                    s += 1.5 if has_value else -1.5
                s += 2.0 * weights.get(chunk.url, 0.0)   # authority of the page this passage is on
                if not q.business and not q.wants_list and BUSINESS.search(chunk.heading + " " + text[:80]):
                    s -= BUSINESS_PENALTY    # "Rewards Visa" and "Business Rewards Visa" are different products
                if chunk.is_table and q.kind in ("rate", "fee"):
                    s += 0.5    # rate tables are maintained; prose often quotes old numbers
                    if page_path(chunk.url) in RATE_SHEETS:
                        s += 0.5    # the official rate sheet beats product pages that copy it
                out.append(Scored(s, coverage, has_value, missing_rare, hits_n > 0 and not conflicts,
                                  conflicts > 0 and hits_n == 0, chunk, text, order))
                order += 1
        out.sort(key=lambda u: (-u.score, u.order))
        return out

    def refuse(self, best: Scored) -> bool:
        floor = MIN_COVERAGE_QUALIFIED if best.qualified else MIN_COVERAGE
        return best.coverage < floor or not best.has_value or best.missing_rare or best.contradicts

    def compose(self, q: Query, units: List[Scored],
                fix_row=None) -> Tuple[str, List[Chunk], List[str]]:
        """Answer text, the chunks it came from, and any extra source URLs.

        `fix_row(chunk, row)` may swap a row for the current value of the same fact stated on a
        more authoritative page; it returns (row, url-or-None)."""
        best = units[0]
        if q.kind == "url":
            return self.compose_url(q, units), [best.chunk], []
        if not best.chunk.is_table:
            text = best.text
            if len(text) > MAX_PROSE_CHARS:
                text = text[:MAX_PROSE_CHARS].rsplit(" ", 1)[0] + "..."
            # Lead with the page's own heading so the answer names what it is about,
            # unless that heading carries numbers (impact stats, counts) that could mislead.
            title = best.chunk.heading.split(" > ")[0].strip()
            if title and not re.search(r"\d", title) and title.lower() not in text.lower():
                text = "%s: %s" % (title, text)
            return text, [best.chunk], []

        # Table answer: the best row plus its ties; a broad question gets the whole table,
        # and a list question also gets the other strong tables (personal + business cards).
        tables: List[Chunk] = [best.chunk]
        if q.wants_list:
            for u in units:
                if (len(tables) < MAX_TABLES and u.chunk.is_table
                        and u.chunk.text not in {t.text for t in tables}   # same table on another page
                        and u.score >= best.score - LIST_TIE and not self.refuse(u)):
                    tables.append(u.chunk)
        lines = []
        extra: List[str] = []
        for table in tables:
            if not q.numbers and not q.term_numbers:
                rows = list(table.rows)
            else:
                rows = [u.text for u in units if u.chunk == table and u.score >= best.score - TIE]
            label = table.heading.split(" > ")[-1]
            fixed = []
            for r in rows[:MAX_ROWS]:
                if fix_row is not None:
                    r, url = fix_row(table, r)
                    if url:
                        extra.append(url)
                fixed.append(r)
            lines.append(label + ":\n" + "\n".join("- " + r for r in dict.fromkeys(fixed)))
        return "\n".join(lines), tables, extra

    def compose_url(self, q: Query, units: List[Scored]) -> str:
        """Point to the most relevant link in the best passages, else the page itself."""
        want = set(q.focus)
        best_link, best_hits = None, 0
        action = bool(ACTION.search(q.text))
        for u in units[:5]:
            for anchor, url in u.chunk.links:
                if not action and APP_HOST.search(url):
                    continue    # an application form is not where you *read* about something
                # The anchor text says what the link is for; the URL itself is weaker evidence.
                hits = 2 * len(want & set(tokenize(anchor))) + len(
                    want & set(tokenize(url.replace("/", " ").replace(".", " ").replace("-", " "))))
                if url.split("#")[0].rstrip("/") == u.chunk.url.rstrip("/"):
                    hits -= 1   # a link back to the same page adds nothing
                if hits > best_hits:
                    best_link, best_hits = url, hits
        url = best_link or units[0].chunk.url
        return "You can use %s" % url



# --------------------------------------------------------------------------------------
# 6. Facts: every dated, sourced value; which one is current; where the site contradicts itself
# --------------------------------------------------------------------------------------
# A fact key names one thing that should have exactly one value, e.g.
# "auto motorcycle loans 2024-27 model year 60 months | APR". Every page version that states
# it is remembered, so the system can say what is true in any snapshot and why.

PCT = re.compile(r"(\d+(?:\.\d+)?)\s?%(?:\s*[–-]\s*(\d+(?:\.\d+)?)\s?%)?")
KEY_FILLER = frozenset("account accounts rate rates the of and".split())
METRIC_PRIORITY = ("APY", "APR", "rate", "dividend rate")   # which column headlines a row

# (fact name, how the topic reads, pattern whose group 1 is the value, answer template)
PROSE_FACTS = [
    ("branch count", "Number of Dupaco branch locations",
     re.compile(r"\b(\d{1,3}) branch(?: office)? locations\b", re.I),
     "Dupaco has {v} branch office locations."),
    ("employee count", "Number of Dupaco employees",
     re.compile(r"\b((?:nearly|over|more than|about|approximately) \d{3,4}) employees\b", re.I),
     "Dupaco has {v} employees."),
    ("total assets", "Dupaco total assets",
     re.compile(r"\bassets (?:exceeding|of more than|of over|of) (\$\d+(?:\.\d+)? billion)", re.I),
     "Dupaco has assets exceeding {v}."),
    ("routing number", "Dupaco routing number",
     re.compile(r"\b(?:routing|ABA) number\b[^0-9\n]{0,40}(\d{9})\b", re.I),
     "Dupaco's routing number is {v}."),
    ("forbes recognitions", "Times Forbes has recognized Dupaco",
     re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth) time Forbes", re.I),
     "Forbes has recognized Dupaco as a top credit union; the most recent release says it was the {v} time."),
]
# Questions answered straight from a resolved prose fact.
FACT_QUESTIONS = [
    ("branch count", re.compile(r"\bhow many (?:\w+ ){0,2}(?:branch|location)|\bnumber of (?:\w+ )?(?:branch|location)", re.I)),
    ("employee count", re.compile(r"\bhow many (?:people|employees|staff)(?: does| do| work)", re.I)),
    ("total assets", re.compile(r"\b(total )?assets\b|\bhow big is\b", re.I)),
    ("forbes recognitions", re.compile(r"\bforbes\b", re.I)),
]

MAX_CONFLICTS = 35
MAX_PER_TABLE = 2        # tiers of one product table reported, so one stale table cannot fill the list


def canon_pct(m) -> str:
    lo = "%g%%" % float(m.group(1))
    return lo + ("–%g%%" % float(m.group(2)) if m.group(2) else "")


def norm_key(text: str) -> str:
    t = text.lower().replace("–", "-").replace("—", "-").replace("\xa0", " ")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)            # "$832,750" and "$832750" are one tier
    t = re.sub(r"\(as low as\)|[*_]", " ", t)
    words = [w.strip(".-") for w in re.findall(r"[a-z0-9$.+-]+", t)]
    words = [stem(w) if w.isalpha() else w for w in words if w and w not in KEY_FILLER]
    # Word order varies with the label ("Special" + "10 Month ... Certificate Special" vs
    # "Special Certificates" + "10 Month ..."), so a key is the sorted set of its words.
    return " ".join(sorted({w for w in words if w and w not in KEY_FILLER}))


YEAR_RANGE = re.compile(r"\b(?:19|20)\d\d-(?:19|20)?\d\d\b|\b(?:19|20)\d\d\b")


def range_end(key: str) -> int:
    """Last model year a key covers ('2024-27' -> 2027), 0 if it names none."""
    m = re.search(r"\b((?:19|20)\d\d)(?:-((?:19|20)?\d\d))?\b", key)
    if not m:
        return 0
    start, end = m.group(1), m.group(2)
    if not end:
        return int(start)
    return int(end) if len(end) == 4 else int(start[:2] + end)


def family_of(key: str) -> str:
    """The key without model-year ranges: '2023-26 model year 60 months' and its successor
    tier '2024-27 model year 60 months' are the same product and term."""
    return re.sub(r"\s+", " ", YEAR_RANGE.sub(" ", key)).strip()


def metric_of(header: str) -> Optional[str]:
    h = header.lower()
    if "apy" in h or "yield" in h:
        return "APY"
    if "apr" in h or "annual percentage rate" in h:
        return "APR"
    if "dividend" in h:
        return "dividend rate"
    if re.search(r"\brate\b", h):
        return "rate"
    return None


@dataclass(frozen=True)
class FactEntry:
    key: str
    value: str
    url: str
    content_hash: str
    row: str          # the rendered table row or sentence that states it


# Pages name the same product differently ("Auto Loan Rates" on a promo, "Auto & Motorcycle
# Loans" on the rate sheet); when the label is the only product name in a row, unify it.
PRODUCT_LABELS = [
    (re.compile(r"\bauto\b", re.I), "auto loan"),
    (re.compile(r"\bboat|\brv\b", re.I), "boat rv loan"),
    (re.compile(r"\batv\b|snowmobile|jet ski", re.I), "powersport loan"),
    (re.compile(r"\bheloc\b|line of credit", re.I), "heloc"),
    (re.compile(r"home improvement", re.I), "home improvement loan"),
    (re.compile(r"\bpersonal loan", re.I), "personal loan"),
]


def product_label(label: str) -> str:
    for pattern, name in PRODUCT_LABELS:
        if pattern.search(label):
            return name
    return label


def table_facts(chunk: Chunk) -> List[Tuple[str, str, str, str]]:
    """(key, topic, value, row) for every percentage in every row of a table chunk."""
    label = chunk.heading.split(" > ")[-1]
    out = []
    for row in chunk.rows:
        if ": " not in row:
            continue
        subject, values = row.rsplit(": ", 1)
        base = norm_key(product_label(label) + " " + subject)
        topic = subject if norm_key(label) in norm_key(subject) else "%s – %s" % (label, subject)
        for part in values.split("; "):
            m = PCT.search(part)
            metric = metric_of(part[: m.start()]) if m else None
            if metric:
                out.append((base + " | " + metric, "%s (%s)" % (topic, metric), canon_pct(m), row))
    return out


LABELLED_PCT = re.compile(r"^\s*[-*]?\s*([A-Za-z0-9][^:%\n]{2,80}?)\s*:\s*(\d+(?:\.\d+)?)\s?%\s*$", re.M)


def prose_facts(text: str) -> List[Tuple[str, str, str, str]]:
    out = []
    # "6-month certificate APY: 3.50%" lines state a fact as plainly as a table row does.
    for m in LABELLED_PCT.finditer(text):
        label = m.group(1).strip()
        metric = metric_of(label)
        if metric and not EXPIRED.search(m.group(0)):
            base = norm_key(re.sub(r"\b(apy|apr|dividend|rate|annual percentage (yield|rate))\b", " ", label, flags=re.I))
            out.append((base + " | " + metric, label, "%g%%" % float(m.group(2)), m.group(0).strip()))
    for name, topic, pattern, _ in PROSE_FACTS:
        for m in pattern.finditer(text):
            value = m.group(1).lower() if name == "forbes recognitions" else m.group(1)
            line = text[max(0, m.start() - 120): m.end() + 60].replace("\n", " ").strip()
            out.append((name, topic, value, line))
    return out


@dataclass
class Resolution:
    value: str
    entry: FactEntry
    version: PageVersion
    reason: str


class FactStore:
    def __init__(self, store: PageStore):
        self.store = store
        self.entries: Dict[str, List[FactEntry]] = defaultdict(list)
        self.topics: Dict[str, str] = {}
        self.families: Dict[str, set] = defaultdict(set)   # family_of(key) -> keys
        self.prose_keys = {name for name, *_ in PROSE_FACTS}
        self._seen = set()
        self._cache: Dict[Tuple[str, str], Optional[Resolution]] = {}

    def add_page(self, url: str, content_hash: str, chunks: List[Chunk], text: str) -> None:
        if (url, content_hash) in self._seen:
            return               # same page content as an earlier snapshot: already recorded
        self._seen.add((url, content_hash))
        found = [f for c in chunks if c.is_table for f in table_facts(c)] + prose_facts(text)
        for key, topic, value, row in found:
            self.entries[key].append(FactEntry(key, value, url, content_hash, row))
            self.topics.setdefault(key, topic)
            self.families[family_of(key)].add(key)
        self._cache.clear()

    def present(self, entry: FactEntry, snapshot_id: str) -> Optional[PageVersion]:
        v = self.store.version(snapshot_id, entry.url)
        return v if v is not None and not v.carried and v.content_hash == entry.content_hash else None

    def rank(self, key: str, v: PageVersion) -> Tuple:
        """Lower sorts first. Tables: rate sheet, then page tier, then newer. Prose: newest statement."""
        if key in self.prose_keys:
            # Newest *published* statement wins; an undated maintained page counts as current,
            # an undated article (or boilerplate footer) as oldest.
            when = v.published or (v.taken_on if v.tier == TIER_PRODUCT else "0000-00-00")
            return (-int(when.replace("-", "")), v.tier)
        date = int((v.published or v.taken_on).replace("-", ""))
        return (0 if page_path(v.url) in RATE_SHEETS else 1, v.tier, -date)

    def resolve(self, key: str, snapshot_id: str) -> Optional[Resolution]:
        ck = (key, snapshot_id)
        if ck not in self._cache:
            self._cache[ck] = self._resolve(key, snapshot_id)
        return self._cache[ck]

    def _resolve(self, key: str, snapshot_id: str) -> Optional[Resolution]:
        cands = [(self.rank(key, v), e, v) for e in self.entries.get(key, ())
                 for v in [self.present(e, snapshot_id)] if v is not None and v.tier != TIER_LISTING]
        if not cands:
            return None
        cands.sort(key=lambda c: (c[0], c[1].value, c[1].url))
        best_rank = cands[0][0]
        group = [c for c in cands if c[0] == best_rank]
        # Several equally authoritative pages: the value most of them state.
        value = Counter(c[1].value for c in group).most_common(1)[0][0]
        _, entry, version = next(c for c in group if c[1].value == value)
        if key not in self.prose_keys and page_path(version.url) not in RATE_SHEETS:
            successor = self.successor(key, snapshot_id)
            if successor is not None:
                return successor
        return Resolution(value, entry, version, self.explain(key, version))

    def successor(self, key: str, snapshot_id: str) -> Optional[Resolution]:
        """When the rate sheet no longer lists a tier but lists exactly one newer tier of the same
        product and term, that row is the current value (e.g. model years rolled forward)."""
        family = family_of(key)
        if family == key:
            return None
        own_end = range_end(key)
        sheet_keys = [k for k in self.families.get(family, ()) if k != key and any(
            page_path(e.url) in RATE_SHEETS and self.present(e, snapshot_id) for e in self.entries[k])]
        # The successor is the nearest newer range: 2023-26 -> 2024-27 (not 2017-20).
        newer = sorted((range_end(k), k) for k in sheet_keys if own_end and range_end(k) > own_end)
        if not newer:
            return None
        sheet_keys = [newer[0][1]]
        res = self.resolve(sheet_keys[0], snapshot_id)
        if res is None or page_path(res.version.url) not in RATE_SHEETS:
            return None
        return Resolution(res.value, res.entry, res.version,
                          res.reason + "; this tier was replaced by \"%s\"" % self.topics[sheet_keys[0]])

    def explain(self, key: str, v: PageVersion) -> str:
        path = "/" + page_path(v.url)
        if key in self.prose_keys:
            when = v.published or v.taken_on
            return "most recent statement (%s, %s)" % (when, path)
        if page_path(v.url) in RATE_SHEETS:
            return "official rate sheet (%s)" % path
        return "most authoritative page stating it (%s)" % path

    def conflicts(self) -> List[Tuple[float, str, Conflict]]:
        """Every key the site has stated with more than one value, with the winner and why."""
        snaps = self.store.snapshot_ids
        if not snaps:
            return []
        latest = snaps[-1]
        dates = dict(zip(self.store.snapshot_ids, self.store.snapshot_dates))
        out = []
        for key, entries in self.entries.items():
            # A key that one page states with several values is a parsing collision, not a contradiction.
            per_page = defaultdict(set)
            for e in entries:
                per_page[(e.url, e.content_hash)].add(e.value)
            if any(len(vals) > 1 for vals in per_page.values()):
                continue
            values = {e.value for e in entries}
            if len(values) < 2:
                continue
            res = self.resolve(key, latest)
            last_snap = latest
            if res is None:
                last_snap = next((s for s in reversed(snaps) if self.resolve(key, s)), None)
                if last_snap is None:
                    continue
                res = self.resolve(key, last_snap)
            seen_in = defaultdict(set)     # value -> snapshots it appears in
            where = {}                     # value -> one (url, snapshot) stating it, newest first
            for s in reversed(snaps):
                for e in entries:
                    if self.present(e, s):
                        seen_in[e.value].add(s)
                        where.setdefault(e.value, (e.url, s, e.row))
            losers = sorted(values - {res.value}, key=lambda val: max(dates[s] for s in seen_in[val]) if seen_in[val] else "")
            live = {val for val in values if latest in seen_in[val]}
            stale_live = sorted(live - {res.value})
            parts = []
            if stale_live:
                pages = sorted({"/" + page_path(e.url) for e in entries
                                if e.value in stale_live and self.present(e, latest)})
                parts.append("%s still shown on %s in the latest snapshot" % (", ".join(stale_live), ", ".join(pages[:3])))
            gone = [val for val in losers if val not in live]
            if gone:
                parts.append("%s last seen %s" % (", ".join(gone), max(dates[s] for s in seen_in[gone[-1]])))
            reason = "%s wins: %s. %s" % (res.value, res.reason, "; ".join(parts))
            if last_snap != latest:
                reason += " (no longer listed after %s)" % dates[last_snap]
            vals = [res.value] + losers
            sources = [Source(url=where[val][0], snapshot_id=where[val][1], quote=where[val][2][:200]) for val in vals
                       if val in where]
            prio = (3 if stale_live else 1) + (2 if key in self.prose_keys else 0) \
                + (1 if page_path(res.version.url) in RATE_SHEETS else 0) - (1 if last_snap != latest else 0)
            pages = len({e.url for e in entries})     # widely repeated rows are the headline tiers
            out.append((prio + min(pages, 8) / 10.0, key, Conflict(topic=self.topics[key], values=vals,
                                                                  sources=sources, resolution=res.value,
                                                                  reason=reason)))
        return out


def row_key(key: str) -> str:
    return key.rsplit(" | ", 1)[0]


TIER_WORDS = re.compile(
    r"\$[\d.]+(?:-\$?[\d.]+)?\+?|\b\d[\d.]*\+|\b(?:or|less|older|model|year)\b|"
    r"\b(?:19|20)\d\d(?:-(?:19|20)?\d\d)?\b")


def tier_group(key: str) -> str:
    """The product and term a tiered row belongs to, without its balance / amount / model year."""
    return re.sub(r"\s+", " ", TIER_WORDS.sub(" ", key)).strip()


def merge_conflicts(members: List[Conflict]) -> Conflict:
    """One conflict for all tiers of a product: every competing value, and each tier's winner."""
    head = members[0]
    values = list(dict.fromkeys(v for c in members for v in [c.resolution] + c.values if v))
    winners = list(dict.fromkeys(c.resolution for c in members if c.resolution))
    sources = list({s.url + (s.snapshot_id or ""): s for c in members for s in c.sources}.values())[:12]
    tiers = "; ".join("%s -> %s" % (c.topic, c.resolution) for c in members[1:4])
    return Conflict(topic="%s (and %d other tiers)" % (head.topic, len(members) - 1), values=values,
                    sources=sources, resolution=", ".join(winners),
                    reason="%s Other tiers: %s." % (head.reason, tiers))


# --------------------------------------------------------------------------------------
# 7. KnowledgeBase: the contract object
# --------------------------------------------------------------------------------------

CARRIED_WEIGHT = 0.85    # pages missing from a crawl still count, slightly discounted
EXPIRED = re.compile(r"\b(offer|promotion|promo|special)\b[^.\n]{0,30}\b(ended|expired|has ended|is over)\b"
                     r"|\boffer ended\b|\bno longer available\b", re.I)
CANDIDATES = 40          # BM25 hits pulled before reranking by authority
TOP_CHUNKS = 25          # reranked chunks whose passages compete for the answer
# No digits on purpose: a refusal that states any number is scored as a guess.
REFUSAL = "I don't know. That isn't listed on the website, so please contact us for details."


class KnowledgeBase:
    def __init__(self):
        self.store = PageStore()
        self.index = SearchIndex()
        self.composer = Composer(self.index)
        self.facts = FactStore(self.store)
        self._cache: Dict[Tuple[str, str], Answer] = {}

    def ingest(self, snapshot: Snapshot) -> None:
        self.store.add(snapshot)
        for p in snapshot.pages:
            if classify(p.url) == TIER_JUNK:
                continue
            chunks = split_page(p.url, p.title, p.markdown)
            for chunk in chunks:
                self.index.add(snapshot.snapshot_id, chunk)
            v = self.store.version(snapshot.snapshot_id, p.url)
            self.facts.add_page(p.url, v.content_hash, chunks, "\n".join(c.text for c in chunks))
        for v in self.store.carried.get(snapshot.snapshot_id, ()):
            if v.tier != TIER_JUNK:
                for chunk in split_page(v.url, v.title, self.store.content[v.content_hash]):
                    self.index.add(snapshot.snapshot_id, chunk)

    def _retrieve(self, question: str, snapshot_id: str, k: int = 5) -> List[Tuple[float, Chunk]]:
        """BM25 candidates reranked by page authority as of that snapshot."""
        strict = bool(CURRENT_FACT.search(question))
        ranked = []
        for order, (score, chunk) in enumerate(self.index.search(question, snapshot_id, k=CANDIDATES)):
            v = self.store.version(snapshot_id, chunk.url)
            w = v.weight(strict) if v is not None else 0.0
            if w > 0:
                ranked.append((score * w, order, chunk))
        ranked.sort(key=lambda r: (-r[0], r[1]))
        return [(s, c) for s, _, c in ranked[:k]]

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        sid = self.store.snapshot_for(as_of)
        key = (" ".join(question.lower().split()), sid or "")
        if key not in self._cache:
            self._cache[key] = self._answer(question, sid)
        return self._cache[key]

    def _answer(self, question: str, sid: Optional[str]) -> Answer:
        q = parse_query(question)
        direct = self._fact_answer(question, sid) if sid else None
        if direct is not None:
            return direct
        hits = self._retrieve(" ".join([q.text] + q.expansion), sid, k=TOP_CHUNKS) if sid else []
        if not hits:
            return Answer(text=REFUSAL, confident=False)
        strict = bool(CURRENT_FACT.search(question))
        weights = {}
        for _, c in hits:
            v = self.store.version(sid, c.url)
            weights[c.url] = v.weight(strict) if v is not None else 0.0
        units = self.composer.score(q, hits, weights)
        if not units or self.composer.refuse(units[0]):
            return Answer(text=REFUSAL, confident=False)
        notes: List[str] = []
        text, chunks, extra = self.composer.compose(q, units, lambda c, r: self._fix_row(c, r, sid, notes))
        sources = [Source(url=c.url, snapshot_id=sid, quote=c.heading[:200]) for c in chunks]
        sources += [Source(url=u, snapshot_id=sid) for u in extra]
        return Answer(text=text, sources=list({s.url: s for s in sources}.values()),
                      conflict_note="; ".join(dict.fromkeys(notes)) or None)

    def _fact_answer(self, question: str, sid: str) -> Optional[Answer]:
        """Count-style questions answered from the resolved prose fact (newest statement wins)."""
        for name, pattern in FACT_QUESTIONS:
            if not pattern.search(question):
                continue
            res = self.facts.resolve(name, sid)
            if res is None:
                return None
            template = next(t for n, _, _, t in PROSE_FACTS if n == name)
            others = {e.value for e in self.facts.entries[name] if self.facts.present(e, sid)} - {res.value}
            note = ("Other pages say %s; using the %s." % (", ".join(sorted(others)), res.reason)) if others else None
            return Answer(text=template.format(v=res.value),
                          sources=[Source(url=res.entry.url, snapshot_id=sid, quote=res.entry.row[:200])],
                          conflict_note=note)
        return None

    def _fix_row(self, chunk: Chunk, row: str, sid: str, notes: List[str]) -> Tuple[str, Optional[str]]:
        """Stale-value guard: if a more authoritative page states this row's headline fact with a
        different value, answer with that page's row instead."""
        facts = [f for f in table_facts(Chunk(chunk.url, chunk.heading, row, rows=(row,))) if f[3] == row]
        facts.sort(key=lambda f: METRIC_PRIORITY.index(f[0].rsplit(" | ", 1)[1]))
        for key, _, value, _ in facts[:1]:
            res = self.facts.resolve(key, sid)
            if res is not None and res.value != value and res.entry.url != chunk.url:
                notes.append("%s shows %s; using %s from the %s" % (
                    "/" + page_path(chunk.url), value, res.value, res.reason))
                return res.entry.row, res.entry.url
        return row, None

    def conflicts(self) -> List[Conflict]:
        """One entry per contradicted fact, most important first.

        A table row stated with several metrics (dividend rate and APY) is reported once, under
        its headline metric, so one stale row does not flood the list."""
        # 1. One metric per row: the headline one (APY for deposits, APR for loans).
        rows: Dict[str, Tuple[float, int, str, Conflict]] = {}
        for prio, key, c in self.facts.conflicts():
            metric = key.rsplit(" | ", 1)[1] if " | " in key else ""
            m = METRIC_PRIORITY.index(metric) if metric in METRIC_PRIORITY else len(METRIC_PRIORITY)
            old = rows.get(row_key(key))
            if old is None or (m, -prio) < (old[1], -old[0]):
                rows[row_key(key)] = (prio, m, key, c)
        # 2. Tiers of one product and term (loan amounts, model years, balances) form one conflict.
        groups: Dict[str, List[Tuple[float, int, str, Conflict]]] = defaultdict(list)
        for item in rows.values():
            groups[tier_group(item[2])].append(item)
        merged = []
        for members in groups.values():
            members.sort(key=lambda t: (-t[0], t[1], t[3].topic))
            prio, m, key, head = members[0]
            if len(members) > 1:
                head = merge_conflicts([t[3] for t in members])
            merged.append((prio, m, head))
        merged.sort(key=lambda t: (-t[0], t[1], t[2].topic))
        return [c for _, _, c in merged[:MAX_CONFLICTS]]


def build() -> KnowledgeBase:
    return KnowledgeBase()
