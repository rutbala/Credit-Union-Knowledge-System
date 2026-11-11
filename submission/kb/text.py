"""Text helpers: tokenizing, cleaning markdown, splitting pages into chunks."""
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
