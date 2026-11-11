"""Ingestion clean-up that needs the whole snapshot, not one page at a time.

Boilerplate: the same block of text (related-article teaser cards, "this link leads to a
different website" notices, tag lists) is repeated on dozens of pages. Indexed everywhere it
makes unrelated product pages look relevant ("money market" teaser on the Shine login page),
so a block repeated on many pages is kept once, on its most authoritative page, and removed
from the rest. Tables and plain headings are never removed: repeated rate tables are real
content (and the source of contradictions), and headings give chunks their structure.
"""
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

from .authority import classify, page_path

MIN_PAGES = 10          # a block on at least this many pages of one snapshot is boilerplate
MIN_BLOCK_CHARS = 30    # shorter blocks ("Read More", a lone link) are too generic to judge
LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
BLANK_LINES = re.compile(r"\n[ \t]*\n")


def _blocks(md: str) -> List[str]:
    return BLANK_LINES.split(md)


def _norm(block: str) -> str:
    return re.sub(r"\s+", " ", block).strip()


def _removable(block: str) -> bool:
    text = block.strip()
    if len(text) < MIN_BLOCK_CHARS:
        return False
    if any(line.lstrip().startswith("|") for line in text.splitlines()):
        return False          # tables are content
    if text.startswith("#") and "\n" not in text and not LINK.search(text):
        return False          # a plain section heading
    return True


def find_boilerplate(pages: Iterable[Tuple[str, str]], min_pages: int = MIN_PAGES) -> Dict[str, str]:
    """Normalized block -> the one URL allowed to keep it (most authoritative, then shortest path)."""
    seen: Dict[str, List[str]] = {}
    for url, md in pages:
        for block in {b for b in _blocks(md) if _removable(b)}:
            seen.setdefault(_norm(block), []).append(url)
    keep = {}
    for block, urls in seen.items():
        if len(set(urls)) >= min_pages:
            keep[block] = min(set(urls), key=lambda u: (classify(u), len(page_path(u)), u))
    return keep


def strip_boilerplate(url: str, md: str, boilerplate: Dict[str, str]) -> str:
    """The page without boilerplate blocks it does not own."""
    if not boilerplate:
        return md
    kept = [b for b in _blocks(md) if boilerplate.get(_norm(b), url) == url]
    return "\n\n".join(kept)


def boilerplate_stats(boilerplate: Dict[str, str]) -> Counter:
    return Counter(classify(u) for u in boilerplate.values())
