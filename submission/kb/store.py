"""PageStore: every version of every page, keyed by snapshot (history, not just latest)."""
import bisect
import dataclasses
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from .authority import authority, classify, published_on

CARRIED_WEIGHT = 0.85    # pages missing from a crawl still count, slightly discounted

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

    def add(self, snapshot) -> List[PageVersion]:
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
