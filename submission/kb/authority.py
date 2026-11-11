"""Authority: classify a URL into a trust tier and read an article's publish date."""
import re
from typing import Optional

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


KIND_LABEL = {TIER_PRODUCT: "official product page", TIER_PROMO: "promotion page", TIER_TOOL: "calculator or PDF",
              TIER_ARTICLE: "news or blog article", TIER_LISTING: "article listing"}


def page_kind(version) -> str:
    """How an LLM should weigh a page: its tier, the rate sheet, and an article's date."""
    if version is None:
        return "page"
    if page_path(version.url) in ("rates", "business-rates"):
        return "official rate sheet"
    kind = KIND_LABEL.get(version.tier, "page")
    if version.tier == TIER_ARTICLE and version.published:
        kind += ", published " + version.published
    return kind
