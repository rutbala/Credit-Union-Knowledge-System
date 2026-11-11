"""Optional LLM layer: a fallback answerer for questions the rules cannot settle, and an optional
rewriter. The deterministic system stays the source of truth:

  * the LLM only sees passages the retriever found (stale tables excluded), labelled with page
    type and date, and must answer from them or say NOT_FOUND;
  * every number in its answer must appear in those passages or in the question, otherwise the
    deterministic answer is kept (no invented or remembered rates);
  * any error, timeout, refusal or empty reply -> the deterministic answer is kept.

Providers (first one configured wins; nothing configured -> LLM off):
  ANTHROPIC_API_KEY                      Claude via the anthropic SDK (model: ANTHROPIC_MODEL,
                                         default claude-haiku-5-5, chosen for latency)
  AZURE_OPENAI_API_KEY + AZURE_OPENAI_MODEL [+ AZURE_OPENAI_ENDPOINT]
                                         Azure OpenAI Responses API (standard library only)

Switches (real environment variables win over a .env file found above this folder):
  KB_LLM          off | on       default: on whenever a provider key is configured, off otherwise
                                 (measured on our holdout sets: 0.753 -> 0.876, see NOTES.md)
  KB_LLM_MODE     fallback | polish | both   default fallback (polish = reword every answer)
  KB_LLM_ROUTE    narrow | prose (default prose; see solution.KnowledgeBase._needs_llm)
  KB_LLM_TIMEOUT  seconds per call, default 3 (the speed score drops past 2 s per answer)
"""
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

DEFAULT_ENDPOINT = "https://llm-model-test-hars.services.ai.azure.com/openai/v1/responses"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
MAX_OUTPUT_TOKENS = 1024     # includes reasoning tokens on reasoning models; too low -> "incomplete"
NOT_FOUND = "NOT_FOUND"

POLISH_INSTRUCTIONS = (
    "You rewrite a draft answer for a credit union member so it reads naturally. "
    "Use only the information in the draft. Keep every number, rate, APY/APR label, fee, date, time, "
    "phone number, URL and product name exactly as written in the draft. Do not add any number that is "
    "not in the draft and do not leave any out. Keep words such as 'noon' or state names as written. "
    "If the draft lists several tiers or rows, keep all of them, as a short list. "
    "Do not add advice, caveats or greetings. Reply with the rewritten answer only."
)

ANSWER_INSTRUCTIONS = (
    "You answer a credit union member's question using ONLY the numbered excerpts from the credit "
    "union's website. Each excerpt shows its page and page type. Official rate and product pages beat "
    "promotions, calculators and news articles; when excerpts disagree, prefer the official page, and "
    "among articles the most recently dated one. Ignore offers that have ended.\n"
    "Rules:\n"
    "- Quote every value exactly as written in the excerpts (rates, APY/APR, amounts, phone numbers, "
    "URLs, dates, times, names). Never use a number that is not in the excerpts.\n"
    "- Answer only the product, tier and term asked about; do not list other tiers or products.\n"
    "- If the question claims a value (\"do any tiers pay 0.35%?\"), say whether the excerpts support "
    "it and state the actual value they give.\n"
    "- If the excerpts do not answer the question, reply exactly NOT_FOUND and nothing else.\n"
    "- Otherwise answer in one to three short sentences (or a short list), with no greeting or advice.\n"
    "- On a final line write SOURCES: followed by the excerpt numbers you used, e.g. SOURCES: 2, 5"
)

# Same idea as the scorer's number extraction: "$2.50", "3.5%", "1,500", "8:30" -> comparable values.
_NUM = re.compile(r"(\$)?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(%|percent)?", re.I)
_SAYS_NOT_FOUND = re.compile(
    r"\b(excerpts?|information provided|provided (?:text|information))\b[^.]{0,40}\b(do(?:es)? not|don't|doesn't|"
    r"no(?:t)? (?:include|mention|list|provide|specify|contain))\b|\bnot (?:listed|mentioned|specified) in\b", re.I)
_SOURCES = re.compile(r"^\s*SOURCES?\s*:\s*(.*)$", re.I | re.M)


def _numbers(text: str) -> set:
    out = set()
    for m in _NUM.finditer(text or ""):
        unit = "$" if m.group(1) else ("%" if m.group(4) else "")
        out.add((unit, round(float(m.group(2).replace(",", "") + ("." + m.group(3) if m.group(3) else "")), 4)))
    return out


def _values(text: str) -> set:
    """Numbers by value only: '$1,000' in the answer may be written '1,000' in a table cell."""
    return {v for _, v in _numbers(text)}


def _load_dotenv() -> None:
    """Fill unset variables from the nearest .env above this file (never overrides real env vars)."""
    for d in Path(__file__).resolve().parents:
        f = d / ".env"
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


# ----------------------------------------------------------------------------- providers

class _Azure:
    name = "azure"

    def __init__(self, endpoint: str, key: str, model: str):
        self.endpoint, self.key, self.model = endpoint, key, model

    def complete(self, instructions: str, prompt: str, timeout: float) -> Optional[str]:
        body = json.dumps({"model": self.model, "instructions": instructions, "input": prompt,
                           "max_output_tokens": MAX_OUTPUT_TOKENS}).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=body, method="POST", headers={
            "Content-Type": "application/json", "api-key": self.key})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            print("[kb] LLM call failed: HTTP %d %s" % (e.code, e.read().decode("utf-8", "replace")[:200]),
                  file=sys.stderr)
            return None
        except Exception as e:      # timeout, DNS, connection refused, bad JSON
            print("[kb] LLM call failed: %s" % type(e).__name__, file=sys.stderr)
            return None
        if data.get("status") not in (None, "completed"):
            return None             # "incomplete" (token cap) or "failed"
        parts = [c.get("text", "") for item in data.get("output", []) if item.get("type") == "message"
                 for c in item.get("content", []) if c.get("type") == "output_text"]
        return "".join(parts).strip() or None


class _Anthropic:
    name = "anthropic"

    def __init__(self, model: str):
        import anthropic            # optional dependency; ImportError -> provider unavailable
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=0)   # a retry would blow the latency budget
        self.model = model

    def complete(self, instructions: str, prompt: str, timeout: float) -> Optional[str]:
        kwargs = dict(model=self.model, max_tokens=MAX_OUTPUT_TOKENS, system=instructions,
                      messages=[{"role": "user", "content": prompt}])
        try:
            client = self.client.with_options(timeout=timeout)
            try:
                resp = client.messages.create(output_config={"effort": "low"}, **kwargs)
            except TypeError:       # an older SDK without output_config
                resp = client.messages.create(**kwargs)
        except Exception as e:      # timeout, auth, rate limit, network
            print("[kb] LLM call failed: %s" % type(e).__name__, file=sys.stderr)
            return None
        if resp.stop_reason in ("refusal", "max_tokens"):
            return None
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip() or None


def _provider():
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        try:
            return _Anthropic(os.environ.get("ANTHROPIC_MODEL", "").strip() or DEFAULT_ANTHROPIC_MODEL)
        except Exception as e:
            print("[kb] anthropic SDK unavailable (%s)" % type(e).__name__, file=sys.stderr)
    key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    model = os.environ.get("AZURE_OPENAI_MODEL", "").strip()
    if key and model:
        return _Azure(os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip() or DEFAULT_ENDPOINT, key, model)
    return None


# ----------------------------------------------------------------------------- the writer

class Writer:
    def __init__(self, provider, timeout: float, mode: str):
        self.provider, self.timeout, self.mode = provider, timeout, mode
        self.calls = self.kept = 0

    @classmethod
    def load(cls) -> Optional["Writer"]:
        _load_dotenv()
        if os.environ.get("KB_LLM", "").strip().lower() == "off":
            return None                 # explicitly disabled; otherwise on when a key is set
        provider = _provider()
        if provider is None:
            if os.environ.get("KB_LLM", "").lower() == "on":
                print("[kb] KB_LLM=on but no ANTHROPIC_API_KEY or AZURE_OPENAI_API_KEY/MODEL; "
                      "deterministic answers only", file=sys.stderr)
            return None
        mode = os.environ.get("KB_LLM_MODE", "fallback").lower()
        return cls(provider, float(os.environ.get("KB_LLM_TIMEOUT", "3")), mode)

    @property
    def fallback_enabled(self) -> bool:
        return self.mode in ("fallback", "both")

    @property
    def polish_enabled(self) -> bool:
        return self.mode in ("polish", "both")

    def polish(self, question: str, draft: str) -> str:
        """The reworded answer if it keeps exactly the draft's numbers, else the draft."""
        if not self.polish_enabled:
            return draft
        self.calls += 1
        out = self.provider.complete(POLISH_INSTRUCTIONS,
                                     "Member question: %s\n\nDraft answer:\n%s" % (question, draft), self.timeout)
        if out is None or _numbers(out) != _numbers(draft):
            return draft
        self.kept += 1
        return out

    def answer(self, question: str, today: str,
               excerpts: Sequence[Tuple[str, str, str]]) -> Optional[Tuple[str, List[int]]]:
        """Answer from (page, page type, text) excerpts.

        Returns (answer text, indices of excerpts used), ("NOT_FOUND", []) when the excerpts do not
        answer it, or None when the call failed or the answer used a number the excerpts lack."""
        if not excerpts:
            return None
        blocks = ["[%d] %s (%s)\n%s" % (i + 1, page, kind, text) for i, (page, kind, text) in enumerate(excerpts)]
        prompt = "Today is %s.\nMember question: %s\n\nExcerpts:\n\n%s" % (today, question, "\n\n".join(blocks))
        self.calls += 1
        out = self.provider.complete(ANSWER_INSTRUCTIONS, prompt, self.timeout)
        if not out:
            return None
        if out.strip().upper().startswith(NOT_FOUND) or _SAYS_NOT_FOUND.search(out):
            return NOT_FOUND, []    # models often say "the excerpts do not ..." instead of NOT_FOUND
        m = _SOURCES.search(out)
        used = [int(n) - 1 for n in re.findall(r"\d+", m.group(1))] if m else []
        text = _SOURCES.sub("", out).strip()
        evidence = set().union(*(_values(t) for _, _, t in excerpts)) | _values(question)
        if not text or not _values(text) <= evidence:
            return None             # a number from nowhere: keep the deterministic answer
        self.kept += 1
        return text, [i for i in used if 0 <= i < len(excerpts)]
