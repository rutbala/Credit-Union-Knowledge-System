"""Optional LLM writer: rewords a composed answer through an Azure OpenAI Responses endpoint.

Off unless KB_LLM=on AND an API key and model (deployment) name are configured. Standard library
only (urllib), so it adds no dependencies. The deterministic answer is always the fallback:

  * any error, timeout, empty or incomplete response  -> the deterministic text is kept
  * the rewrite must state exactly the same numbers as the draft (no new values, none dropped),
    otherwise it is discarded. Accuracy and consistency are scored on those numbers, so the
    model may change the wording but never the facts.

Configuration (real environment variables win over a .env file found above this folder):

  KB_LLM                 on | off (default off)
  AZURE_OPENAI_API_KEY   the key for the Azure resource
  AZURE_OPENAI_MODEL     the deployment name to call (the Responses API "model" field)
  AZURE_OPENAI_ENDPOINT  defaults to the project's Responses URL below
  KB_LLM_TIMEOUT         seconds per call (default 8)
"""
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

DEFAULT_ENDPOINT = "https://llm-model-test-hars.services.ai.azure.com/openai/v1/responses"
MAX_OUTPUT_TOKENS = 1024     # includes reasoning tokens on reasoning models; too low -> "incomplete"

INSTRUCTIONS = (
    "You rewrite a draft answer for a credit union member so it reads naturally. "
    "Use only the information in the draft. Keep every number, rate, APY/APR label, fee, date, time, "
    "phone number, URL and product name exactly as written in the draft. Do not add any number that is "
    "not in the draft and do not leave any out. Keep words such as 'noon' or state names as written. "
    "If the draft lists several tiers or rows, keep all of them, as a short list. "
    "Do not add advice, caveats or greetings. Reply with the rewritten answer only."
)

# Same idea as the scorer's number extraction: "$2.50", "3.5%", "1,500", "8:30" -> comparable values.
_NUM = re.compile(r"(\$)?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(%|percent)?", re.I)


def _numbers(text: str) -> set:
    out = set()
    for m in _NUM.finditer(text or ""):
        unit = "$" if m.group(1) else ("%" if m.group(4) else "")
        out.add((unit, round(float(m.group(2).replace(",", "") + ("." + m.group(3) if m.group(3) else "")), 4)))
    return out


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


class Writer:
    def __init__(self, endpoint: str, key: str, model: str, timeout: float):
        self.endpoint, self.key, self.model, self.timeout = endpoint, key, model, timeout
        self.calls = self.kept = 0

    @classmethod
    def load(cls) -> Optional["Writer"]:
        _load_dotenv()
        if os.environ.get("KB_LLM", "off").lower() != "on":
            return None
        key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        model = os.environ.get("AZURE_OPENAI_MODEL", "").strip()
        if not key or not model:
            print("[kb] KB_LLM=on but AZURE_OPENAI_API_KEY / AZURE_OPENAI_MODEL not set; deterministic answers only",
                  file=sys.stderr)
            return None
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip() or DEFAULT_ENDPOINT
        return cls(endpoint, key, model, float(os.environ.get("KB_LLM_TIMEOUT", "8")))

    def _call(self, question: str, draft: str) -> Optional[str]:
        body = json.dumps({
            "model": self.model,
            "instructions": INSTRUCTIONS,
            "input": "Member question: %s\n\nDraft answer:\n%s" % (question, draft),
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=body, method="POST", headers={
            "Content-Type": "application/json", "api-key": self.key})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            print("[kb] LLM call failed: HTTP %d %s" % (e.code, detail), file=sys.stderr)
            return None
        except Exception as e:      # timeout, DNS, connection refused, bad JSON
            print("[kb] LLM call failed: %s: %s" % (type(e).__name__, e), file=sys.stderr)
            return None
        if data.get("status") not in (None, "completed"):
            return None             # "incomplete" (token cap) or "failed": keep the draft
        parts = [c.get("text", "") for item in data.get("output", []) if item.get("type") == "message"
                 for c in item.get("content", []) if c.get("type") == "output_text"]
        text = "".join(parts).strip()
        return text or None

    def polish(self, question: str, draft: str) -> str:
        """The reworded answer if it keeps exactly the draft's numbers, else the draft."""
        self.calls += 1
        out = self._call(question, draft)
        if out is None or _numbers(out) != _numbers(draft):
            return draft
        self.kept += 1
        return out
