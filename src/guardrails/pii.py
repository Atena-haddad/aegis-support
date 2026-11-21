import re
from dataclasses import dataclass


@dataclass
class SanitizedInput:
    clean_text: str
    redacted_entities: list[str]


class PIIGuardrail:
    """
    Lightweight pre-LLM privacy guardrail that masks sensitive identifiers
    (emails, API tokens, Stripe IDs, IPv4 addresses) before external LLM calls.
    """

    PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "API_KEY": r"\bags_live_[A-Za-z0-9_-]+\b",
        "STRIPE_ID": r"\bch_[A-Za-z0-9_]+\b",
        "IPV4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

    def sanitize(self, text: str) -> SanitizedInput:
        clean = text
        found: list[str] = []
        for label, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, clean)
            if matches:
                found.extend([f"{label}:{m}" for m in matches])
                clean = re.sub(pattern, f"[REDACTED_{label}]", clean)
        return SanitizedInput(clean_text=clean, redacted_entities=found)