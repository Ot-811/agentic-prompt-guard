"""Step 4 — Policy-as-Code Validator.

A deterministic sanity check on the rewrite before it is allowed to execute
(deck: "Pydantic schema, keyword blocklist, toxicity classifier"). Deterministic
means no LLM: the same input always yields the same verdict, so it can gate the
non-deterministic rewrite stage.
"""

import re

from .schemas import RewriteResult, RewriteStatus, ValidationResult

# Phrases that must never survive into an executable prompt.
_BLOCKLIST = [
    r"\boff[- ]?label\b",
    r"ignore .*(instructions|rules)",
    r"lower (their )?(dose|dosage)",
    r"even if (the )?evidence is limited",
    r"remove .*(minor|adverse|safety)",
    r"guarantee.*(cure|results)",
]

# Lightweight lexicon stand-in for the deck's toxicity classifier.
_TOXIC_TERMS = {"idiot", "stupid", "hate", "kill", "scam", "fraudulent"}


def _toxicity_score(text: str) -> float:
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in _TOXIC_TERMS)
    return hits / len(tokens)


def validate(rewrite: RewriteResult) -> ValidationResult:
    # 1. Schema/structure check: a rewrite that claims REWRITTEN must carry text.
    if rewrite.status == RewriteStatus.REWRITTEN and not (rewrite.rewritten_prompt or "").strip():
        return ValidationResult(passed=False, reasons=["Rewrite marked REWRITTEN but produced no prompt text."])
    if rewrite.status in (RewriteStatus.INVALID, RewriteStatus.NEEDS_CLARIFICATION):
        return ValidationResult(passed=False, reasons=[f"Rewrite status is {rewrite.status.value}; not executable."])

    candidate = rewrite.rewritten_prompt or ""

    # 2. Keyword blocklist.
    reasons = [f"Blocklisted pattern present: /{pat}/" for pat in _BLOCKLIST
               if re.search(pat, candidate, re.IGNORECASE)]

    # 3. Toxicity threshold.
    score = _toxicity_score(candidate)
    if score > 0.02:
        reasons.append(f"Toxicity score {score:.3f} exceeds threshold.")

    return ValidationResult(passed=not reasons, reasons=reasons)
