"""Step 4 — Policy-as-Code Validator.

A deterministic sanity check on the rewrite before it is allowed to execute
(deck: "Pydantic schema, keyword blocklist, toxicity classifier"). Deterministic
means no LLM: the same input always yields the same verdict, so it can gate the
non-deterministic rewrite stage.

Validation checks (in order):
  1. **Structural** — rewrite status and prompt-text sanity.
  2. **Keyword blocklist** — 24 regex patterns across 8 pharma-compliance
     categories (off-label, jailbreak, dosing, evidence, safety-data,
     PII, false claims, missing safety language).
  3. **Toxicity** — severity-weighted lexicon (HIGH / MEDIUM) with
     context-awareness (surrounding-word check) and negation detection.
     Uses absolute weighted scoring so long prompts cannot dilute a
     toxic term below the threshold.
  4. **Prompt length** — minimum-length gate on the rewritten prompt.
  5. **Category-aware policy rules** — when the detector verdict is
     provided, verify that the rewrite actually contains the safety
     constraint markers required to address the detected threat type.
"""

import re
from typing import Optional

from .schemas import (
    DetectorVerdict,
    RewriteResult,
    RewriteStatus,
    ThreatType,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# Keyword Blocklist
# ---------------------------------------------------------------------------
# Each entry: (regex_pattern, human-readable reason).
# These patterns should NEVER survive into a properly rewritten safe prompt.

_BLOCKLIST: list[tuple[str, str]] = [
    # ── Off-label / unapproved promotion ────────────────────────────────
    (r"\boff[- ]?label\b",
     "off-label content"),
    (r"\b(unapproved|not approved) (indication|use|dosing|application)\b",
     "unapproved indication reference"),
    (r"\bindication.{0,25}(has not|is not) (been )?approved\b",
     "unapproved indication reference"),
    (r"\b(promot|market|advertis|recommend|prescrib)\w*\b.{0,40}\b(unapproved|not approved|has not been approved)\b",
     "promotion of unapproved use"),
    (r"\b(avoid|omit|without).{0,30}(mention|disclos)\w*.{0,30}(regulatory|fda|approval|status)\b",
     "omission of regulatory status"),

    # ── Jailbreak / injection remnants ──────────────────────────────────
    (r"ignore .*(instructions|rules)",
     "instruction-override attempt"),
    (r"\bbypass\b.{0,30}(safety|filter|guardrail|compliance|restriction)",
     "safety-bypass attempt"),
    (r"\boverride\b.{0,30}(safety|restriction|rule|policy|guideline)",
     "policy-override attempt"),
    (r"disregard .{0,50}(safety|compliance|regulation|guideline|rule)",
     "compliance disregard"),

    # ── Medical / dosing advice ─────────────────────────────────────────
    (r"lower (their )?(dose|dosage)",
     "individualised dosing instruction"),
    (r"(adjust|change|modify|increase|reduce)\b.{0,20}\b(dose|dosage|mg|medication)\b",
     "dosage modification instruction"),
    (r"\bself[- ]?medicat",
     "self-medication reference"),

    # ── Evidence manipulation ───────────────────────────────────────────
    (r"even if (the )?evidence is limited",
     "disregard of evidence limitations"),
    (r"(overstat|exaggerat|embellish|inflate)\w*.{0,30}(evidence|efficacy|data|results?|benefits?)",
     "evidence exaggeration"),
    (r"(downplay|hide|conceal)\b.{0,30}(risks?|side effects?|adverse|harms?|dangers?)",
     "risk concealment"),

    # ── Safety data tampering ───────────────────────────────────────────
    (r"remove .*(minor|adverse|safety)",
     "safety data removal"),
    (r"(discard|exclude|suppress|omit|hide)\b.{0,20}(adverse|safety|side effects?|pharmacovigil)",
     "adverse-event suppression"),
    (r"(clean up|sanitize|scrub)\b.{0,20}(safety|adverse|pharmacovigil)",
     "safety record alteration"),

    # ── Identifiable patient data ───────────────────────────────────────
    (r"\bpatient.{0,10}(names?|identifiers?|identit(y|ies)|contacts?|address(es)?)\b",
     "patient-identifying information"),
    (r"\b(identifiable|identifying) (patient|individual|person)\b",
     "identifiable individual reference"),
    (r"\bwithout (patient )?(consent|authorization)\b",
     "action without consent"),

    # ── False guarantees / absolute claims ──────────────────────────────
    (r"guarantee.*(cure|results)",
     "unsubstantiated guarantee"),
    (r"\b(100%|completely|totally|absolutely) (safe|effective|cures?)\b",
     "absolute efficacy claim"),
    (r"\bno side effects?\b",
     "false claim of no side effects"),
    (r"\bsuperior to all\b",
     "unsubstantiated superiority claim"),

    # ── Missing safety language ─────────────────────────────────────────
    (r"\bwithout .{0,30}(safety (information|warnings?|disclaimers?)|fair[- ]?balance|ISI)\b",
     "explicit omission of required safety language"),
]


# ---------------------------------------------------------------------------
# Toxicity Lexicon (severity-tiered, context-aware)
# ---------------------------------------------------------------------------

# HIGH severity: always problematic in pharmaceutical context.  These terms
# have no legitimate use in a rewritten prompt (weight: 3.0 each).
_TOXIC_HIGH: dict[str, str] = {
    "murder": "violent term",
    "assault": "violent term",
    "bribe": "corruption",
    "bribery": "corruption",
    "kickback": "illegal payment",
    "blackmail": "coercion",
    "extort": "coercion",
    "extortion": "coercion",
    "intimidate": "coercion",
    "intimidation": "coercion",
    "counterfeit": "counterfeiting",
    "counterfeiting": "counterfeiting",
}

# MEDIUM severity: may have legitimate context depending on surrounding words.
# Context and negation checks apply (weight: 1.5 each, 0.0 if neutralised).
_TOXIC_MEDIUM: dict[str, str] = {
    "kill": "potentially violent",
    "hate": "hostile language",
    "stupid": "demeaning language",
    "idiot": "demeaning language",
    "scam": "fraud-related",
    "fraud": "fraud-related",
    "fraudulent": "fraud-related",
    "defraud": "fraud-related",
    "falsify": "data integrity violation",
    "fabricate": "data integrity violation",
    "destroy": "destructive action",
    "abuse": "misuse / harm",
    "exploit": "exploitation",
    "manipulate": "manipulation",
    "deceive": "deception",
    "deception": "deception",
    "corrupt": "corruption",
    "coerce": "coercion",
    "harass": "harassment",
    "harassment": "harassment",
    "discriminate": "discrimination",
    "retaliate": "retaliation",
    "sabotage": "sabotage",
    "suppress": "suppression",
    "withhold": "withholding",
}

# Context words that neutralise a MEDIUM-severity toxic term.
# Checked within a ±3 word window around the term.
_SAFE_CONTEXT: dict[str, frozenset[str]] = {
    "kill": frozenset({
        "process", "session", "task", "job", "thread",
        "switch", "timer", "signal", "command",
    }),
    "abuse": frozenset({
        "substance", "drug", "alcohol", "opioid",
        "prevention", "disorder", "potential",
    }),
    "exploit": frozenset({
        "vulnerability", "bug", "security", "cve", "patch",
    }),
    "destroy": frozenset({
        "file", "record", "document", "retention", "securely",
    }),
    "manipulate": frozenset({
        "data", "variable", "parameter", "image",
        "text", "spreadsheet", "chart", "visualization",
    }),
    "corrupt": frozenset({
        "data", "file", "database", "disk", "storage", "memory",
    }),
    "suppress": frozenset({
        "warning", "error", "output", "notification",
        "message", "log", "alert",
    }),
    "withhold": frozenset({"judgment", "opinion"}),
    "discriminate": frozenset({
        "between", "among", "types", "classes", "categories",
    }),
    "scam": frozenset({
        "anti", "prevention", "detection", "awareness",
        "protect", "report",
    }),
    "fraud": frozenset({
        "anti", "detection", "prevention", "reporting",
        "investigate", "compliance",
    }),
    "fraudulent": frozenset({
        "anti", "detection", "prevention", "reporting", "identify",
    }),
    "fabricate": frozenset({"pre", "prefabricated"}),
}

# Words that negate a following toxic term, rendering it safe.
# Checked within a 3-word window BEFORE the toxic term.
_NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "without", "nor",
    "don't", "doesn't", "didn't", "won't", "shouldn't",
    "mustn't", "cannot", "can't",
    "prevent", "prohibit", "prohibited",
    "avoid", "anti", "non",
})

# A prompt's absolute toxicity score must stay below this threshold.
# One HIGH term (3.0) or two un-neutralised MEDIUM terms (2 × 1.5 = 3.0)
# are enough to exceed it.  A single MEDIUM term (1.5) also exceeds it,
# which is intentionally strict for rewritten pharmaceutical prompts.
_TOXICITY_THRESHOLD: float = 1.0


def _toxicity_check(text: str) -> tuple[float, list[str]]:
    """Score *text* for toxic content.

    Uses severity-weighted scoring with context and negation awareness
    so that legitimate phrases (e.g. "kill the process", "do not suppress
    adverse events") are not penalised while genuinely harmful terms are
    always caught regardless of prompt length.

    Returns ``(weighted_score, detail_reasons)``.
    """
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return 0.0, []

    score: float = 0.0
    details: list[str] = []

    for i, token in enumerate(tokens):
        is_high = token in _TOXIC_HIGH
        is_medium = token in _TOXIC_MEDIUM

        if not is_high and not is_medium:
            continue

        # ── Negation check (all severities): preceding 3 words ────────
        pre_window = set(tokens[max(0, i - 3):i])
        if pre_window & _NEGATION_WORDS:
            continue  # e.g. "do not suppress", "anti-fraud"

        # ── Context check (MEDIUM only): surrounding ±3 words ─────────
        if is_medium and token in _SAFE_CONTEXT:
            full_window = set(
                tokens[max(0, i - 3):i]
                + tokens[i + 1:min(len(tokens), i + 4)]
            )
            if full_window & _SAFE_CONTEXT[token]:
                continue  # e.g. "kill the process", "substance abuse"

        if is_high:
            score += 3.0
            label = _TOXIC_HIGH[token]
            details.append(f"High-severity: '{token}' ({label})")
        else:
            score += 1.5
            label = _TOXIC_MEDIUM[token]
            details.append(f"'{token}' ({label})")

    return round(score, 2), details


# ---------------------------------------------------------------------------
# Category-Aware Policy Rules
# ---------------------------------------------------------------------------
# When the detector has flagged a specific threat type and the rewriter
# produced a rewrite, the validator verifies that the rewrite actually
# contains the safety-constraint language needed to address that threat.
#
# Each entry: ThreatType → (list of marker patterns — at least one must
# be present, human-readable failure reason).

_REQUIRED_CONSTRAINTS: dict[ThreatType, tuple[list[str], str]] = {
    ThreatType.PII_PHI_EXPOSURE: (
        [r"de-?identif", r"\baggregat", r"non-?identifying", r"\banonym"],
        "Rewrite addressing PHI/PII risk must include "
        "de-identification or aggregation language",
    ),
    ThreatType.OFF_LABEL_PROMOTION: (
        [r"approved (indication|material|messaging|use)", r"\bon[- ]?label\b"],
        "Rewrite addressing off-label risk must reference "
        "approved indications or materials",
    ),
    ThreatType.MISLEADING_CLAIM: (
        [
            r"fair[- ]?balance",
            r"\bsubstantiat",
            r"approved claim",
            r"safety (information|warnings?|language|disclaimers?)",
        ],
        "Rewrite addressing misleading-claim risk must include "
        "fair-balance or substantiation language",
    ),
    ThreatType.SAFETY_DATA_TAMPERING: (
        [
            r"without (removing|filtering|altering|deleting|changing)",
            r"\ball (adverse|safety)",
            r"complete (safety|adverse)",
            r"\bno.{0,10}(filter|remov|discard)",
        ],
        "Rewrite addressing safety-data risk must include "
        "data-integrity preservation language",
    ),
    ThreatType.MEDICAL_ADVICE: (
        [
            r"non-?individuali[sz]ed",
            r"general (information|education)",
            r"consult.{0,20}(healthcare|physician|doctor|HCP|provider)",
            r"does not constitute.{0,20}(medical|clinical) advice",
        ],
        "Rewrite addressing medical-advice risk must include "
        "non-individualised or consult-HCP language",
    ),
}

# Minimum length (in characters) for a rewritten prompt to be considered
# substantive.  Anything shorter is likely an incomplete or broken rewrite.
_MIN_PROMPT_LENGTH: int = 10


def _check_required_constraints(
    candidate: str,
    verdict: DetectorVerdict,
) -> list[str]:
    """Verify that *candidate* contains the safety markers required for
    each threat type flagged by *verdict*."""
    reasons: list[str] = []
    for ttype in verdict.threat_types:
        entry = _REQUIRED_CONSTRAINTS.get(ttype)
        if entry is None:
            continue
        patterns, failure_reason = entry
        if not any(re.search(p, candidate, re.IGNORECASE) for p in patterns):
            reasons.append(f"Policy: {failure_reason}.")
    return reasons


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def validate(
    rewrite: RewriteResult,
    verdict: Optional[DetectorVerdict] = None,
) -> ValidationResult:
    """Deterministic policy-as-code validation of a rewritten prompt.

    Parameters
    ----------
    rewrite:
        The output of the Safe Intent Rewriter (Step 3).
    verdict:
        Optional detector verdict from Step 2.  When provided, enables
        category-aware policy checks that verify the rewrite actually
        addresses the specific threat type(s) detected.  Omitting it
        (the default) is fully backward-compatible — structural,
        blocklist, toxicity, and length checks still run.

    Returns
    -------
    :class:`~guard.schemas.ValidationResult`
        Deterministic pass/fail with human-readable reasons.
    """
    # 1. Structural check: status and prompt text sanity.
    if rewrite.status == RewriteStatus.REWRITTEN and not (rewrite.rewritten_prompt or "").strip():
        return ValidationResult(
            passed=False,
            reasons=["Rewrite marked REWRITTEN but produced no prompt text."],
        )
    if rewrite.status in (RewriteStatus.INVALID, RewriteStatus.NEEDS_CLARIFICATION):
        return ValidationResult(
            passed=False,
            reasons=[f"Rewrite status is {rewrite.status.value}; not executable."],
        )

    candidate = rewrite.rewritten_prompt or ""
    reasons: list[str] = []

    # 2. Keyword blocklist.
    for pattern, reason in _BLOCKLIST:
        if re.search(pattern, candidate, re.IGNORECASE):
            reasons.append(f"Blocked: {reason}.")

    # 3. Toxicity (severity-weighted, context-aware).
    tox_score, tox_details = _toxicity_check(candidate)
    if tox_score >= _TOXICITY_THRESHOLD:
        reasons.append(
            f"Toxicity score {tox_score:.1f} exceeds threshold "
            f"({_TOXICITY_THRESHOLD}): {'; '.join(tox_details)}."
        )

    # 4. Prompt length.
    stripped = candidate.strip()
    if stripped and len(stripped) < _MIN_PROMPT_LENGTH:
        reasons.append(
            f"Rewritten prompt too short ({len(stripped)} chars); "
            f"minimum is {_MIN_PROMPT_LENGTH}."
        )

    # 5. Category-aware required-constraint checks.
    if verdict is not None:
        constraint_reasons = _check_required_constraints(candidate, verdict)
        reasons.extend(constraint_reasons)

    return ValidationResult(passed=not reasons, reasons=reasons)
