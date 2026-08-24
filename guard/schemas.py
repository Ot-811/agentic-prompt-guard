"""Pydantic schemas — the structured contract passed between pipeline stages.

The classification categories mirror the PDF's expected output
(Good/Valid / Responsible / Risky); threat types mirror the disguise classes
called out on slide 2 of the deck (jailbreak, encoded payload, roleplay,
split-request) plus the pharma-domain risks from the seed dataset.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Category(str, Enum):
    """Top-level classification from the PDF's expected outcome."""

    GOOD_VALID = "Good/Valid"      # clearly safe, no changes needed
    RESPONSIBLE = "Responsible"    # safe but constrained / minor ambiguity noted
    RISKY = "Risky"                # ambiguous or disguised risk; needs remediation


class ThreatType(str, Enum):
    NONE = "none"
    JAILBREAK = "jailbreak"
    ENCODED_PAYLOAD = "encoded_payload"
    ROLEPLAY_EXPLOIT = "roleplay_exploit"
    SPLIT_REQUEST = "split_request"
    PII_PHI_EXPOSURE = "pii_phi_exposure"
    OFF_LABEL_PROMOTION = "off_label_promotion"
    MISLEADING_CLAIM = "misleading_claim"
    UNSAFE_TARGETING = "unsafe_targeting"
    MEDICAL_ADVICE = "medical_advice"
    SAFETY_DATA_TAMPERING = "safety_data_tampering"
    AMBIGUOUS = "ambiguous"
    HYPOTHETICAL_BYPASS = "hypothetical_bypass"   # fictional/counterfactual framing to bypass rules
    PROMPT_OVERLOAD = "prompt_overload"           # harmful payload buried in a long prompt
    OTHER = "other"


class RewriteStatus(str, Enum):
    NOT_NEEDED = "not_needed"                # prompt was already safe
    REWRITTEN = "rewritten"                  # benign intent extracted and reframed
    NEEDS_CLARIFICATION = "needs_clarification"  # cannot rewrite without user input
    INVALID = "invalid"                      # no benign intent recoverable -> block


class DetectorVerdict(BaseModel):
    """Output of the Threat & Disguise Detector (step 2)."""

    is_safe: bool
    category: Category
    threat_types: list[ThreatType] = Field(default_factory=list)
    ambiguity_flags: list[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RewriteResult(BaseModel):
    """Output of the Safe Intent Extractor & Rewriter (step 3)."""

    status: RewriteStatus
    extracted_intent: str = ""
    rewritten_prompt: Optional[str] = None
    clarification_questions: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Output of the deterministic Policy-as-Code Validator (step 4)."""

    passed: bool
    reasons: list[str] = Field(default_factory=list)


class SandboxResult(BaseModel):
    """Output of the Safe Execution Sandbox (step 5)."""

    response: str
    output_filtered: bool = False
    filtered_items: list[str] = Field(default_factory=list)


class GuardResult(BaseModel):
    """The full decision returned to the caller, plus an auditable trace."""

    prompt: str
    allowed: bool
    category: Category
    path: str  # "fast_path" | "remediation" | "blocked"
    detector: DetectorVerdict
    rewrite: Optional[RewriteResult] = None
    validation: Optional[ValidationResult] = None
    verification: Optional[DetectorVerdict] = None
    sandbox: Optional[SandboxResult] = None
    effective_prompt: Optional[str] = None  # what would actually be executed
    audit_log: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        verdict = "ALLOWED" if self.allowed else "BLOCKED"
        return f"[{verdict}] {self.category.value} via {self.path}"
