"""Agentic Prompt Guard.

A router/classifier pipeline that screens a prompt before an agent executes it:
ingestion -> threat & disguise detection -> safe rewrite -> policy-as-code
validation -> verification -> safe execution sandbox. Safe prompts take a fast
path; risky or disguised prompts enter a remediation loop.

Implements the design in the project deck (Agentic_Guardrail_Optimized_Pipeline)
and the problem statement PDF.
"""

from .pipeline import PromptGuard
from .schemas import (
    Category,
    DetectorVerdict,
    GuardResult,
    RewriteResult,
    RewriteStatus,
    ThreatType,
    ValidationResult,
)

__all__ = [
    "PromptGuard",
    "Category",
    "ThreatType",
    "RewriteStatus",
    "DetectorVerdict",
    "RewriteResult",
    "ValidationResult",
    "GuardResult",
]
