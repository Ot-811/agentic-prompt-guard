"""Step 6 — Router logic.

Ties the stages together: safe prompts skip straight to the sandbox (fast path);
risky or disguised prompts enter the remediation loop (rewrite -> validate ->
verify) before they may execute. Every routing decision is logged for
auditability. The verification step re-runs the detector on the rewritten prompt
to confirm the disguised-risk signals are gone (PDF: "Verification step").
"""

from typing import Optional

from .detector import ThreatDetector
from .ingestion import ingest
from .llm import OllamaClient
from .rewriter import SafeRewriter
from .sandbox import SafeSandbox
from .schemas import Category, GuardResult, RewriteStatus, ThreatType
from .validator import validate


class PromptGuard:
    def __init__(self, model: str = "llama3", use_llm: bool = True, host: str = "http://localhost:11434"):
        llm: Optional[OllamaClient] = None
        if use_llm:
            candidate = OllamaClient(model=model, host=host)
            if candidate.available():
                llm = candidate
        self.llm_active = llm is not None
        self.detector = ThreatDetector(llm)
        self.rewriter = SafeRewriter(llm)
        self.sandbox = SafeSandbox(llm)

    def check(self, prompt: str, execute: bool = False) -> GuardResult:
        """Screen a prompt. If execute=True, run allowed prompts in the sandbox."""
        log: list[str] = []
        backend = "ollama" if self.llm_active else "heuristic-fallback"
        log.append(f"backend={backend}")

        # Step 1: ingestion.
        ing = ingest(prompt)
        log.append(f"ingest: signatures={ing.signature_hits} decoded={len(ing.decoded_payloads)} "
                   f"similarity={ing.similarity} -> flagged={ing.flagged}")

        # Step 2: detection.
        verdict = self.detector.detect(prompt, ing)
        log.append(f"detect: is_safe={verdict.is_safe} category={verdict.category.value} "
                   f"threats={[t.value for t in verdict.threat_types]}")

        # Fast path: safe prompts skip steps 3 & 4 entirely.
        if verdict.is_safe:
            log.append("route=fast_path (skip rewrite & validation)")
            result = GuardResult(prompt=prompt, allowed=True, category=verdict.category,
                                 path="fast_path", detector=verdict,
                                 effective_prompt=ing.normalized, audit_log=log)
            return self._maybe_execute(result, execute)

        # Remediation loop — step 3: rewrite.
        log.append("route=remediation")
        rewrite = self.rewriter.rewrite(ing.normalized, verdict)
        log.append(f"rewrite: status={rewrite.status.value}")

        if rewrite.status in (RewriteStatus.INVALID, RewriteStatus.NEEDS_CLARIFICATION):
            log.append("blocked: no executable rewrite")
            return GuardResult(prompt=prompt, allowed=False, category=Category.RISKY,
                               path="blocked", detector=verdict, rewrite=rewrite, audit_log=log)

        # Step 4: deterministic validation.
        validation = validate(rewrite)
        log.append(f"validate: passed={validation.passed} reasons={validation.reasons}")
        if not validation.passed:
            log.append("blocked: validation failed")
            return GuardResult(prompt=prompt, allowed=False, category=Category.RISKY,
                               path="blocked", detector=verdict, rewrite=rewrite,
                               validation=validation, audit_log=log)

        # Verification step: re-check the rewrite carries no residual risk.
        verification = self.detector.detect(rewrite.rewritten_prompt, ingest(rewrite.rewritten_prompt))
        log.append(f"verify: is_safe={verification.is_safe} "
                   f"threats={[t.value for t in verification.threat_types]}")
        if not verification.is_safe:
            log.append("blocked: rewrite still carries risk signals")
            return GuardResult(prompt=prompt, allowed=False, category=Category.RISKY,
                               path="blocked", detector=verdict, rewrite=rewrite,
                               validation=validation, verification=verification, audit_log=log)

        log.append("allowed: rewrite passed validation and verification")
        result = GuardResult(prompt=prompt, allowed=True, category=Category.RESPONSIBLE,
                             path="remediation", detector=verdict, rewrite=rewrite,
                             validation=validation, verification=verification,
                             effective_prompt=rewrite.rewritten_prompt, audit_log=log)
        return self._maybe_execute(result, execute)

    def _maybe_execute(self, result: GuardResult, execute: bool) -> GuardResult:
        if execute and result.allowed and result.effective_prompt:
            result.sandbox = self.sandbox.execute(result.effective_prompt)
            result.audit_log.append(
                f"sandbox: output_filtered={result.sandbox.output_filtered} "
                f"items={result.sandbox.filtered_items}")
        return result
