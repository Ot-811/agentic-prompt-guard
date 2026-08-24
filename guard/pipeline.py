"""Step 6 — Router / pipeline orchestrator.

Ties the six pipeline stages together:
  - Safe prompts take a fast path (rewrite and validation skipped).
  - Risky prompts enter the remediation loop (rewrite → validate → verify)
    before they can proceed.
  - Every routing decision is appended to ``audit_log`` for traceability.
  - The verification step re-runs the detector on the rewritten prompt to
    confirm disguised-risk signals are gone.

``_LRUCache``
    Thread-safe in-memory LRU cache (default 256 slots) keyed on a SHA-256
    prefix of ``(execute, prompt)``.  Repeated identical prompts — common
    in agentic systems with templated requests — are served at ~0 ms.

``check_async()``
    asyncio wrapper via ``run_in_executor`` for use from FastAPI / aiohttp
    without blocking the event loop.  Results share the same LRU cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections import OrderedDict
from typing import Optional

from .detector import ThreatDetector
from .ingestion import ingest
from .llm import OllamaClient
from .rewriter import SafeRewriter
from .sandbox import SafeSandbox
from .schemas import Category, GuardResult, RewriteStatus, ThreatType
from .validator import validate


# ---------------------------------------------------------------------------
# Thread-safe LRU cache
# ---------------------------------------------------------------------------


class _LRUCache:
    """Simple thread-safe LRU cache backed by an ``OrderedDict``.

    Parameters
    ----------
    maxsize:
        Maximum number of entries to keep in the cache.  When the limit is
        reached, the least-recently-used entry is evicted.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._cache: OrderedDict[str, GuardResult] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[GuardResult]:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: GuardResult) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)  # evict least-recently-used

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# ---------------------------------------------------------------------------
# PromptGuard
# ---------------------------------------------------------------------------


class PromptGuard:
    """The main entry point for the agentic prompt guard.

    Parameters
    ----------
    model:
        Ollama model name to use for LLM-based detection/rewriting.
    use_llm:
        Set to ``False`` to force the deterministic heuristic path (useful
        for testing and offline environments).
    host:
        Ollama HTTP server address.
    cache_size:
        Maximum number of prompt results to keep in the in-memory LRU cache.
        Set to ``0`` to disable caching.
    """

    def __init__(
        self,
        model: str = "llama3",
        use_llm: bool = True,
        host: str = "http://localhost:11434",
        cache_size: int = 256,
    ) -> None:
        llm: Optional[OllamaClient] = None
        if use_llm:
            candidate = OllamaClient(model=model, host=host)
            if candidate.available():
                llm = candidate
        self.llm_active = llm is not None
        self.detector = ThreatDetector(llm)
        self.rewriter = SafeRewriter(llm)
        self.sandbox = SafeSandbox(llm)
        self._cache = _LRUCache(maxsize=cache_size) if cache_size > 0 else None

    # ── Public API ───────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(prompt: str, execute: bool) -> str:
        return hashlib.sha256(f"{execute}:{prompt}".encode()).hexdigest()[:24]

    def check(self, prompt: str, execute: bool = False) -> GuardResult:
        """Screen *prompt* synchronously.

        Parameters
        ----------
        prompt:
            The user prompt to evaluate.
        execute:
            When ``True``, run the validated prompt through the safe execution
            sandbox and include the response in the returned :class:`GuardResult`.

        Returns
        -------
        :class:`~guard.schemas.GuardResult`
            Full decision record including category, allowed/blocked verdict,
            rewrite (if any), and an append-only ``audit_log``.
        """
        if self._cache is not None:
            key = self._cache_key(prompt, execute)
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        result = self._check_impl(prompt, execute)

        if self._cache is not None:
            self._cache.put(key, result)

        return result

    async def check_async(self, prompt: str, execute: bool = False) -> GuardResult:
        """Asyncio-compatible screen — runs in the default thread-pool executor.

        Safe to ``await`` from FastAPI, aiohttp, or any ``asyncio`` service
        without blocking the event loop.  Results are served from / stored to
        the same LRU cache as :meth:`check`.

        Usage example::

            guard = PromptGuard()
            result = await guard.check_async("Write a compliant report...")
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.check, prompt, execute)

    # ── Internal pipeline implementation ─────────────────────────────────

    def _check_impl(self, prompt: str, execute: bool) -> GuardResult:
        log: list[str] = []
        backend = "ollama" if self.llm_active else "heuristic-fallback"
        log.append(f"backend={backend}")

        # Step 1: ingestion.
        ing = ingest(prompt)
        log.append(
            f"ingest: signatures={len(ing.signature_hits)} "
            f"decoded={len(ing.decoded_payloads)} "
            f"similarity={ing.similarity:.3f} "
            f"homoglyph={ing.homoglyph_detected} "
            f"leetspeak={ing.leetspeak_detected} "
            f"whitespace_injection={ing.whitespace_injection_detected} "
            f"-> flagged={ing.flagged}"
        )

        # Step 2: detection.
        verdict = self.detector.detect(prompt, ing)
        log.append(
            f"detect: is_safe={verdict.is_safe} "
            f"category={verdict.category.value} "
            f"threats={[t.value for t in verdict.threat_types]} "
            f"confidence={verdict.confidence:.3f}"
        )

        # Fast path: safe prompts skip steps 3 & 4 entirely.
        if verdict.is_safe:
            log.append("route=fast_path (skip rewrite & validation)")
            result = GuardResult(
                prompt=prompt,
                allowed=True,
                category=verdict.category,
                path="fast_path",
                detector=verdict,
                effective_prompt=ing.normalized,
                audit_log=log,
            )
            return self._maybe_execute(result, execute)

        # Remediation loop — step 3: rewrite.
        log.append("route=remediation")
        rewrite = self.rewriter.rewrite(ing.normalized, verdict)
        log.append(f"rewrite: status={rewrite.status.value}")

        if rewrite.status in (RewriteStatus.INVALID, RewriteStatus.NEEDS_CLARIFICATION):
            log.append("blocked: no executable rewrite")
            return GuardResult(
                prompt=prompt,
                allowed=False,
                category=Category.RISKY,
                path="blocked",
                detector=verdict,
                rewrite=rewrite,
                audit_log=log,
            )

        # Step 4: deterministic validation.
        validation = validate(rewrite)
        log.append(f"validate: passed={validation.passed} reasons={validation.reasons}")
        if not validation.passed:
            log.append("blocked: validation failed")
            return GuardResult(
                prompt=prompt,
                allowed=False,
                category=Category.RISKY,
                path="blocked",
                detector=verdict,
                rewrite=rewrite,
                validation=validation,
                audit_log=log,
            )

        # Verification step: re-run the detector on the rewrite.
        # A rewrite that still carries risk signals is blocked even if it
        # passed the deterministic validator.
        verification = self.detector.detect(
            rewrite.rewritten_prompt, ingest(rewrite.rewritten_prompt)
        )
        log.append(
            f"verify: is_safe={verification.is_safe} "
            f"threats={[t.value for t in verification.threat_types]}"
        )
        if not verification.is_safe:
            log.append("blocked: rewrite still carries risk signals")
            return GuardResult(
                prompt=prompt,
                allowed=False,
                category=Category.RISKY,
                path="blocked",
                detector=verdict,
                rewrite=rewrite,
                validation=validation,
                verification=verification,
                audit_log=log,
            )

        log.append("allowed: rewrite passed validation and verification")
        result = GuardResult(
            prompt=prompt,
            allowed=True,
            category=Category.RESPONSIBLE,
            path="remediation",
            detector=verdict,
            rewrite=rewrite,
            validation=validation,
            verification=verification,
            effective_prompt=rewrite.rewritten_prompt,
            audit_log=log,
        )
        return self._maybe_execute(result, execute)

    def _maybe_execute(self, result: GuardResult, execute: bool) -> GuardResult:
        if execute and result.allowed and result.effective_prompt:
            result.sandbox = self.sandbox.execute(result.effective_prompt)
            result.audit_log.append(
                f"sandbox: output_filtered={result.sandbox.output_filtered} "
                f"items={result.sandbox.filtered_items}"
            )
        return result
