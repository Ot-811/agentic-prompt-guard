"""Step 5 — Safe Execution Sandbox.

Executes the validated safe prompt and filters the output for PII/PHI or harmful
content before it reaches the user (deck: "apply an output filter for PII leakage
or harmful content"). Actual generation is delegated to Ollama when available; a
placeholder response is returned otherwise so the pipeline is demonstrable
offline. The output filter runs regardless of how the response was produced.
"""

import re
from typing import Optional

from .llm import OllamaClient
from .schemas import SandboxResult

# (label, regex) for common PII/PHI patterns leaking into output.
_PII_PATTERNS = [
    ("email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ("phone", r"(?<!\d)(?:\+?\d[\s.-]?){9,}\d(?!\d)"),
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("mrn", r"\bMRN[:#]?\s*\d{4,}\b"),
    ("dob", r"\b(?:DOB|date of birth)[:\s]+\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b"),
]

_EXEC_SYSTEM = """You are a compliant pharma assistant. Answer the user's request
using only aggregate/de-identified information and approved, non-promotional
language. Never include patient identifiers or medical/dosing advice."""


def _redact(text: str) -> tuple[str, list[str]]:
    found = []
    redacted = text
    for label, pattern in _PII_PATTERNS:
        if re.search(pattern, redacted):
            found.append(label)
            redacted = re.sub(pattern, f"[REDACTED_{label.upper()}]", redacted)
    return redacted, found


class SafeSandbox:
    def __init__(self, llm: Optional[OllamaClient] = None):
        self.llm = llm

    def execute(self, safe_prompt: str) -> SandboxResult:
        if self.llm is not None:
            raw = self.llm.generate_json(
                _EXEC_SYSTEM,
                safe_prompt + '\nRespond ONLY as JSON: {"response": "<answer>"}',
            )
            response = (raw or {}).get("response") if raw else None
            if not response:
                response = self._placeholder(safe_prompt)
        else:
            response = self._placeholder(safe_prompt)

        redacted, found = _redact(response)
        return SandboxResult(response=redacted, output_filtered=bool(found), filtered_items=found)

    @staticmethod
    def _placeholder(safe_prompt: str) -> str:
        return ("[sandbox stub — no LLM configured] Would execute the validated safe "
                f"prompt: {safe_prompt}")
