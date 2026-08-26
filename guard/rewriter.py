"""Step 3 — Safe Intent Extractor & Rewriter.

For risky prompts, extract the benign intent and reframe it as a safe, neutral
query rather than rejecting outright (deck: "Rewrite, Don't Reject"). When the
intent cannot be recovered safely, either ask targeted clarification questions
or return INVALID.

Uses the LLM when available; otherwise falls back to deterministic,
per-threat heuristic rules.
"""

from typing import Optional

from .llm import OllamaClient
from .schemas import DetectorVerdict, RewriteResult, RewriteStatus, ThreatType


_SYSTEM = """You rewrite risky prompts for a regulated pharma agent into safe,
compliant versions.

Extract the user's benign underlying intent and reframe it so it:
- uses only aggregate/de-identified data,
- references only approved materials,
- stays educational and non-promotional,
- adds required fair-balance/safety language,
- never gives medical or dosing advice,
- never exposes patient-level information,
- never removes, filters, or alters safety records.

The rewritten prompt must NOT preserve unsafe instructions merely by adding
a disclaimer. Transform the request into a genuinely safe version.

If the intent can be safely recovered, return status "rewritten".
If the intent is legitimate but required information is missing, return
"needs_clarification" and ask targeted questions.
If there is no recoverable benign intent, return "invalid".

Respond ONLY with JSON:
{
  "status": "rewritten"|"needs_clarification"|"invalid",
  "extracted_intent": str,
  "rewritten_prompt": str|null,
  "clarification_questions": [str]
}
"""


# Threat -> (constraint appended to a safe rewrite,
#            clarifying question if information is genuinely required).
_CONSTRAINTS: dict[ThreatType, tuple[str, str]] = {
    ThreatType.PII_PHI_EXPOSURE: (
        "using only de-identified, aggregate data and no patient-level details",
        "Which non-identifying, consented fields may be used?",
    ),
    ThreatType.OFF_LABEL_PROMOTION: (
        "restricted to approved indications and approved materials only",
        "What is the approved indication and jurisdiction for this content?",
    ),
    ThreatType.MISLEADING_CLAIM: (
        "using only substantiated, approved claims with required fair-balance and safety language",
        "Which approved claims and evidence sources should be used?",
    ),
    ThreatType.UNSAFE_TARGETING: (
        "using only non-sensitive, consented attributes and stating all selection criteria explicitly",
        "What compliant selection criteria and governance rules apply to this targeting?",
    ),
    ThreatType.MEDICAL_ADVICE: (
        "as general, non-individualized information that does not constitute medical or dosing advice",
        "Should this be limited to directing the reader to their healthcare professional?",
    ),
    ThreatType.SAFETY_DATA_TAMPERING: (
        "without removing, filtering, or altering any adverse-event or safety records",
        "What is the compliant, auditable reason for filtering these records?",
    ),
}


# For some threats, simply appending a constraint to the original prompt is
# not sufficient because the original request itself may ask for unsafe action.
# These templates transform the task into a genuinely safer form.
_REWRITE_TEMPLATES: dict[ThreatType, str] = {
    ThreatType.MEDICAL_ADVICE: (
        "Provide general educational information about the relevant medical "
        "topic, including factors and considerations clinicians may evaluate, "
        "without recommending a diagnosis, treatment, or dosage for an individual."
    ),
    ThreatType.PII_PHI_EXPOSURE: (
        "Provide the requested analysis using only de-identified, aggregate "
        "data and exclude patient-level identifying information."
    ),
    ThreatType.OFF_LABEL_PROMOTION: (
        "Provide neutral, educational information limited to approved "
        "indications and approved materials, without promoting off-label use."
    ),
    ThreatType.MISLEADING_CLAIM: (
        "Provide only substantiated, approved claims supported by appropriate "
        "evidence, with required fair-balance and safety information."
    ),
    ThreatType.UNSAFE_TARGETING: (
        "Describe the requested analysis using only non-sensitive, consented "
        "attributes and explicitly state the applicable selection criteria "
        "and governance requirements."
    ),
    ThreatType.SAFETY_DATA_TAMPERING: (
        "Analyze the safety data without removing, filtering, or altering "
        "adverse-event or other safety records, while preserving an auditable "
        "record of the analysis."
    ),
}


class SafeRewriter:
    """Extract benign intent from risky prompts and produce safe rewrites."""

    def __init__(self, llm: Optional[OllamaClient] = None):
        self.llm = llm

    def rewrite(
        self,
        prompt: str,
        verdict: DetectorVerdict,
    ) -> RewriteResult:
        """Rewrite a prompt according to the detector verdict."""

        # Safe prompts do not need remediation.
        if verdict.is_safe:
            return RewriteResult(
                status=RewriteStatus.NOT_NEEDED,
                rewritten_prompt=prompt,
                extracted_intent="Prompt already safe.",
            )

        # These threats do not have a benign task that should be preserved.
        blocking = {
            ThreatType.JAILBREAK,
            ThreatType.ENCODED_PAYLOAD,
            ThreatType.ROLEPLAY_EXPLOIT,
            ThreatType.SPLIT_REQUEST,
        }

        if blocking.intersection(verdict.threat_types):
            return RewriteResult(
                status=RewriteStatus.INVALID,
                extracted_intent=(
                    "Prompt-injection or jailbreak attempt; "
                    "no benign intent can be safely preserved."
                ),
            )

        # Try the LLM first when available.
        if self.llm is not None:
            result = self._rewrite_llm(prompt, verdict)
            if result is not None:
                return result

        # Deterministic fallback when the LLM is unavailable or fails.
        return self._rewrite_heuristic(prompt, verdict)

    def _rewrite_llm(
        self,
        prompt: str,
        verdict: DetectorVerdict,
    ) -> Optional[RewriteResult]:
        """Use Ollama to extract intent and generate a safe rewrite."""

        threats = ", ".join(
            threat.value for threat in verdict.threat_types
        )

        ambiguity = ", ".join(verdict.ambiguity_flags) or "none"

        llm_prompt = (
            f"Detected threat types: {threats}\n\n"
            f"Ambiguity flags: {ambiguity}\n\n"
            f"User prompt:\n{prompt}"
        )

        raw = self.llm.generate_json(_SYSTEM, llm_prompt)

        if not raw:
            return None

        try:
            result = RewriteResult(**raw)

            # Basic semantic consistency checks.
            if result.status == RewriteStatus.REWRITTEN:
                if not result.rewritten_prompt:
                    return None

                # A successful rewrite should not simultaneously require
                # clarification.
                result.clarification_questions = []

            elif result.status == RewriteStatus.NEEDS_CLARIFICATION:
                if not result.clarification_questions:
                    return None

                # No executable rewrite should be returned while clarification
                # is still required.
                result.rewritten_prompt = None

            elif result.status == RewriteStatus.INVALID:
                result.rewritten_prompt = None
                result.clarification_questions = []

            return result

        except Exception:
            # Invalid LLM output -> deterministic fallback.
            return None

    def _rewrite_heuristic(
        self,
        prompt: str,
        verdict: DetectorVerdict,
    ) -> RewriteResult:
        """Deterministic fallback for offline operation."""

        constraints: list[str] = []
        questions: list[str] = []

        for threat_type in verdict.threat_types:
            pair = _CONSTRAINTS.get(threat_type)

            if pair:
                constraint, question = pair
                constraints.append(constraint)
                questions.append(question)

        # Pure ambiguity is best resolved by asking rather than guessing.
        if (
            not constraints
            and (
                ThreatType.AMBIGUOUS in verdict.threat_types
                or verdict.ambiguity_flags
            )
        ):
            if verdict.ambiguity_flags:
                clarification_questions = [
                    f"Please clarify: {flag}."
                    for flag in verdict.ambiguity_flags
                ]
            else:
                clarification_questions = [
                    "Please specify the exact scope, data sources, "
                    "and constraints for this request."
                ]

            return RewriteResult(
                status=RewriteStatus.NEEDS_CLARIFICATION,
                extracted_intent="Underlying task is unclear.",
                clarification_questions=list(
                    dict.fromkeys(clarification_questions)
                ),
            )

        # No known threat has a safe remediation rule.
        if not constraints:
            return RewriteResult(
                status=RewriteStatus.INVALID,
                extracted_intent=(
                    "No safe reframing is available for this request."
                ),
            )

        base = prompt.rstrip(". ")

        # If a threat has a dedicated transformation, use it instead of
        # simply appending a disclaimer to potentially unsafe instructions.
        primary_threat = next(
            (
                threat_type
                for threat_type in verdict.threat_types
                if threat_type in _REWRITE_TEMPLATES
            ),
            None,
        )

        if primary_threat is not None:
            rewritten = (
                f"{_REWRITE_TEMPLATES[primary_threat]} "
                f"Original task context: {base}."
            )
        else:
            # Generic fallback for threats where appending constraints is
            # sufficient to preserve the benign task.
            unique_constraints = list(dict.fromkeys(constraints))
            rewritten = (
                f"{base}, "
                f"{', '.join(unique_constraints)}."
            )

        return RewriteResult(
            status=RewriteStatus.REWRITTEN,
            extracted_intent=(
                "Benign task preserved and reframed with "
                "appropriate compliance constraints."
            ),
            rewritten_prompt=rewritten,
        )