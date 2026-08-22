# Project Report

## Agentic Guardrail for Responsible Prompt Engineering
### Risk & Disguise Detection + Safe Rewriting

---

## 1. Executive summary

Agentic AI systems turn a single user prompt into multi-step actions — tool
calls, retrieval, generation, and customer- or patient-facing outputs. When that
prompt is ambiguous or *disguised-risky* (it looks safe but carries hidden unsafe
intent), the agent can proceed and produce outputs that breach governance,
safety, or compliance rules.

This project delivers an **upstream responsibility gate**: a router/classifier
pipeline that screens every prompt *before* the agent acts. Safe prompts take a
fast path straight to execution; risky or disguised prompts enter a remediation
loop that rewrites and re-validates them. The design follows the "only risky
prompts pay the heavy cost" principle, so the safe majority is not slowed down.

The deliverable is a runnable, tested Python package (`guard/`) plus a synthetic
data generator and a labeled evaluation dataset. It runs fully offline via
deterministic fallbacks and upgrades transparently to a self-hosted LLM (Ollama)
when one is available.

---

## 2. Problem statement

(Condensed from the source problem document.)

There is no reliable, automated mechanism that ensures an agent:

1. Understands prompt intent beyond surface wording.
2. Detects ambiguity that could lead to unsafe or non-compliant interpretations.
3. Identifies disguised risky prompts framed "safely" but effectively harmful.
4. Reduces risk proactively by rewriting or blocking before it continues.

**Why it matters more for agentic AI:** a single risky prompt cascades into
multiple planned steps and tool calls. Without an upstream gate the system
confidently proceeds even when the prompt is misleading.

**Who is affected:** downstream users (patients/customers/colleagues) receiving
non-compliant outputs; compliance/governance teams; developers/ops (incidents,
rework); and prompt authors debugging unsafe outcomes.

---

## 3. Objectives (expected outcomes)

| # | Requirement | Where delivered |
|---|-------------|-----------------|
| 1 | Agentic classification: Good/Valid · Responsible · Risky | `guard/schemas.py` `Category`; produced by `detector.py` |
| 2 | Explainable risk rationale | `DetectorVerdict.rationale` + `threat_types` |
| 3 | Ambiguity detection | `DetectorVerdict.ambiguity_flags` |
| 4 | Remediation: safe rewrite **or** clarification questions | `guard/rewriter.py` |
| 5 | Verification step re-checking the rewrite | `guard/pipeline.py` (re-runs the detector) |

---

## 4. Solution architecture

A six-stage pipeline with a router that branches on the detector's verdict.

```
User prompt
  │
  ▼
[1] Ingestion ─ normalize, decode hidden payloads, regex signatures,
  │             similarity vs known-attack corpus
  ▼
[2] Threat & Disguise Detector ─ structured is_safe / category / threat_types
  │
  ├─ SAFE ───────────────────────────► [5] Safe Execution Sandbox ─► response
  │                                          (+ PII/harm output filter)
  ▼ RISKY / DISGUISED
[3] Safe Intent Rewriter ─ extract benign intent → safe query / clarify / INVALID
  │
  ▼
[4] Policy-as-Code Validator ─ Pydantic schema · blocklist · toxicity (deterministic)
  │
  ▼
  Verification ─ re-run [2] on the rewrite; block if risk signals remain
  │
  ▼
  Router logs every decision → allowed prompt goes to [5]
```

### Stage responsibilities

- **[1] Ingestion (`ingestion.py`).** Unicode NFKC normalization and zero-width
  character stripping (so hidden characters can't smuggle instructions),
  base64/hex/ROT13 decoding to surface encoded payloads, regex signature
  matching for jailbreak markers, and a `difflib` similarity check against a
  known-attack corpus. The similarity check is a dependency-free stand-in for the
  deck's Chroma/Pinecone vector-embedding lookup.
- **[2] Threat & Disguise Detector (`detector.py`).** Emits a structured verdict:
  `is_safe`, `category`, `threat_types`, `ambiguity_flags`, `rationale`,
  `confidence`. Ingestion-level attack signals are treated as decisive. Uses the
  LLM when available; otherwise a rule set tuned to the disguise classes
  (jailbreak, encoded, roleplay, split-request) and pharma risks (PHI/PII,
  off-label, misleading claims, unsafe targeting, safety-data tampering, medical
  advice).
- **[3] Safe Intent Rewriter (`rewriter.py`).** "Rewrite, don't reject": extracts
  the benign intent and reframes it with compliance constraints (de-identified,
  aggregate, approved-materials-only, fair-balance, no dosing advice). If intent
  is unrecoverable it returns INVALID; if under-specified it returns targeted
  clarification questions.
- **[4] Policy-as-Code Validator (`validator.py`).** Deterministic (no LLM), so it
  can reliably gate the non-deterministic rewrite: Pydantic structural check,
  keyword blocklist, and a lexicon-based toxicity score.
- **[5] Safe Execution Sandbox (`sandbox.py`).** Executes the validated prompt
  (via the LLM, or a stub offline) and redacts PII/PHI (email, phone, SSN, MRN,
  DOB) from the output.
- **[6] Router + verification (`pipeline.py`).** Implements the fast-path /
  remediation branching, the verification re-check, and an append-only
  `audit_log` on every result for compliance auditability.

### Key design decisions

- **LLM with graceful fallback.** The reasoning stages call a self-hosted Ollama
  model over its HTTP API using only the Python standard library (`llm.py`). If
  no server responds, each stage falls back to deterministic heuristics. This
  keeps the entire system runnable, testable, and portable with zero model
  infrastructure, while upgrading automatically when a model is present.
- **Determinism where it counts.** Step 4 is intentionally LLM-free so a
  non-deterministic rewrite can never bypass policy.
- **Defense in depth.** The verification step re-runs detection on the rewrite;
  a rewrite that fails to remove a risky phrase is blocked even if it passed
  validation. (Observed in testing: a heuristic rewrite that merely *appended* a
  de-identification constraint without removing "patient specifics from the
  notes" was correctly blocked by verification.)
- **Clearly-marked stand-ins.** The vector-DB similarity and the toxicity check
  are lightweight placeholders behind clean seams, documented as swappable for a
  real vector store (Chroma/Pinecone) and a real toxicity classifier.

---

## 5. Data

- **Seed dataset** (`data/seed_dataset.csv`): 20 hand-labeled pharma/HCP
  engagement prompts (Safe vs Unsafe/Risky) with rationales — the ground truth.
- **Synthetic generator** (`generate_dataset.py`): produces balanced Safe/Unsafe
  prompts from templates and variable pools (audiences, drugs, competitors,
  sources, off-label uses). Configurable rows/seed, optional dedup, and column
  headers identical to the seed set for clean concatenation.
- **Generated dataset** (`generated_pharma_dataset_100000.csv`): 100,000 rows,
  exactly 50/50 balanced, no nulls, no exact-duplicate rows, and no label leakage
  (each unique prompt maps to one label).

**Diversity caveat.** The template approach yields only ~2,200 unique prompts
across 100k rows (each repeated ~45×) from 14 rationale templates. This is
adequate synthetic training data but requires a **prompt- or template-level
train/test split** — a naive row-level split would leak prompts across sets and
inflate metrics. Expanding the variable pools or template set increases
diversity.

---

## 6. Evaluation

Measured with the offline heuristic backend (positive class = Unsafe/blocked),
via `python -m guard.cli eval`:

| Dataset | Accuracy | Precision | Recall | F1 |
|---------|----------|-----------|--------|----|
| Seed (20 curated prompts) | 0.95 | 1.00 | 0.90 | 0.95 |
| Generated (2,000-row sample) | 0.76 | 0.78 | 0.72 | 0.75 |

**Interpretation.** The heuristics score highly on the curated seed set (perfect
precision — no safe prompt wrongly blocked). On the generated set, recall drops
because those templates use phrasings the heuristic rules were not tuned for;
this is expected, since the heuristics are a *fallback* and the LLM detector is
the intended primary classifier. The gap also quantifies the value the LLM path
adds and provides a baseline to beat.

---

## 7. How the objectives are met

- **Classification** — every result carries a `Category` (Good/Valid,
  Responsible, Risky).
- **Explainability** — `rationale` + `threat_types` + full `audit_log` on each
  decision.
- **Ambiguity detection** — `ambiguity_flags`, with pure-ambiguity prompts routed
  to clarification rather than a guessed rewrite.
- **Remediation** — safe rewrite when intent is recoverable; clarification
  questions when under-specified; INVALID/block for jailbreaks.
- **Verification** — the rewrite is re-screened before it can execute.

---

## 8. Limitations

1. **Heuristic coverage.** The fallback rules are keyword/regex-based and will
   miss novel phrasings and non-English prompts. The LLM path mitigates this.
2. **Stand-in components.** Similarity search and toxicity scoring are simplified;
   production use should swap in a real vector DB and toxicity model.
3. **Synthetic-data diversity.** As noted, template repetition limits unique
   prompts; not a substitute for real, diverse traffic.
4. **Sandbox execution** is a stub when no LLM is present; it demonstrates the
   output-filter stage rather than producing real answers.

---

## 9. Future work

- Replace the difflib similarity check with a real embedding store
  (Chroma/Pinecone) of known-attack vectors.
- Integrate a dedicated guardrails framework (NeMo Guardrails / Guardrails AI)
  and a real toxicity classifier in step 4.
- Fine-tune or few-shot the Ollama detector on the labeled dataset and re-run the
  evaluation harness (already provided) to quantify the lift over heuristics.
- Expand the generator's templates/variables and add a prompt-level split utility.
- Add streaming/output-time guarding for long agent responses, and persist audit
  logs to a store for compliance reporting.

---

## 10. Conclusion

The project implements the specified guardrail end to end: an explainable,
auditable, upstream gate that classifies prompts, detects ambiguity and disguised
risk, remediates by rewriting or asking for clarification, and verifies the
result before execution — with a fast path that keeps safe prompts cheap. It is
runnable and tested offline today and designed to upgrade cleanly to a
self-hosted LLM and production-grade components.

---

### Appendix A — Repository layout

```
agentic_prompt_guard/
├── guard/            llm · ingestion · detector · rewriter · validator · sandbox · pipeline · schemas · cli
├── tests/            test_pipeline.py (11 tests)
├── generate_dataset.py
├── data/             seed_dataset.csv + source PDF & PPTX
├── generated_pharma_dataset_100000.csv
├── requirements.txt · setup.sh
└── README.md · SETUP_AND_RUN.md · PROJECT_REPORT.md
```

### Appendix B — Reproducing the results

```bash
bash setup.sh
source .venv/bin/activate
pytest -q
python -m guard.cli eval data/seed_dataset.csv --no-llm
```
