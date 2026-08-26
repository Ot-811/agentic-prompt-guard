# Agentic Prompt Guard

An implementation of the **Agentic Guardrail for Responsible Prompt Engineering**
described in `data/*.pdf` (problem statement) and `data/*.pptx` (pipeline deck).

It is a **router/classifier pipeline** that screens a prompt *before* an agent
acts on it. Safe prompts take a fast path straight to execution; risky or
disguised prompts enter a remediation loop that rewrites and re-validates them.
Only risky prompts pay the heavy cost.

```
User prompt
   │
   ▼
1. Ingestion            normalize, decode hidden payloads, regex signatures,
   (guard/ingestion.py)  similarity vs known-attack corpus  ── vector-DB stand-in
   │
   ▼
2. Threat & Disguise    structured verdict: is_safe / category / threat_types
   Detector             (Ollama when available, else heuristic fallback)
   (guard/detector.py)
   │
   ├── SAFE ─────────────────────────────► 5. Safe Execution Sandbox
   │                                          (guard/sandbox.py) + PII/harm filter
   ▼ RISKY / DISGUISED
3. Safe Intent Rewriter  extract benign intent → safe query, or ask
   (guard/rewriter.py)   clarification, or INVALID
   │
   ▼
4. Policy-as-Code        deterministic: Pydantic schema, keyword blocklist,
   Validator             toxicity check
   (guard/validator.py)
   │
   ▼
   Verification          re-run the detector on the rewrite to confirm the
   (in pipeline.py)      disguised-risk signals are gone
   │
   ▼
   Router logs every decision  (guard/pipeline.py)
```

## Design notes

- **Ollama + fallback.** The Threat Detector, Rewriter, and Sandbox call a
  self-hosted Ollama model via its HTTP API (`guard/llm.py`, standard-library
  only). If no Ollama server is reachable, each stage falls back to a
  deterministic heuristic so the whole pipeline runs and is testable offline.
  The pipeline auto-detects availability at construction time.
- **Deterministic step 4.** The validator uses no LLM, so it can reliably gate
  the non-deterministic rewrite.
- **Stand-ins, clearly marked.** The similarity check (a difflib comparison
  against a small attack corpus) stands in for the deck's Chroma/Pinecone vector
  DB, and the toxicity check is a lexicon stand-in for a real classifier. Both
  are isolated behind clean seams and swappable.

## Install

```bash
pip install -r requirements.txt        # pandas, pydantic, pytest, streamlit
# optional, for the LLM path:  install Ollama and `ollama pull llama3`
```

## Frontend (Streamlit)

An interactive dashboard exposes every stage of the pipeline in its own tab.

```bash
streamlit run app.py
```

| Tab | What it shows |
|-----|---------------|
| 🛡️ **Live Guard** | Full pipeline on a prompt: verdict, threats, rewrite, sandbox, audit log. |
| 🔬 **Preprocessing** | The normalisation cascade peeling back a disguised prompt, pass by pass. |
| 🔎 **Regex / Signatures** | Jailbreak signature hits, decoded payloads, and heuristic threat labels. |
| 🧠 **Embeddings** | Top-k nearest known-attack strings by semantic similarity (bar chart). |
| 📥 **Dataset Ingestion** | Upload/load a CSV, auto-detect columns, view label distribution + stats. |
| 🧪 **Generate Dataset** | Synthesise labelled prompts (rows/seed/dedup) and download the CSV. |
| 📊 **Evaluate** | Score the guard against an ingested labelled dataset (accuracy/P/R/F1). |

The sidebar toggles the LLM (Ollama) vs. heuristic backend and reports which
similarity backend (sentence-transformers or the difflib fallback) is active.

## Usage

Run from the `agentic_prompt_guard/` directory so the `guard` package is importable.

```bash
# screen a single prompt (uses Ollama if running, else heuristics)
python -m guard.cli check "Segment engagement by likelihood to start therapy and create a targeting list."

# force the offline heuristic backend, and run allowed prompts in the sandbox
python -m guard.cli check "Generate a de-identified, aggregated engagement report." --execute --no-llm

# evaluate the guard's safe/unsafe calls against a labeled CSV
python -m guard.cli eval data/seed_dataset.csv --no-llm
```

Programmatic use:

```python
from guard import PromptGuard

guard = PromptGuard(use_llm=True)          # falls back automatically if Ollama is down
result = guard.check("Write a catchy social post for the new biologic.", execute=True)
print(result.summary())                     # [BLOCKED] Risky via blocked
print(result.audit_log)                     # full routing trace
```

Every `GuardResult` carries the classification, the detector rationale, the
rewrite/clarification, the validation and verification verdicts, the sandbox
output, and an `audit_log` — satisfying the PDF's explainability and
auditability requirements.

## Tests

```bash
pytest -q      # 11 tests, offline (heuristic backend)
```

## Dataset generator

`generate_dataset.py` produces labeled Safe/Unsafe pharma prompts to train and
evaluate the detector.

```bash
python generate_dataset.py                       # 100k rows -> generated_pharma_dataset_100000.csv
python generate_dataset.py --rows 1000 --seed 7
python generate_dataset.py --dedup               # unique prompts only
```

| Flag      | Default                              | Description                          |
|-----------|--------------------------------------|--------------------------------------|
| `--rows`  | `100000`                             | Number of rows (must be ≥ 1).        |
| `--seed`  | `42`                                 | Random seed for reproducibility.     |
| `--dedup` | off                                  | Emit only unique prompt strings.     |
| `--out`   | `generated_pharma_dataset_<rows>.csv`| Output path.                         |

Output columns match `data/seed_dataset.csv` exactly (`#`, `Risky ambiguous
prompt`, `Why it's ambiguous (and what could go wrong)`, `Safe or Unsafe?`) so
the two files concatenate for training. Labels are balanced ~50/50.

> **Training caveat:** the generator fills a fixed template set, so 100k rows
> contain only ~2,200 unique prompts. Split at the prompt/template level (not
> row level) to avoid train/test leakage. Expand the variable pools or templates
> in `generate_dataset.py` for more diversity.

## Layout

```
agentic_prompt_guard/
├── guard/
│   ├── llm.py           # Ollama HTTP client + availability check
│   ├── ingestion.py     # step 1
│   ├── detector.py      # step 2
│   ├── rewriter.py      # step 3
│   ├── validator.py     # step 4
│   ├── sandbox.py        # step 5
│   ├── pipeline.py       # step 6 router + verification
│   ├── schemas.py        # Pydantic contract / classification enums
│   ├── datasets.py       # dataset ingestion: load, column-detect, stats, evaluate
│   └── cli.py            # `python -m guard.cli check|eval`
├── app.py                # Streamlit frontend (all stages, one tab each)
├── tests/test_pipeline.py
├── generate_dataset.py
├── data/                 # seed dataset + source PDF & PPTX
└── generated_pharma_dataset_100000.csv
```
