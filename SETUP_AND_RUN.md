# Setup & Run Guide — Agentic Prompt Guard

This guide takes a fresh machine (Linux, macOS, or Windows/WSL) from the zip file
to a running, tested pipeline. The project runs **fully offline** by default; a
local LLM (Ollama) is optional.

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.9+** | Check with `python3 --version`. |
| `pip` + `venv` | Ships with standard Python. On Debian/Ubuntu: `sudo apt install python3-venv python3-pip`. |
| ~50 MB disk | For the virtual environment and dependencies. |
| **Ollama** *(optional)* | Only if you want the LLM path instead of heuristics. See §6. |

No internet is needed at runtime — only during setup, to install the three
Python packages (`pandas`, `pydantic`, `pytest`).

---

## 2. Unzip

```bash
unzip agentic_prompt_guard.zip
cd agentic_prompt_guard
```

---

## 3. One-command setup

```bash
bash setup.sh
```

This creates `.venv/`, installs dependencies, and runs the test suite. On
success it prints the commands to try next. If `python3` is not your interpreter
name, override it: `PYTHON=python3.11 bash setup.sh`.

### Manual setup (equivalent, if you prefer)

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

---

## 4. Run the guard

Activate the environment first: `source .venv/bin/activate`.
Run all commands from the `agentic_prompt_guard/` directory.

**Screen a single prompt:**

```bash
python -m guard.cli check "Segment engagement by likelihood to start therapy and create a targeting list." --no-llm
```

Prints a one-line verdict (`[BLOCKED] Risky via blocked`) and the full JSON
result: classification, detector rationale, rewrite/clarification, validation,
verification, and the routing audit log.

**Screen and execute an allowed prompt in the sandbox:**

```bash
python -m guard.cli check "Generate a de-identified, aggregated engagement report." --execute --no-llm
```

**Read a prompt from stdin:**

```bash
echo "Turn this claim into a stronger slide even if evidence is limited." | python -m guard.cli check --no-llm
```

**Evaluate against a labeled dataset:**

```bash
python -m guard.cli eval data/seed_dataset.csv --no-llm
# -> accuracy / precision / recall / f1 and the confusion matrix
```

**Use programmatically:**

```python
from guard import PromptGuard
guard = PromptGuard(use_llm=False)
result = guard.check("Write a catchy social post for the new biologic.", execute=True)
print(result.summary())
print(result.audit_log)
```

---

## 5. Generate training data

```bash
python generate_dataset.py                    # 100k rows (default)
python generate_dataset.py --rows 5000 --seed 7
python generate_dataset.py --dedup            # unique prompts only
```

Output CSV columns match `data/seed_dataset.csv`, so files concatenate for
training. See the training caveat in `README.md` (split at prompt level).

---

## 6. Enable the LLM path (optional)

By default every command falls back to deterministic heuristics, which is why
`--no-llm` is shown above (it just skips the availability probe). To use a real
model:

1. Install Ollama: <https://ollama.com>
2. Pull a model: `ollama pull llama3`
3. Ensure the server is running: `ollama serve` (usually automatic).
4. Drop `--no-llm`:
   ```bash
   python -m guard.cli check "..." --model llama3
   ```

The pipeline probes `http://localhost:11434` at startup and uses the model if
reachable, otherwise it silently falls back to heuristics. Set a remote host in
code via `PromptGuard(host="http://my-ollama-host:11434")`.

---

## 7. Run the tests

```bash
pytest -q          # 11 tests, offline
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named guard` | Run from the `agentic_prompt_guard/` directory (where `guard/` lives). |
| `No module named venv` / `ensurepip` | Install OS package `python3-venv`. |
| `command not found: python3` | Use `python` or set `PYTHON=...` for `setup.sh`. |
| Guard always uses heuristics despite Ollama | Confirm `curl http://localhost:11434/api/tags` returns 200 and you dropped `--no-llm`. |
| Slow first LLM call | The model loads on first request; subsequent calls are faster. |

---

## 9. What's in the box

```
agentic_prompt_guard/
├── setup.sh                 # one-command bootstrap (this guide, §3)
├── requirements.txt
├── README.md                # architecture + design notes
├── SETUP_AND_RUN.md         # this file
├── PROJECT_REPORT.md        # full project report
├── guard/                   # the pipeline package (see README)
├── tests/                   # pytest suite
├── generate_dataset.py      # synthetic-data generator
├── data/                    # seed dataset + source PDF & PPTX
└── generated_pharma_dataset_100000.csv
```
