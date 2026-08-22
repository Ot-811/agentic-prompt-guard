"""Command-line entry point for the Agentic Prompt Guard.

Examples:
    # screen a single prompt (and run it in the sandbox)
    python -m guard.cli check "Draft a follow-up email using patient specifics from the notes." --execute

    # evaluate the guard's safe/unsafe calls against a labeled dataset
    python -m guard.cli eval data/seed_dataset.csv
"""

import argparse
import json
import sys

from .pipeline import PromptGuard


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="llama3", help="Ollama model name (default: llama3).")
    p.add_argument("--no-llm", action="store_true", help="Force heuristic fallback; never call Ollama.")


def cmd_check(args) -> int:
    guard = PromptGuard(model=args.model, use_llm=not args.no_llm)
    prompt = args.prompt or sys.stdin.read().strip()
    result = guard.check(prompt, execute=args.execute)
    print(result.summary())
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0 if result.allowed else 1


def cmd_eval(args) -> int:
    import pandas as pd

    guard = PromptGuard(model=args.model, use_llm=not args.no_llm)
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    prompt_col = "Risky ambiguous prompt"
    label_col = "Safe or Unsafe?"

    tp = tn = fp = fn = 0
    for _, row in df.iterrows():
        gold_unsafe = str(row[label_col]).strip().lower() != "safe"
        result = guard.check(str(row[prompt_col]))
        pred_unsafe = not result.allowed
        if gold_unsafe and pred_unsafe:
            tp += 1
        elif not gold_unsafe and not pred_unsafe:
            tn += 1
        elif not gold_unsafe and pred_unsafe:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    backend = "ollama" if guard.llm_active else "heuristic-fallback"

    print(f"Evaluated {total} prompts (backend: {backend})")
    print(f"  accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")
    print(f"  TP={tp} TN={tn} FP={fp} FN={fn}  (positive class = Unsafe/blocked)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="guard", description="Agentic Prompt Guard.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Screen a single prompt.")
    p_check.add_argument("prompt", nargs="?", help="Prompt text (or read from stdin).")
    p_check.add_argument("--execute", action="store_true", help="Run allowed prompts in the sandbox.")
    _add_common(p_check)
    p_check.set_defaults(func=cmd_check)

    p_eval = sub.add_parser("eval", help="Evaluate against a labeled CSV.")
    p_eval.add_argument("csv", help="CSV with 'Risky ambiguous prompt' and 'Safe or Unsafe?' columns.")
    _add_common(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
