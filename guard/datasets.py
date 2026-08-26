"""Dataset ingestion utilities.

Loads a labelled prompt-safety dataset (the hand-labelled seed file, a
synthetic file produced by ``generate_dataset.py``, or any user-supplied CSV),
normalises its columns and labels, reports descriptive statistics, and provides
a reusable evaluation loop that runs a :class:`~guard.PromptGuard` over the rows.

The seed dataset uses a UTF-8 BOM, a ``#`` id column, a free-text rationale
column, and a ``Safe or Unsafe?`` label whose "unsafe" value is spelled
``"Unsafe / Risky"``.  This module hides those quirks behind a small, stable API
so both the CLI and the Streamlit frontend can share one ingestion path.

Public API
----------
``CANONICAL_*``        canonical column names emitted after ingestion
``load_dataset``       read a CSV path or file-like object into a DataFrame
``detect_columns``     locate the prompt / label / rationale columns
``normalize_labels``   collapse label spellings to ``"Safe"`` / ``"Unsafe"``
``ingest_dataset``     load + detect + normalise in one call
``dataset_stats``      descriptive statistics for a normalised DataFrame
``evaluate_dataset``   run a guard over the rows and return metrics + predictions
"""

from __future__ import annotations

from typing import IO, Any, Optional, Union

import pandas as pd

# Canonical column names produced by :func:`ingest_dataset`.  These match the
# seed dataset and ``generate_dataset.py`` so ingested and generated files line
# up without renaming.
CANONICAL_ID = "#"
CANONICAL_PROMPT = "Risky ambiguous prompt"
CANONICAL_RATIONALE = "Why it’s ambiguous (and what could go wrong)"
CANONICAL_LABEL = "Safe or Unsafe?"

# Case-insensitive header aliases used when a CSV does not use the canonical
# names (e.g. a user uploads their own "text","label" file).
_PROMPT_ALIASES = (
    CANONICAL_PROMPT, "prompt", "text", "input", "query", "message",
)
_LABEL_ALIASES = (
    CANONICAL_LABEL, "label", "class", "target", "safe", "verdict",
)
_RATIONALE_ALIASES = (
    CANONICAL_RATIONALE, "rationale", "reason", "why", "explanation", "notes",
)

# Canonical label values.
LABEL_SAFE = "Safe"
LABEL_UNSAFE = "Unsafe"


def load_dataset(source: Union[str, IO[Any]]) -> pd.DataFrame:
    """Read *source* into a DataFrame, tolerating the seed file's UTF-8 BOM.

    *source* may be a filesystem path or any file-like object (e.g. a Streamlit
    ``UploadedFile``).  Header whitespace is stripped so " label " matches
    "label".
    """
    df = pd.read_csv(source, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> Optional[str]:
    """Return the first DataFrame column matching *aliases* (case-insensitive)."""
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        hit = lowered.get(alias.lower())
        if hit is not None:
            return hit
    return None


def detect_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """Locate the prompt, label, and rationale columns in *df*.

    Returns a dict with keys ``prompt``, ``label``, and ``rationale``; a value
    is ``None`` when no matching column is found.  The prompt column falls back
    to the first text-like column so ingestion still works for minimally
    labelled files.
    """
    prompt = _find_column(df, _PROMPT_ALIASES)
    label = _find_column(df, _LABEL_ALIASES)
    rationale = _find_column(df, _RATIONALE_ALIASES)

    if prompt is None:
        # Fall back to the first non-label object column.
        for col in df.columns:
            if col != label and df[col].dtype == object:
                prompt = col
                break

    return {"prompt": prompt, "label": label, "rationale": rationale}


def normalize_labels(labels: "pd.Series") -> "pd.Series":
    """Collapse assorted label spellings to ``"Safe"`` / ``"Unsafe"``.

    Anything whose lowercased text starts with "safe" (and is not negated) maps
    to ``"Safe"``; everything else — "Unsafe / Risky", "unsafe", "risky",
    "1", "true" — maps to ``"Unsafe"``.  Blank / NaN values map to ``"Unsafe"``
    (fail-closed), matching the CLI's evaluation convention.
    """
    def _one(value: Any) -> str:
        text = str(value).strip().lower()
        if not text or text in {"nan", "none"}:
            return LABEL_UNSAFE
        # "safe" but not "unsafe"/"not safe".
        if text.startswith("safe") or text in {"0", "false", "ok", "valid"}:
            return LABEL_SAFE
        return LABEL_UNSAFE

    return labels.map(_one)


def ingest_dataset(source: Union[str, IO[Any]]) -> pd.DataFrame:
    """Load *source* and return a normalised DataFrame.

    The returned frame always has the canonical prompt column and, when a label
    column was found, a canonical label column holding ``"Safe"`` / ``"Unsafe"``
    values.  A rationale column is carried through when present.  Rows with an
    empty prompt are dropped.
    """
    df = load_dataset(source)
    cols = detect_columns(df)
    if cols["prompt"] is None:
        raise ValueError(
            "Could not find a prompt/text column in the dataset. "
            f"Columns present: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out[CANONICAL_PROMPT] = df[cols["prompt"]].astype(str).str.strip()
    if cols["label"] is not None:
        out[CANONICAL_LABEL] = normalize_labels(df[cols["label"]])
    if cols["rationale"] is not None:
        out[CANONICAL_RATIONALE] = df[cols["rationale"]].astype(str)

    out = out[out[CANONICAL_PROMPT].str.len() > 0].reset_index(drop=True)
    out.insert(0, CANONICAL_ID, range(1, len(out) + 1))
    return out


def dataset_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute descriptive statistics for a normalised dataset.

    Returns row count, unique-prompt count, duplicate count, prompt-length
    summary, and (when a label column is present) the label distribution.
    """
    prompts = df[CANONICAL_PROMPT] if CANONICAL_PROMPT in df.columns else pd.Series(dtype=str)
    lengths = prompts.str.len()
    stats: dict[str, Any] = {
        "rows": int(len(df)),
        "unique_prompts": int(prompts.nunique()),
        "duplicate_prompts": int(len(prompts) - prompts.nunique()),
        "avg_prompt_chars": round(float(lengths.mean()), 1) if len(prompts) else 0.0,
        "max_prompt_chars": int(lengths.max()) if len(prompts) else 0,
        "has_labels": CANONICAL_LABEL in df.columns,
    }
    if stats["has_labels"]:
        counts = df[CANONICAL_LABEL].value_counts()
        stats["label_distribution"] = {str(k): int(v) for k, v in counts.items()}
    return stats


def evaluate_dataset(
    guard: Any,
    df: pd.DataFrame,
    limit: Optional[int] = None,
    progress: Optional[Any] = None,
) -> dict[str, Any]:
    """Run *guard* over the labelled rows of *df* and return metrics + predictions.

    Parameters
    ----------
    guard:
        A :class:`~guard.PromptGuard` (anything with a ``check(prompt)`` method
        returning an object with an ``allowed`` bool).
    df:
        A normalised dataset (see :func:`ingest_dataset`) with a label column.
    limit:
        Evaluate only the first *limit* rows (useful for a quick UI preview).
    progress:
        Optional callable ``progress(done, total)`` invoked after each row, so a
        frontend can render a progress bar without this module importing any UI.

    Returns
    -------
    dict
        ``metrics`` (accuracy/precision/recall/f1 + confusion counts) and
        ``predictions`` (a DataFrame with gold label, predicted label, and
        detected category per row).  The positive class is *Unsafe/blocked*.
    """
    if CANONICAL_LABEL not in df.columns:
        raise ValueError("Dataset has no label column; cannot evaluate.")

    rows = df if limit is None else df.head(limit)
    total = len(rows)
    tp = tn = fp = fn = 0
    records: list[dict[str, Any]] = []

    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        prompt = str(row[CANONICAL_PROMPT])
        gold_unsafe = str(row[CANONICAL_LABEL]).strip().lower() != LABEL_SAFE.lower()
        result = guard.check(prompt)
        pred_unsafe = not result.allowed

        if gold_unsafe and pred_unsafe:
            tp += 1
        elif not gold_unsafe and not pred_unsafe:
            tn += 1
        elif not gold_unsafe and pred_unsafe:
            fp += 1
        else:
            fn += 1

        records.append({
            CANONICAL_PROMPT: prompt,
            "gold": LABEL_UNSAFE if gold_unsafe else LABEL_SAFE,
            "predicted": LABEL_UNSAFE if pred_unsafe else LABEL_SAFE,
            "category": result.category.value,
            "correct": gold_unsafe == pred_unsafe,
        })
        if progress is not None:
            progress(i, total)

    n = tp + tn + fp + fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    metrics = {
        "total": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    return {"metrics": metrics, "predictions": pd.DataFrame(records)}
