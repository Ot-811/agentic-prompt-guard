"""Step 1 — Intent & Context Ingestion.

Pre-processes the raw prompt before any LLM call or regex scan.  Four passes
are applied in sequence:

1. Invisible-character stripping + NFKC normalisation (zero-width, bidi chars).
2. Homoglyph substitution — Cyrillic/Greek look-alikes replaced with ASCII.
3. Whitespace-injection defuser — collapses 'i g n o r e' into 'ignore'.
4. Leetspeak normalisation — '1gnore @ll' becomes 'ignore all'.

Encoded payloads (base64 / hex / ROT13) are decoded *before* the leetspeak
pass to prevent digit substitutions from corrupting valid tokens.  The decoded
strings are added to the regex scan haystacks.

Semantic similarity is computed via :mod:`guard.semantic` (sentence-transformers
when available, difflib ratio as fallback).
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Normalisation tables
# ---------------------------------------------------------------------------

# Zero-width and bidirectional control characters used to hide payloads.
_INVISIBLE = dict.fromkeys(
    map(
        ord,
        (
            "\u200b"  # zero width space
            "\u200c"  # zero width non-joiner
            "\u200d"  # zero width joiner
            "\u2060"  # word joiner
            "\ufeff"  # BOM / zero width no-break space
            "\u202a"  # left-to-right embedding
            "\u202b"  # right-to-left embedding
            "\u202c"  # pop directional formatting
            "\u202d"  # left-to-right override
            "\u202e"  # right-to-left override
        ),
    ),
    None,
)

# Homoglyph table: maps ord() of a confusable non-ASCII character to its
# ASCII string equivalent.  Covers the most common Cyrillic and Greek
# confusables used in bypass attacks.
_HOMOGLYPH: dict[int, str] = {
    # ── Cyrillic confusables ──────────────────────────────────────────────
    ord("а"): "a",  ord("А"): "A",   # U+0430 / U+0410
    ord("е"): "e",  ord("Е"): "E",   # U+0435 / U+0415
    ord("о"): "o",  ord("О"): "O",   # U+043E / U+041E
    ord("р"): "p",  ord("Р"): "P",   # U+0440 / U+0420
    ord("с"): "c",  ord("С"): "C",   # U+0441 / U+0421
    ord("х"): "x",  ord("Х"): "X",   # U+0445 / U+0425
    ord("у"): "y",  ord("У"): "Y",   # U+0443 / U+0423
    ord("і"): "i",  ord("І"): "I",   # U+0456 / U+0406
    ord("ѕ"): "s",                    # U+0455
    ord("ј"): "j",                    # U+0458
    ord("В"): "B",                    # U+0412
    ord("К"): "K",                    # U+041A
    ord("М"): "M",                    # U+041C
    ord("Н"): "H",                    # U+041D
    ord("Т"): "T",                    # U+0422
    # ── Greek confusables ────────────────────────────────────────────────
    ord("α"): "a",  ord("Α"): "A",
    ord("β"): "b",  ord("Β"): "B",
    ord("ε"): "e",  ord("Ε"): "E",
    ord("η"): "n",  ord("Η"): "H",
    ord("ι"): "i",  ord("Ι"): "I",
    ord("κ"): "k",  ord("Κ"): "K",
    ord("μ"): "u",  ord("Μ"): "M",
    ord("ν"): "v",  ord("Ν"): "N",
    ord("ο"): "o",  ord("Ο"): "O",
    ord("ρ"): "p",  ord("Ρ"): "P",
    ord("τ"): "t",  ord("Τ"): "T",
    ord("υ"): "u",  ord("Υ"): "Y",
    ord("χ"): "x",  ord("Χ"): "X",
    ord("ω"): "w",  ord("Ω"): "W",
}

# Leetspeak substitution map — single characters only (applied char-by-char).
# Note: this is aggressive by design; it may alter legitimate numeric text
# (e.g. "Q1" → "Qi") but the resulting string is only used internally for
# threat detection, not returned to the caller.  The original prompt is
# preserved in IngestionResult.original.
_LEET: dict[str, str] = {
    "1": "i",
    "!": "i",
    "|": "i",
    "0": "o",
    "3": "e",
    "@": "a",
    "4": "a",
    "$": "s",
    "5": "s",
    "7": "t",
    "+": "t",
    "8": "b",
    "6": "g",
}

# ---------------------------------------------------------------------------
# Jailbreak signatures (expanded)
# ---------------------------------------------------------------------------

_JAILBREAK_SIGNATURES: list[str] = [
    # ── Classic prompt-injection phrases ──────────────────────────────────
    r"ignore (all )?(previous|prior|above) (instructions|prompts|rules|directives)",
    r"disregard (the )?(system|previous|prior|all) (prompt|instructions|guidelines|rules)",
    r"\bDAN\b",
    r"do anything now",
    r"developer mode",
    r"pretend (you are|to be)",
    r"act as (an? )?(unfiltered|jailbroken|uncensored|unrestricted)",
    r"you have no (restrictions|rules|guidelines|limits|filters)",
    r"bypass (your )?(safety|content|guardrail|filter|restriction)",
    r"reveal (your )?(system prompt|instructions|training|guidelines)",
    # ── Forget / override instructions ───────────────────────────────────
    r"(forget|disregard|ignore).{0,20}(everything|what|all).{0,20}"
    r"(told|said|instructed|programmed|trained)",
    r"(act|behave|respond).{0,20}as if.{0,30}(no|without|bypass).{0,20}"
    r"(restriction|limit|filter|rule)",
    # ── Hypothetical / roleplay bypass ───────────────────────────────────
    r"(hypothetically|imagine|suppose|what if|let.?s say|pretend).{0,80}"
    r"(fda|regulation|compliance|safety|rules|law|restriction).{0,50}"
    r"(was not|not required|didn.?t exist|do(es)? not exist|gone|ignored|removed|bypass|without|no )",
    r"(hypothetically|imagine|suppose|what if|let.?s say|pretend).{0,80}"
    r"(no|without|ignoring|bypass|not following|waiv\w*).{0,50}"
    r"(fda|regulation|compliance|safety|rules|restriction)",
    # ── Multi-step / split injection ─────────────────────────────────────
    r"(step 1|first).{0,50}(step 2|second|then).{0,50}(step 3|third|finally)",
    r"you (must|will|shall|are required to) (ignore|bypass|disregard|override)",
    # ── Encoding tricks ───────────────────────────────────────────────────
    r"encode.{0,30}(harmful|restricted|blocked|unsafe).{0,30}(base64|hex|rot)",
]

# Similarity threshold used for the difflib fallback path.
# The semantic matcher uses its own threshold defined in guard/semantic.py.
_SIMILARITY_THRESHOLD = 0.72

# ---------------------------------------------------------------------------
# Normalisation functions
# ---------------------------------------------------------------------------


def _strip_invisibles(text: str) -> str:
    """Remove zero-width / bidi control chars and apply NFKC normalisation."""
    return unicodedata.normalize("NFKC", text.translate(_INVISIBLE))


def _apply_homoglyph(text: str) -> str:
    """Replace visually-identical non-ASCII characters with ASCII equivalents."""
    return text.translate(_HOMOGLYPH)


def _defuse_whitespace_injection(text: str) -> str:
    """Collapse whitespace-injected strings like 'i g n o r e' into 'ignore'.

    Detects runs of **3 or more** consecutive single-character tokens separated
    by single spaces and joins them into one word.  Normal prose (words of
    length > 1) is left entirely unchanged.

    Examples
    --------
    >>> _defuse_whitespace_injection("i g n o r e all previous instructions")
    'ignore all previous instructions'
    >>> _defuse_whitespace_injection("Hello World")
    'Hello World'
    """
    tokens = text.split(" ")
    result: list[str] = []
    i = 0
    while i < len(tokens):
        # Check if a run of single-char tokens starts here.
        if (
            len(tokens[i]) == 1
            and i + 2 < len(tokens)
            and len(tokens[i + 1]) == 1
        ):
            run = [tokens[i]]
            j = i + 1
            while j < len(tokens) and len(tokens[j]) == 1:
                run.append(tokens[j])
                j += 1
            if len(run) >= 3:
                result.append("".join(run))
            else:
                result.extend(run)
            i = j
        else:
            result.append(tokens[i])
            i += 1
    return " ".join(result)


def _apply_leetspeak(text: str) -> str:
    """Translate common leet-speak digit/symbol substitutions back to letters."""
    return "".join(_LEET.get(ch, ch) for ch in text)


def _full_normalize(text: str) -> str:
    """Apply all four normalisation passes in sequence."""
    text = _strip_invisibles(text)
    text = _apply_homoglyph(text)
    text = _defuse_whitespace_injection(text)
    text = _apply_leetspeak(text)
    return text.strip()


# ---------------------------------------------------------------------------
# Encoded-payload decoder
# ---------------------------------------------------------------------------


def _try_decode(text: str) -> list[str]:
    """Surface hidden instructions smuggled via base64 / hex / ROT13 encoding."""
    decoded: list[str] = []

    # base64 — look for long token-like runs of base64 alphabet chars.
    for token in re.findall(r"[A-Za-z0-9+/=]{16,}", text):
        try:
            candidate = base64.b64decode(token, validate=True).decode("utf-8", "strict")
            if candidate.isprintable() and re.search(r"[a-zA-Z]{3,}", candidate):
                decoded.append(candidate)
        except (binascii.Error, ValueError, UnicodeDecodeError):
            pass

    # hex — look for sequences of 4+ hex byte pairs.
    for token in re.findall(r"(?:[0-9a-fA-F]{2}\s*){8,}", text):
        try:
            raw = bytes.fromhex(re.sub(r"\s+", "", token))
            candidate = raw.decode("utf-8", "strict")
            if candidate.isprintable() and re.search(r"[a-zA-Z]{3,}", candidate):
                decoded.append(candidate)
        except (ValueError, UnicodeDecodeError):
            pass

    # ROT13 — only flag if it surfaces a known jailbreak phrase.
    rot = codecs.encode(text, "rot_13")
    if re.search(r"ignore .*instructions|developer mode", rot, re.IGNORECASE):
        decoded.append(rot)

    return decoded


# ---------------------------------------------------------------------------
# IngestionResult
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Structured output of the ingestion stage (Step 1).

    Attributes
    ----------
    original:
        The raw, unmodified prompt as received.
    normalized:
        Fully normalised text (invisible chars removed, homoglyphs replaced,
        whitespace injections collapsed, leetspeak translated).  This is what
        all downstream stages operate on.
    signature_hits:
        Jailbreak regex patterns that matched the normalised text or decoded
        payloads.
    decoded_payloads:
        Strings extracted from base64 / hex / ROT13 encodings found in the
        prompt.
    nearest_attack:
        The closest known-attack string in the corpus by similarity score.
    similarity:
        Cosine similarity score (semantic) or difflib ratio (fallback).
    homoglyph_detected:
        True if any homoglyph substitution was performed during normalisation.
    leetspeak_detected:
        True if any leetspeak substitution was performed.
    whitespace_injection_detected:
        True if whitespace-injected single-char runs were collapsed.
    """

    original: str
    normalized: str
    signature_hits: list[str] = field(default_factory=list)
    decoded_payloads: list[str] = field(default_factory=list)
    nearest_attack: str = ""
    similarity: float = 0.0
    homoglyph_detected: bool = False
    leetspeak_detected: bool = False
    whitespace_injection_detected: bool = False

    @property
    def flagged(self) -> bool:
        """True when any decisive pre-detection attack signal is present."""
        return (
            bool(self.signature_hits or self.decoded_payloads)
            or self.similarity >= _SIMILARITY_THRESHOLD
        )


# ---------------------------------------------------------------------------
# Introspection helper (used by the frontend to visualise the cascade)
# ---------------------------------------------------------------------------


def normalization_stages(prompt: str) -> list[tuple[str, str]]:
    """Return the text after each normalisation pass, in order.

    Each tuple is ``(pass_name, text_after_pass)``.  Purely for explainability
    — it lets a UI show exactly how a disguised prompt is peeled back to its
    canonical form.  The detector operates on :func:`ingest`'s ``normalized``
    field, which equals the last stage returned here (before ``.strip()``).
    """
    after_invisible = _strip_invisibles(prompt)
    after_homoglyph = _apply_homoglyph(after_invisible)
    after_ws = _defuse_whitespace_injection(after_homoglyph)
    after_leet = _apply_leetspeak(after_ws)
    return [
        ("raw input", prompt),
        ("strip invisibles + NFKC", after_invisible),
        ("homoglyph substitution", after_homoglyph),
        ("whitespace de-injection", after_ws),
        ("leetspeak normalisation", after_leet),
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def ingest(prompt: str) -> IngestionResult:
    """Normalise, decode, and pre-screen *prompt* before the detector runs.

    Applies normalisation passes in order, decodes encoded payloads, matches
    jailbreak signatures, and computes semantic (or difflib) similarity against
    the known-attack corpus.  All of this is deterministic and runs without any
    LLM call.
    """
    # ── Normalisation cascade ────────────────────────────────────────────
    after_invisible = _strip_invisibles(prompt).strip()

    after_homoglyph = _apply_homoglyph(after_invisible)
    homoglyph_detected = after_homoglyph != after_invisible

    after_ws = _defuse_whitespace_injection(after_homoglyph)
    whitespace_injection_detected = after_ws != after_homoglyph

    # ── Encoded payload detection (before leetspeak to preserve valid tokens) ──
    # base64 / hex tokens contain digits (0, 3, 5 …) that leetspeak would corrupt,
    # so we decode first, then apply the leet pass to the plain text only.
    decoded = _try_decode(after_ws)

    after_leet = _apply_leetspeak(after_ws)
    leetspeak_detected = after_leet != after_ws

    normalized = after_leet.strip()

    # ── Signature matching (on normalised text + any decoded payloads) ───
    haystacks = [normalized] + decoded
    hits: list[str] = []
    for hay in haystacks:
        for pattern in _JAILBREAK_SIGNATURES:
            if re.search(pattern, hay, re.IGNORECASE) and pattern not in hits:
                hits.append(pattern)

    # ── Similarity (semantic embeddings with difflib fallback) ───────────
    # Lazy import to avoid circular dependency (semantic imports nothing from guard).
    from .semantic import ATTACK_CORPUS, difflib_similarity, get_matcher

    matcher = get_matcher()
    if matcher is not None:
        _, sim_score, nearest = matcher.match(normalized)
    else:
        sim_score, nearest = difflib_similarity(normalized, ATTACK_CORPUS)

    return IngestionResult(
        original=prompt,
        normalized=normalized,
        signature_hits=hits,
        decoded_payloads=decoded,
        nearest_attack=nearest,
        similarity=sim_score,
        homoglyph_detected=homoglyph_detected,
        leetspeak_detected=leetspeak_detected,
        whitespace_injection_detected=whitespace_injection_detected,
    )
