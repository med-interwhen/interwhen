"""
interwhen/utils/confidence.py

Paragraph-level confidence scoring for the Medical Reasoning Monitor.

Two modes, used in priority order:
  1. Logprob-based  — uses top-token log-probabilities from the vLLM stream.
                      Requires the caller to feed raw SSE logprob payloads via
                      push_logprob_chunk().  Mean of max-logprob per token,
                      converted to probability: conf = mean(exp(max_logprob)).
  2. Text-heuristic — falls back when no logprobs are available.
                      Scores the paragraph on hedging language, explicit UNKNOWN
                      markers, structural incompleteness, and option-comparison
                      hedging.  Returns a float in [0, 1] that correlates with
                      how "uncertain" the paragraph reads.

Design contract
---------------
- A confidence score of 1.0 = completely certain (skip verifier).
- A confidence score of 0.0 = maximally uncertain (always verify).
- Caller compares score against a threshold τ:
      should_verify = (score < τ)  OR  (UNKNOWN in chunk)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ══════════════════════════════════════════════════════════════════════════════
# HEDGING LEXICON
# Each entry: (regex_pattern, penalty_weight)
# Weights are additive; total is clipped to [0, 1] before returning.
# ══════════════════════════════════════════════════════════════════════════════

_HEDGES: List[tuple] = [
    # Explicit uncertainty markers
    (r"\bUNKNOWN\b",                                        0.40),
    (r"\b(uncertain|uncertainty)\b",                        0.25),
    (r"\b(unclear|not clear|not sure)\b",                   0.20),
    (r"\b(difficult to (say|determine|assess))\b",          0.20),
    (r"\b(cannot (determine|say|confirm|rule out))\b",      0.18),

    # Probability hedges
    (r"\b(possibly|possible|may|might|could be)\b",         0.12),
    (r"\b(perhaps|presumably|conceivably)\b",               0.10),
    (r"\b(likely|unlikely|probable|improbable)\b",          0.08),
    (r"\b(suspect(ed)?|suspicious for)\b",                  0.07),

    # Clinical hedging
    (r"\b(consider(ing)?|should consider)\b",               0.06),
    (r"\b(cannot (rule out|exclude))\b",                    0.15),
    (r"\b(need(s)? (more|further|additional))\b",           0.12),
    (r"\b(insufficient (evidence|information|data))\b",     0.18),
    (r"\b(without (more|further|additional))\b",            0.10),

    # Contrastive / revision signals (model reconsidering)
    (r"\b(however|but|although|yet|nevertheless)\b",        0.05),
    (r"\bwait\b",                                           0.08),   # "wait, actually..."
    (r"\b(actually|on second thought|re-?evaluat)\b",       0.10),
    (r"\b(alternatively|another possibility)\b",            0.08),
]

_HEDGE_COMPILED = [(re.compile(pat, re.IGNORECASE), w) for pat, w in _HEDGES]


# ══════════════════════════════════════════════════════════════════════════════
# TEXT-HEURISTIC SCORER
# ══════════════════════════════════════════════════════════════════════════════

def text_confidence(paragraph: str) -> float:
    """
    Returns a confidence score in [0, 1].
    Higher = more certain = less likely to need verification.

    Algorithm
    ---------
    1. Compute a raw uncertainty score (penalty) from hedge matches.
    2. Penalise very short paragraphs (< 30 words) — insufficient context.
    3. Penalise option-comparison paragraphs that don't firmly rule anything out.
    4. Return confidence = 1 - clipped_penalty.
    """
    if not paragraph.strip():
        return 1.0   # empty → nothing to verify

    # 1. Hedge penalty
    penalty = 0.0
    for pattern, weight in _HEDGE_COMPILED:
        matches = pattern.findall(paragraph)
        if matches:
            # Each additional match of the same hedger adds diminishing returns
            penalty += weight * (1 + 0.3 * (len(matches) - 1))

    # 2. Length penalty — very short paragraphs are low-info
    word_count = len(paragraph.split())
    if word_count < 15:
        penalty += 0.30
    elif word_count < 30:
        penalty += 0.15

    # 3. Option-comparison without commitment penalty
    #    Detects phrases like "A could be ... B could be ..." with no ruling-out
    option_mentions = len(re.findall(r"\b[A-E]\b", paragraph))
    has_rule_out    = bool(re.search(
        r"\b(rule(s)? out|ruled out|eliminate(d)?|not (the answer|correct|right)|incorrect|wrong)\b",
        paragraph, re.IGNORECASE,
    ))
    if option_mentions >= 3 and not has_rule_out:
        penalty += 0.12

    # 4. Hard floor for UNKNOWN
    if re.search(r"\bUNKNOWN\b", paragraph):
        penalty = max(penalty, 0.60)   # always treat UNKNOWN as low confidence

    confidence = 1.0 - min(penalty, 1.0)
    return round(confidence, 4)


# ══════════════════════════════════════════════════════════════════════════════
# LOGPROB BUFFER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LogprobBuffer:
    """
    Accumulates per-token top-logprob values pushed from the vLLM SSE stream.

    Expected input per token (vLLM /v1/completions logprobs format):
        {"token": "...", "logprob": -0.23, "top_logprobs": {"tok": logprob, ...}}

    Usage
    -----
        buf = LogprobBuffer()
        # In your stream loop, for each SSE chunk that has logprob data:
        buf.push(chunk_logprob_list)   # list of per-token dicts from the chunk
        ...
        # When a paragraph boundary fires:
        conf = buf.paragraph_confidence()
        buf.reset_paragraph()          # clear for next paragraph
    """

    _token_max_logprobs: List[float] = field(default_factory=list)

    def push(self, token_logprob_dicts: Sequence[Dict]) -> None:
        """
        Accept a list of token logprob dicts from one SSE chunk and store
        the max logprob (= top-1 token logprob) for each token.
        """
        for tok_data in token_logprob_dicts:
            # vLLM may give the chosen-token logprob directly
            lp = tok_data.get("logprob")
            if lp is None:
                # Fall back: take max over top_logprobs dict
                top = tok_data.get("top_logprobs", {})
                if top:
                    lp = max(top.values())
            if lp is not None:
                self._token_max_logprobs.append(float(lp))

    def paragraph_confidence(self) -> Optional[float]:
        """
        Returns mean(exp(max_logprob)) over all tokens in the current paragraph,
        or None if no logprobs have been accumulated.

        Interpretation: the average probability the model assigned to each token
        it actually generated, using only its own top-token probability as proxy.
        Range: (0, 1].  Lower = less certain.
        """
        if not self._token_max_logprobs:
            return None
        mean_logprob = sum(self._token_max_logprobs) / len(self._token_max_logprobs)
        return round(math.exp(mean_logprob), 4)

    def reset_paragraph(self) -> None:
        self._token_max_logprobs.clear()

    def __len__(self) -> int:
        return len(self._token_max_logprobs)


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED CONFIDENCE SCORER
# ══════════════════════════════════════════════════════════════════════════════

class ParagraphConfidenceScorer:
    """
    Single entry-point used by MedicalMonitor.

    Priority:
      1. Logprob-based score  (if LogprobBuffer has data)
      2. Text-heuristic score (fallback)

    Also exposes push_logprob_chunk() so the stream runner can feed logprob
    payloads without MedicalMonitor needing to know about the buffer internals.
    """

    def __init__(self) -> None:
        self._buf = LogprobBuffer()
        self._source: str = "none"   # last score source, for logging

    # ── external feed ─────────────────────────────────────────────────────────

    def push_logprob_chunk(self, token_logprob_dicts: Sequence[Dict]) -> None:
        """
        Call this from your stream loop whenever a chunk carries logprob data.
        Compatible with vLLM /v1/completions SSE logprob payloads.
        """
        self._buf.push(token_logprob_dicts)

    # ── scoring ───────────────────────────────────────────────────────────────

    def score(self, paragraph: str) -> float:
        """
        Returns confidence ∈ [0, 1] for the given paragraph.
        Resets the logprob buffer after scoring (paragraph boundary consumed).
        """
        lp_conf = self._buf.paragraph_confidence()
        self._buf.reset_paragraph()

        if lp_conf is not None:
            self._source = "logprob"
            return lp_conf

        self._source = "text_heuristic"
        return text_confidence(paragraph)

    @property
    def last_source(self) -> str:
        """'logprob' or 'text_heuristic' — useful for decision log."""
        return self._source


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

# Recommended starting thresholds.
# Call the verifier when confidence < threshold.
#
# Logprob-based:
#   τ = 0.85  means "verify unless the model is ≥85% confident per token on average"
#   Empirically: MedQA hard questions often score 0.70–0.80 even when correct.
#   Start at 0.80, tune upward if verification is still too frequent.
#
# Text-heuristic:
#   τ = 0.70  because the heuristic is noisier; being more lenient avoids
#   over-verification on straightforward paragraphs with mild hedging.
#
DEFAULT_LOGPROB_THRESHOLD:  float = 0.80
DEFAULT_HEURISTIC_THRESHOLD: float = 0.70