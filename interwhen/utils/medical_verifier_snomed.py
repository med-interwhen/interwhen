"""
medical_verifier_snomed.py  —  SNOMED-first enrichment for structured verifier.

Extends MedicalReasoningVerifier with per-section real-time SNOMED enrichment.

Changes from the free-reasoning version:
  - _enrich_for_section() replaces _enrich_for_paragraph(). Signature is the
    same; the name change reflects that the input is now a tagged section body,
    not an unstructured paragraph.
  - _verify_option_comparison() is overridden so SNOMED enrichment fires for
    that section type too (not just INFERENCE and CONCLUSION).
  - All three overrides follow the same pattern:
      enrich once on first attempt → inherited verify method does the work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .medical_reasoning_prompts import MedicalReasoningPromptBuilder
from .medical_verifier import (
    LocalVLLMClient,
    SnomedClient,
    VerifierConfig,
    MedicalReasoningVerifier,
)


@dataclass
class SnomedFirstConfig(VerifierConfig):
    max_terms: int = 5   # max new terms fetched per section


class MedicalReasoningVerifierSnomedFirst(MedicalReasoningVerifier):
    """
    Extends MedicalReasoningVerifier with per-section real-time SNOMED enrichment.

    For every [INFERENCE], [OPTION_COMPARISON], and [CONCLUSION] section that
    arrives, this subclass:
      1. Calls build_snomed_term_extraction_prompt to extract clinical terms.
      2. Fetches SNOMED definitions for any terms not already in the cache.
      3. Delegates to the base class verify method (which now has a richer cache).

    Enrichment fires only on the FIRST attempt — the retry (if UNKNOWN) already
    has the populated cache and goes directly to the base verify method.

    Observation grounding is inherited unchanged (no SNOMED enrichment needed
    for pure fact-checking against the compact case).
    """

    def __init__(
        self,
        vllm:         LocalVLLMClient,
        snomed:       Optional[SnomedClient]      = None,
        config:       Optional[SnomedFirstConfig] = None,
        compact_case: str                         = "",
        snomed_cache: Optional[Dict[str, str]]    = None,
    ):
        super().__init__(
            vllm         = vllm,
            snomed       = snomed,
            config       = config or SnomedFirstConfig(),
            compact_case = compact_case,
            snomed_cache = snomed_cache or {},
        )

    # ── Core enrichment helper ─────────────────────────────────────────────────

    def _enrich_for_section(
        self,
        section_body: str,
        question:     str,
        options:      dict,
    ) -> None:
        """
        Extract clinical terms from a section body and fetch SNOMED for any
        not already in the cache. Updates self.snomed_cache in-place.
        """
        if self.snomed is None:
            return

        opts_text  = self._options_text(options)
        terms_resp = self.vllm.call(
            MedicalReasoningPromptBuilder.build_snomed_term_extraction_prompt(
                question       = question,
                options_text   = opts_text,
                reasoning_chunk= section_body,
            )
        )
        terms = terms_resp.get("terms", [])
        if not isinstance(terms, list):
            return

        new_terms = [
            str(t).strip() for t in terms
            if str(t).strip() and str(t).strip() not in self.snomed_cache
        ][: self.config.max_terms]

        if new_terms:
            self._realtime_snomed(new_terms, question)

    # ── Overrides ──────────────────────────────────────────────────────────────

    def _verify_inference(
        self,
        paragraph: str,
        options:   dict,
        question:  str  = "",
        _retried:  bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Enrich SNOMED for this [INFERENCE] section, then run base verification."""
        if not _retried:
            self._enrich_for_section(paragraph, question, options)
        return super()._verify_inference(paragraph, options, question, _retried=_retried)

    def _verify_option_comparison(
        self,
        paragraph: str,
        options:   dict,
        question:  str  = "",
        _retried:  bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Enrich SNOMED for this [OPTION_COMPARISON] section, then run base verification."""
        if not _retried:
            self._enrich_for_section(paragraph, question, options)
        return super()._verify_option_comparison(paragraph, options, question, _retried=_retried)

    def _verify_conclusion(
        self,
        paragraph: str,
        options:   dict,
        question:  str = "",
    ) -> Tuple[bool, Optional[str]]:
        """Enrich SNOMED for this [CONCLUSION] section, then run base verification."""
        self._enrich_for_section(paragraph, question, options)
        return super()._verify_conclusion(paragraph, options, question)
