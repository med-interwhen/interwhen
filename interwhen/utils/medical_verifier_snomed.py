"""
medical_verifier_snomed.py
"""

from __future__ import annotations

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
    max_terms: int = 5


class MedicalReasoningVerifierSnomedFirst(MedicalReasoningVerifier):
    """
    Extends MedicalReasoningVerifier with per-paragraph real-time SNOMED enrichment.

    For every INFERENCE/CONCLUSION paragraph, clinical terms are extracted and
    any not already in the cache are fetched from BioPortal before the base
    verification call runs — so the SNOMED context is maximally populated at
    the point the verifier prompt is assembled.

    _verify_inference always passes _retried=True to super() so the base class
    skips its own redundant SNOMED extraction on UNKNOWN. SnomedFirst has
    already done the enrichment for this paragraph; the base's retry loop
    would just repeat the same term extraction and BioPortal calls.

    Fixes vs original:
      - _enrich_for_paragraph now respects config.run_snomed (was checking
        only self.snomed is None, so --no_snomed was silently ignored)
      - _verify_inference passes _retried=True to super, preventing double
        SNOMED enrichment when the verifier returns UNKNOWN
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

    def _enrich_for_paragraph(
        self,
        paragraph: str,
        question:  str,
        options:   dict,
    ) -> None:
        """
        Extract clinical terms from this paragraph and fetch SNOMED for any
        not already in the cache. Updates self.snomed_cache in-place.
        """
        # Fix: was only checking `self.snomed is None`, ignoring config.run_snomed
        if self.snomed is None or not self.config.run_snomed:
            return

        opts_text  = self._options_text(options)
        terms_resp = self.vllm.call(
            MedicalReasoningPromptBuilder.build_snomed_term_extraction_prompt(
                question        = question,
                options_text    = opts_text,
                reasoning_chunk = paragraph,
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

    def _verify_inference(
        self,
        paragraph: str,
        options:   dict,
        question:  str  = "",
        _retried:  bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Enrich SNOMED cache for this paragraph, then call super with
        _retried=True so the base class skips its own redundant SNOMED
        extraction and retry on UNKNOWN — SnomedFirst has already done it.
        """
        if not _retried:
            self._enrich_for_paragraph(paragraph, question, options)

        # Always tell super it's pre-enriched: prevents base from re-extracting
        # terms and making duplicate BioPortal calls when label == UNKNOWN
        return super()._verify_inference(paragraph, options, question, _retried=True)

    def _verify_conclusion(
        self,
        paragraph: str,
        options:   dict,
        question:  str = "",
    ) -> Tuple[bool, Optional[str]]:
        """Enrich SNOMED cache for this paragraph, then run base conclusion verification."""
        self._enrich_for_paragraph(paragraph, question, options)
        return super()._verify_conclusion(paragraph, options, question)