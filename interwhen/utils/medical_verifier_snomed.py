"""
medical_verifier_snomed.py
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
    max_terms: int = 5   # max terms extracted per inference paragraph for real-time lookup


class MedicalReasoningVerifierSnomedFirst(MedicalReasoningVerifier):
    """
    Extends MedicalReasoningVerifier with per-paragraph real-time SNOMED enrichment.

    The base class uses the SNOMED cache pre-fetched by MedicalPreprocessor
    (option terms, fetched before generation). This subclass additionally
    extracts clinical terms from each INFERENCE/CONCLUSION paragraph as it
    arrives and fetches SNOMED definitions for any terms not already cached.

    Result: the cache grows richer throughout generation. By the time a
    conclusion is reached, both the pre-known option concepts and any
    paragraph-specific concepts encountered during reasoning are available.

    Only _verify_inference and _verify_conclusion are overridden — all
    observation grounding, state management, splitting, and feedback
    formatting are inherited unchanged.
    """

    def __init__(
        self,
        vllm:         LocalVLLMClient,
        snomed:       Optional[SnomedClient]           = None,
        config:       Optional[SnomedFirstConfig]      = None,
        compact_case: str                              = "",
        snomed_cache: Optional[Dict[str, str]]         = None,
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
        Extract clinical terms specific to this paragraph and fetch SNOMED
        for any not already in the cache. Updates self.snomed_cache in-place.
        """
        if self.snomed is None:
            return

        opts_text  = self._options_text(options)
        terms_resp = self.vllm.call(
            MedicalReasoningPromptBuilder.build_snomed_term_extraction_prompt(
                question       = question,
                options_text   = opts_text,
                reasoning_chunk= paragraph,
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
        """Enrich SNOMED cache for this paragraph, then run base inference verification."""
        if not _retried:
            # Only enrich on the first attempt — cache is populated for the retry
            self._enrich_for_paragraph(paragraph, question, options)
        return super()._verify_inference(paragraph, options, question, _retried=_retried)

    def _verify_conclusion(
        self,
        paragraph: str,
        options:   dict,
        question:  str = "",
    ) -> Tuple[bool, Optional[str]]:
        """Enrich SNOMED cache for this paragraph, then run base conclusion verification."""
        self._enrich_for_paragraph(paragraph, question, options)
        return super()._verify_conclusion(paragraph, options, question)