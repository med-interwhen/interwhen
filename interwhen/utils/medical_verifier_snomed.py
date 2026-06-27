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
    max_terms: int = 5


class MedicalReasoningVerifierSnomedFirst(MedicalReasoningVerifier):
    """
    Extends MedicalReasoningVerifier with per-paragraph real-time SNOMED enrichment.
    Extracts terms from the paragraph being verified and fetches SNOMED definitions
    before every inference/conclusion verification call.
    """

    def __init__(self, vllm, snomed=None, config=None, compact_case="", snomed_cache=None):
        super().__init__(
            vllm         = vllm,
            snomed       = snomed,
            config       = config or SnomedFirstConfig(),
            compact_case = compact_case,
            snomed_cache = snomed_cache or {},
        )

    def _enrich_for_paragraph(self, paragraph: str) -> None:
        """Extract clinical terms from paragraph and fetch SNOMED for uncached ones."""
        if self.snomed is None:
            return

        terms_resp = self.vllm.call(
            MedicalReasoningPromptBuilder.build_snomed_term_extraction_prompt(
                question       = "",
                options_text   = "",
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
            self._realtime_snomed(new_terms)

    def _verify_inference(self, paragraph, prior_context, _retried=False):
        if not _retried:
            self._enrich_for_paragraph(paragraph)
        return super()._verify_inference(paragraph, prior_context, _retried=_retried)

    def _verify_conclusion(self, paragraph, prior_context):
        self._enrich_for_paragraph(paragraph)
        return super()._verify_conclusion(paragraph, prior_context)