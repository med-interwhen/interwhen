"""
medical_verifier_snomed.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .medical_reasoning_prompts import MedicalReasoningPromptBuilder
from .medical_verifier import (
    LocalVLLMClient,
    SnomedClient,
    PubMedClient,
    VerifierConfig,
    MedicalReasoningVerifier,
)


@dataclass
class SnomedFirstConfig(VerifierConfig):
    max_terms: int = 5


class MedicalReasoningVerifierSnomedFirst(MedicalReasoningVerifier):
    """
    Extends MedicalReasoningVerifier with per paragraph real time SNOMED enrichment.
    Only enriches SNOMED cache when evidence_source includes "snomed".
    When evidence_source is "pubmed" or "none", _enrich_for_paragraph is a no op.
    """

    def __init__(
        self,
        vllm,
        snomed       = None,
        pubmed       = None,
        config       = None,
        compact_case = "",
        snomed_cache = None,
    ):
        super().__init__(
            vllm         = vllm,
            snomed       = snomed,
            pubmed       = pubmed,
            config       = config or SnomedFirstConfig(),
            compact_case = compact_case,
            snomed_cache = snomed_cache or {},
        )

    def _enrich_for_paragraph(self, content: str) -> None:
        """
        Extract clinical terms from paragraph and fetch SNOMED for uncached ones.
        Skipped entirely when evidence_source is "pubmed" or "none" 
        no point populating the SNOMED cache if we won't use it.
        """
        if self.config.evidence_source not in ("snomed", "both"):
            return
        if self.snomed is None:
            return

        terms_resp = self.vllm.call(
            MedicalReasoningPromptBuilder.build_snomed_term_extraction_prompt(
                question       = "",
                options_text   = "",
                reasoning_chunk= content,
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

    def _verify_inference(self, content, prior_context, _retried=False):
        if not _retried:
            self._enrich_for_paragraph(content)
        return super()._verify_inference(content, prior_context, _retried=_retried)

    def _verify_conclusion(self, content, prior_context):
        self._enrich_for_paragraph(content)
        return super()._verify_conclusion(content, prior_context)
