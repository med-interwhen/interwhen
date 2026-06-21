"""
medical_verifier_snomed_first.py
===================================
Alternate verifier: queries SNOMED CT on every verification cycle, not
only when the judge returns UNKNOWN.

Order per cycle: extract the clinical terms in the new content -> look
each one up in SNOMED CT -> judge the content once, with those
definitions already in the prompt. There is no trace-only judgment call
at all when SNOMED data is available; the judge always sees it.

Subclasses MedicalReasoningVerifier (medical_verifier.py) and overrides
only verify_trace(). _split_latest, _truncate_prior, _options_text, and
_format_feedback are inherited unchanged. LocalVLLMClient, SnomedClient,
and VerifierConfig are also reused as-is from medical_verifier.py.

Term extraction has no equivalent in medical_reasoning_prompts.py, so a
small prompt for it is defined below, in TERM_EXTRACTION_PROMPT. The
judgment call itself still uses build_reasoning_hypothesis_snomed_prompt
from medical_reasoning_prompts.py, unmodified.

Contract
--------
    verify_trace(text, question, options) -> (passed: bool, feedback: str | None)

Same as MedicalReasoningVerifier — this is a drop-in alternative. To use
it instead of the UNKNOWN-only verifier, change the import and the class
constructed in MedicalMonitor.__init__:

    from ..utils.medical_verifier_snomed_first import MedicalReasoningVerifierSnomedFirst
    self.verifier = MedicalReasoningVerifierSnomedFirst(vllm=..., snomed=..., config=...)

Cost note
---------
Every verification cycle now does a term-extraction LLM call plus one
SNOMED API call per extracted term, in addition to the judgment call —
up to 1 + max_terms calls per cycle instead of 1 (or 2 only on UNKNOWN,
as in the base verifier). Latency and BioPortal usage scale accordingly.

.env
----
  VLLM_BASE_URL=http://localhost:8000/v1
  VLLM_MODEL=medverifier
  BIOPORTAL_API_KEY=<your-key>
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .medical_reasoning_prompts import MedicalReasoningPromptBuilder
from .medical_verifier import LocalVLLMClient, SnomedClient, VerifierConfig, MedicalReasoningVerifier


TERM_EXTRACTION_PROMPT = """You are a clinical terminology extractor.

Read the medical text below and list the specific clinical terms it
contains: drug names, diagnoses, anatomical structures, lab or imaging
findings, and procedures. Do not include whole sentences or vague phrases.

Return ONLY valid JSON in this format:
{{
  "terms": ["term 1", "term 2"]
}}

List at most {max_terms} terms. If there are no identifiable clinical
terms, return an empty list.

Text:
{text}
"""


@dataclass
class SnomedFirstConfig(VerifierConfig):
    max_terms: int = 5   # caps both the extraction call's output and the SNOMED lookups per cycle


class MedicalReasoningVerifierSnomedFirst(MedicalReasoningVerifier):
    """Same contract as MedicalReasoningVerifier; SNOMED runs before every judgment, not only on UNKNOWN."""

    def __init__(
        self,
        vllm: LocalVLLMClient,
        snomed: Optional[SnomedClient] = None,
        config: Optional[SnomedFirstConfig] = None,
    ):
        super().__init__(vllm=vllm, snomed=snomed, config=config or SnomedFirstConfig())

    def _extract_terms(self, hypothesis: str) -> List[str]:
        prompt = TERM_EXTRACTION_PROMPT.format(text=hypothesis, max_terms=self.config.max_terms)
        resp = self.vllm.call(prompt)
        terms = resp.get("terms", [])
        if not isinstance(terms, list):
            return []
        return [str(t).strip() for t in terms if str(t).strip()][: self.config.max_terms]

    def _fetch_snomed_context(self, question: str, hypothesis: str) -> Optional[str]:
        """Returns a SNOMED feedback block, or None if there's nothing to look up."""
        if self.snomed is None:
            return None

        terms = self._extract_terms(hypothesis)
        if not terms:
            return None

        enrichments = {}
        for term in terms:
            print(f"  [SNOMED] looking up: '{term}'")
            enrichments[term] = self.snomed.enrich(question=question, option_text=term)
            time.sleep(self.config.snomed_rate_limit_sleep)

        return SnomedClient.build_feedback_block(enrichments)

    def verify_trace(
        self,
        text: str,
        question: str = "",
        options: Optional[dict] = None,
    ) -> Tuple[bool, Optional[str]]:
        options = options or {}
        prior_context, new_content = self._split_latest(text)

        if not new_content.strip():
            return True, None

        context_block = question
        if options:
            context_block += "\n\nOptions:\n" + self._options_text(options)
        if prior_context.strip():
            context_block += "\n\nReasoning so far:\n" + self._truncate_prior(prior_context)

        snomed_block = self._fetch_snomed_context(question, new_content)

        if snomed_block is not None:
            prompt = MedicalReasoningPromptBuilder.build_reasoning_hypothesis_snomed_prompt(
                reasoning_trace=context_block,
                hypothesis=new_content,
                snomed_context=snomed_block,
                allow_unknown=self.config.allow_unknown,
            )
        else:
            # No SNOMED client configured, or no terms were found to look up —
            # fall back to the trace-only prompt rather than skip judgment.
            prompt = MedicalReasoningPromptBuilder.build_reasoning_hypothesis_prompt(
                reasoning_trace=context_block,
                hypothesis=new_content,
                allow_unknown=self.config.allow_unknown,
            )

        resp = self.vllm.call(prompt)
        label = str(resp.get("label", "ERROR")).strip().upper()

        if label == "TRUE":
            return True, None
        if label == "FALSE":
            return False, self._format_feedback(resp, snomed_block=snomed_block)
        if label == "UNKNOWN":
            if self.config.unknown_defaults_to_pass:
                return True, None
            return False, self._format_feedback(resp, snomed_block=snomed_block, prefix="Unresolved uncertainty: ")

        # Malformed/unexpected label from the judge model — fail open.
        return True, None