"""
medical_reasoning_prompts.py  —  Prompt builder for the structured verifier.

All section-specific prompts are keyed to the tagged format the solver now emits.
The paragraph classifier has been REMOVED — the verifier reads section tags directly.
"""

from __future__ import annotations


# ── Prompt Config ─────────────────────────────────────────────────────────────

class PromptConfig:

    # ── Roles ──────────────────────────────────────────────────────────────────
    ROLE_WORLD    = "You are a medical reasoning system."
    ROLE_VERIFIER = "You are a medical reasoning verifier."

    # ── Universal rules ────────────────────────────────────────────────────────
    RULE_VALID_REASONING = "- Use medically valid reasoning."
    RULE_CONSISTENT      = "- Be logically consistent."
    RETURN_JSON          = "Return ONLY valid JSON."

    # ── Knowledge-restriction rules ────────────────────────────────────────────
    RULE_NO_EXTERNAL_KNOWLEDGE = "- Do NOT use external medical knowledge."
    RULE_TRACE_AS_WORLD        = "- Treat the reasoning trace as complete world knowledge."
    RULE_NO_HALLUCINATE        = "- Do NOT hallucinate unsupported facts."

    # ── TRUE / FALSE definitions — world hypothesis ────────────────────────────
    TRUE_WORLD_HYPO  = "- TRUE = correct"
    FALSE_WORLD_HYPO = "- FALSE = incorrect"

    # ── TRUE / FALSE definitions — reasoning trace hypothesis ──────────────────
    TRUE_TRACE_HYPO  = "- TRUE = directly supported or logically inferable from the reasoning trace."
    FALSE_TRACE_HYPO = "- FALSE = contradicted by the reasoning trace."

    # ── UNKNOWN definitions ────────────────────────────────────────────────────
    UNKNOWN_WORLD_HYPO = (
        "- UNKNOWN = information is insufficient to judge medical certainty; "
        " if you have even a slight doubt about answer , return UNKNOWN . "
        "'evidence' describing what additional information would resolve the uncertainty."
    )
    UNKNOWN_TRACE_HYPO = (
        "- UNKNOWN = insufficient information in the reasoning trace; "
        " if you have doubt about answer , return UNKNOWN . "
        "'evidence' describing what additional information would resolve the uncertainty."
    )


C = PromptConfig


# ── Builder ───────────────────────────────────────────────────────────────────

class MedicalReasoningPromptBuilder:

    # Shared helpers

    @staticmethod
    def _allowed_labels(allow_unknown: bool) -> str:
        labels = ["TRUE", "FALSE"]
        if allow_unknown:
            labels.append("UNKNOWN")
        return "/".join(labels)

    # ── CASE EXTRACTION ───────────────────────────────────────────────────────
    # Converts the raw case text into compact structured JSON before generation.
    # Called once per sample in MedicalPreprocessor.extract_case_facts().

    @classmethod
    def build_case_extraction_prompt(cls, *, case_text: str) -> str:
        return f"""{C.ROLE_VERIFIER}

Extract all stated clinical facts from the following case into a compact
structured JSON. Be exhaustive but concise. Do not add inferred information —
only include what is explicitly stated.

{C.RETURN_JSON}

Case:
{case_text}

Output format:
{{
    "patient": "age, sex",
    "chief_complaint": "presenting symptoms and duration",
    "vitals": "BP, HR, RR, Temp, SpO2 as stated",
    "physical_exam": ["finding 1", "finding 2"],
    "labs": ["lab: value"],
    "imaging": ["finding"],
    "ecg": "findings if stated, else null",
    "history": "PMH, surgical, family, social if stated, else null",
    "medications": ["med 1"],
    "other": ["anything not fitting above"]
}}
"""

    # ── OBSERVATION GROUNDING ─────────────────────────────────────────────────
    # The solver emits [OBSERVATION]…[/OBSERVATION] blocks containing ONLY
    # verbatim case facts. This prompt verifies nothing crept in that isn't
    # explicitly stated in the case.
    #
    # Note: the paragraph_classifier prompt has been removed. The verifier now
    # reads section tags directly — no LLM classification call required.

    @classmethod
    def build_observation_grounding_prompt(cls, *, compact_case: str, paragraph: str) -> str:
        return f"""{C.ROLE_VERIFIER}

The solver has written an [OBSERVATION] section. It should contain ONLY facts
explicitly stated in the case. Check every claim.

Flag these error types:
- hallucination: a fact not present in the case
- misread: a value stated incorrectly (e.g. wrong number)
- hidden_inference: a conclusion presented as a fact (contains causal language:
  "suggests", "indicates", "consistent with", "likely", "supports")

{C.RULE_NO_EXTERNAL_KNOWLEDGE}
{C.RULE_NO_HALLUCINATE}
{C.RETURN_JSON}

Case Facts:
{compact_case}

[OBSERVATION] section to verify:
{paragraph}

Output format (if all grounded):
{{
    "grounded": true,
    "issues": []
}}

Output format (if issues found):
{{
    "grounded": false,
    "issues": [
        {{"claim": "exact text of the problematic claim", "type": "hallucination|misread|hidden_inference", "reason": "why"}}
    ]
}}
"""

    # ── INFERENCE VERIFICATION ────────────────────────────────────────────────
    # Verifies an [INFERENCE] block: one clinical claim + its rationale.
    # The solver is required to emit exactly ONE claim per block so feedback
    # can be specific and actionable.

    @classmethod
    def build_inference_verification_prompt(
        cls,
        *,
        compact_case:  str,
        compact_state: str,
        options_text:  str,
        snomed_context: str,
        paragraph:     str,
        allow_unknown: bool = True,
    ) -> str:
        unknown_rule   = f"{C.UNKNOWN_TRACE_HYPO}\n" if allow_unknown else ""
        snomed_section = f"SNOMED CT Definitions:\n{snomed_context}" if snomed_context.strip() else ""
        allowed        = cls._allowed_labels(allow_unknown)

        return f"""{C.ROLE_VERIFIER}

The solver has written an [INFERENCE] block containing one clinical claim.
Verify that the claim is medically valid AND correctly supports or eliminates
the appropriate answer options.

Rules:
{C.RULE_VALID_REASONING}
{C.TRUE_TRACE_HYPO}
{C.FALSE_TRACE_HYPO}
{unknown_rule}{C.RULE_CONSISTENT}
{C.RETURN_JSON}

Case Facts:
{compact_case}

Established So Far:
{compact_state}

Options:
{options_text}

{snomed_section}

[INFERENCE] block to verify:
{paragraph}

Allowed Labels: {allowed}

Output format (if correct):
{{
    "label": "TRUE",
    "evidence": ["reason this inference is valid"],
    "wrong_claim": null,
    "correction": null
}}

Output format (if incorrect):
{{
    "label": "FALSE",
    "evidence": ["what is wrong and why"],
    "wrong_claim": "the specific incorrect statement verbatim",
    "correction": "what the correct reasoning should state"
}}
"""

    # ── OPTION COMPARISON VERIFICATION ───────────────────────────────────────
    # Verifies [OPTION_COMPARISON]: each option is evaluated against the
    # established observations and inferences.

    @classmethod
    def build_option_comparison_verification_prompt(
        cls,
        *,
        compact_case:  str,
        compact_state: str,
        options_text:  str,
        snomed_context: str,
        paragraph:     str,
        allow_unknown: bool = True,
    ) -> str:
        unknown_rule   = f"{C.UNKNOWN_TRACE_HYPO}\n" if allow_unknown else ""
        snomed_section = f"SNOMED CT Definitions:\n{snomed_context}" if snomed_context.strip() else ""
        allowed        = cls._allowed_labels(allow_unknown)

        return f"""{C.ROLE_VERIFIER}

The solver has written an [OPTION_COMPARISON] block where it evaluates each
answer option. Verify that:
  1. Every option is addressed (not silently skipped).
  2. Ruled-out options have a valid, case-supported reason for elimination.
  3. The supported option is consistent with the established inferences.
  4. No option is ruled out using hallucinated or externally introduced facts.

Rules:
{C.RULE_VALID_REASONING}
{C.TRUE_TRACE_HYPO}
{C.FALSE_TRACE_HYPO}
{unknown_rule}{C.RULE_CONSISTENT}
{C.RETURN_JSON}

Case Facts:
{compact_case}

Established So Far:
{compact_state}

Options:
{options_text}

{snomed_section}

[OPTION_COMPARISON] block to verify:
{paragraph}

Allowed Labels: {allowed}

Output format (if correct):
{{
    "label": "TRUE",
    "evidence": ["reason the comparison is valid"],
    "ruled_out": ["option letters correctly eliminated, e.g. A, C"],
    "wrong_claim": null,
    "correction": null
}}

Output format (if incorrect):
{{
    "label": "FALSE",
    "evidence": ["what is wrong and why"],
    "ruled_out": [],
    "wrong_claim": "the specific incorrect statement verbatim",
    "correction": "what the correct reasoning should state"
}}
"""

    # ── CONCLUSION VERIFICATION ───────────────────────────────────────────────
    # Verifies [CONCLUSION]: must be consistent with prior inferences and
    # option comparison. UNKNOWN is NOT allowed here.

    @classmethod
    def build_conclusion_verification_prompt(
        cls,
        *,
        compact_case:  str,
        compact_state: str,
        options_text:  str,
        snomed_context: str,
        paragraph:     str,
    ) -> str:
        snomed_section = f"SNOMED CT Definitions:\n{snomed_context}" if snomed_context.strip() else ""

        return f"""{C.ROLE_VERIFIER}

The solver has written a [CONCLUSION] block selecting a final answer.
Verify that:
  1. The selected option letter is explicitly stated.
  2. The selection is consistent with the established inferences.
  3. The rationale does not contradict earlier reasoning.
  4. The conclusion does not introduce new, unsupported claims.

Rules:
{C.RULE_VALID_REASONING}
{C.RULE_CONSISTENT}
{C.RETURN_JSON}

Case Facts:
{compact_case}

Established So Far:
{compact_state}

Options:
{options_text}

{snomed_section}

[CONCLUSION] block to verify:
{paragraph}

Allowed Labels: TRUE/FALSE

Output format (if consistent):
{{
    "label": "TRUE",
    "selected_option": "<letter>",
    "evidence": ["reason the conclusion is valid"],
    "wrong_claim": null,
    "correction": null
}}

Output format (if inconsistent):
{{
    "label": "FALSE",
    "selected_option": "<letter or null>",
    "evidence": ["what is inconsistent and why"],
    "wrong_claim": "the specific inconsistent statement verbatim",
    "correction": "what the conclusion should state to be consistent"
}}
"""

    # ── SNOMED TERM EXTRACTION ────────────────────────────────────────────────
    # Extracts SNOMED CT lookup terms from any structured section.

    @classmethod
    def build_snomed_term_extraction_prompt(
        cls,
        *,
        question:        str,
        options_text:    str,
        reasoning_chunk: str,
    ) -> str:
        return f"""{C.ROLE_VERIFIER}

Extract SNOMED CT lookup terms from the reasoning section below.

Rules:
- Return ONLY valid JSON.
- Extract concise clinical concepts, not full sentences or paragraphs.
- Prefer disorders, symptoms, signs, body structures, drugs, procedures, tests,
  organisms, substances, and clinically meaningful findings.
- Include terms from the question/options if they are needed to disambiguate
  the reasoning section.
- Do not explain the terms.
- Do not include duplicate terms.

Question:
{question}

Options:
{options_text}

Reasoning section:
{reasoning_chunk}

Output format:
{{
    "terms": [
        "term 1",
        "term 2"
    ]
}}
"""

    # ── ENTITY-TO-CUI MAPPING  (new — graph pipeline) ─────────────────────────
    # Replaces the flat string-list output of build_snomed_term_extraction_prompt
    # with a structured span->CUI candidate mapping, plus a claimed relation
    # between the two most clinically salient entities in the section. The CUI
    # is a CANDIDATE only — medical_graph.EntityMapper validates it against
    # SnomedRelationshipClient before it is trusted anywhere downstream.

    @classmethod
    def build_entity_cui_mapping_prompt(
        cls,
        *,
        question:        str,
        options_text:    str,
        section_body:    str,
    ) -> str:
        return f"""{C.ROLE_VERIFIER}

Identify the clinically salient entities in the reasoning section below and
propose a SNOMED CT concept (CUI candidate) for each one.

Rules:
- Return ONLY valid JSON.
- Only include entities you can name a plausible SNOMED CT concept for —
  disorders, findings, symptoms, body structures, substances, procedures,
  organisms. Skip vague or non-clinical spans.
- "cui_candidate" is your best guess at the SNOMED CT identifier if you know
  it, else null. It will be independently validated — do not fabricate a
  number with false confidence; use null and a clear "fsn_candidate" instead
  if you are not sure of the exact code.
- "confidence" reflects how sure you are this span maps to this exact
  concept (0.0-1.0), not how sure you are the concept exists in SNOMED.
- If the section asserts a relationship between two entities (e.g. "X causes
  Y", "X is a risk factor for Y", "X is treated with Y"), report it once in
  "claimed_relation". If no clear relation is asserted, set it to null.
- Do not include duplicate entities.

Question:
{question}

Options:
{options_text}

Reasoning section:
{section_body}

Output format:
{{
    "entities": [
        {{"span": "exact text span", "fsn_candidate": "preferred SNOMED term", "cui_candidate": "id or null", "confidence": 0.0}}
    ],
    "claimed_relation": {{"source_span": "...", "target_span": "...", "relation_type": "causes|treats|finding_site|associated_with|risk_factor|contraindicated_with|other"}}
}}

If there is no relation, set "claimed_relation" to null.
"""

    # ── HYPOTHESIS — REASONING TRACE (unchanged) ──────────────────────────────

    @classmethod
    def build_reasoning_hypothesis_prompt(
        cls,
        *,
        reasoning_trace: str,
        hypothesis:      str,
        allow_unknown:   bool = True,
    ) -> str:
        allowed      = cls._allowed_labels(allow_unknown)
        unknown_rule = f"{C.UNKNOWN_TRACE_HYPO}\n" if allow_unknown else ""

        return f"""{C.ROLE_VERIFIER}

Evaluate the following hypothesis ONLY using the provided reasoning trace.

Rules:
{C.RULE_NO_EXTERNAL_KNOWLEDGE}
{C.RULE_TRACE_AS_WORLD}
{C.RULE_NO_HALLUCINATE}
{C.TRUE_TRACE_HYPO}
{C.FALSE_TRACE_HYPO}
{unknown_rule}{C.RULE_CONSISTENT}

{C.RETURN_JSON}

Reasoning Trace:
{reasoning_trace}

Hypothesis:
{hypothesis}

Allowed Labels:
{allowed}

Output format:
{{
    "label": "TRUE",
    "evidence": [
        "direct or semantically supported statement from the reasoning trace"
    ]
}}
"""

    # ── HYPOTHESIS — REASONING TRACE + SNOMED (unchanged) ────────────────────

    @classmethod
    def build_reasoning_hypothesis_snomed_prompt(
        cls,
        *,
        reasoning_trace: str,
        hypothesis:      str,
        snomed_context:  str,
        allow_unknown:   bool = True,
    ) -> str:
        unknown_rule  = f"{C.UNKNOWN_WORLD_HYPO}\n" if allow_unknown else ""
        snomed_section = (
            f"SNOMED CT Definitions:\n{snomed_context}"
            if snomed_context.strip()
            else "No SNOMED enrichment was needed."
        )

        return f"""{C.ROLE_WORLD}

Re-evaluate the following hypothesis now that SNOMED CT definitions are
available for the terms that were initially uncertain. Use the reasoning
trace AND the SNOMED CT definitions together.

Rules:
{C.RULE_VALID_REASONING}
{C.TRUE_WORLD_HYPO}
{C.FALSE_WORLD_HYPO}
{unknown_rule}{C.RULE_CONSISTENT}

{C.RETURN_JSON}

Reasoning Trace:
{reasoning_trace}

Hypothesis:
{hypothesis}

{snomed_section}

Output format:
{{
    "label": "TRUE",
    "evidence": [
        "statement from the reasoning trace or the SNOMED definitions"
    ]
}}
"""


if __name__ == "__main__":
    B = MedicalReasoningPromptBuilder

    print(B.build_inference_verification_prompt(
        compact_case  = '{"patient": "45M", "chief_complaint": "chest pain"}',
        compact_state = "Nothing established yet.",
        options_text  = "A. STEMI\nB. Pericarditis\nC. Aortic dissection",
        snomed_context= "",
        paragraph     = "The ST elevation in leads II, III, aVF suggests inferior STEMI.",
        allow_unknown = True,
    ))
