"""
MedicalReasoningPromptBuilder
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
    # Converts the raw case text into a compact structured JSON before generation.
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

    # ── PARAGRAPH CLASSIFIER ──────────────────────────────────────────────────
    # Classifies each completed paragraph so the verifier knows which
    # verification path to take. Called once per trigger.

    @classmethod
    def build_paragraph_classifier_prompt(cls, *, paragraph: str) -> str:
        return f"""{C.ROLE_VERIFIER}

Classify the following paragraph from a medical reasoning trace into exactly
one of these classes:

- OBSERVATION: states facts directly from the case (vitals, symptoms, exam
  findings, test results). No causal claims or conclusions.
- INFERENCE: draws a clinical conclusion, makes a causal claim, or evaluates
  likelihood. Contains words like "suggests", "indicates", "consistent with",
  "rules out", "supports", "likely", "unlikely".
- OPTION_COMPARISON: explicitly compares or evaluates the answer options against
  each other or against the findings.
- CONCLUSION: states a final diagnosis, final answer selection, or summary judgment.
- OTHER: transitions, restatements, revision acknowledgements, or non-verifiable
  content.

{C.RETURN_JSON}

Paragraph:
{paragraph}

Output format:
{{
    "class": "INFERENCE",
    "reason": "one sentence explaining the classification"
}}
"""

    # ── OBSERVATION GROUNDING ─────────────────────────────────────────────────
    # Checks that observations only contain facts from the case — no hallucination,
    # misread values, or inferences disguised as observations.

    @classmethod
    def build_observation_grounding_prompt(cls, *, compact_case: str, paragraph: str) -> str:
        return f"""{C.ROLE_VERIFIER}

Check whether every fact in the observation paragraph is directly stated or
clearly implied by the case facts. Do not use external medical knowledge.

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

Observation Paragraph:
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
    # Verifies clinical validity of an inference or conclusion paragraph.
    # Includes options context so the verifier can judge whether the inference
    # is pointing toward/away from the right options for the right reasons.
    # Used for both INFERENCE and CONCLUSION paragraph types.

    @classmethod
    def build_inference_verification_prompt(
        cls,
        *,
        compact_case: str,
        compact_state: str,
        options_text: str,
        snomed_context: str,
        paragraph: str,
        allow_unknown: bool = True,
    ) -> str:
        unknown_rule  = f"{C.UNKNOWN_WORLD_HYPO}\n" if allow_unknown else ""
        snomed_section = f"SNOMED CT Definitions:\n{snomed_context}" if snomed_context.strip() else ""
        allowed        = cls._allowed_labels(allow_unknown)

        return f"""{C.ROLE_VERIFIER}

Verify the following reasoning paragraph. The model is choosing between the
listed options — judge whether the reasoning is medically valid AND correctly
supports or eliminates the appropriate options.

Rules:
{C.RULE_VALID_REASONING}
{C.TRUE_WORLD_HYPO}
{C.FALSE_WORLD_HYPO}
{unknown_rule}{C.RULE_CONSISTENT}
{C.RETURN_JSON}

Case Facts:
{compact_case}

Established So Far:
{compact_state}

Options:
{options_text}

{snomed_section}

Paragraph to Verify:
{paragraph}

Allowed Labels: {allowed}

Output format (if correct):
{{
    "label": "TRUE",
    "evidence": ["reason this is correct"],
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

    # ── HYPOTHESIS — REASONING TRACE (original, unchanged) ────────────────────

    @classmethod
    def build_reasoning_hypothesis_prompt(
        cls,
        *,
        reasoning_trace: str,
        hypothesis: str,
        allow_unknown: bool = True,
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

    # ── HYPOTHESIS — REASONING TRACE + SNOMED (original, unchanged) ───────────

    @classmethod
    def build_reasoning_hypothesis_snomed_prompt(
        cls,
        *,
        reasoning_trace: str,
        hypothesis: str,
        snomed_context: str,
        allow_unknown: bool = True,
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

    # ── SNOMED TERM EXTRACTION (original, unchanged) ───────────────────────────

    @classmethod
    def build_snomed_term_extraction_prompt(
        cls,
        *,
        question: str,
        options_text: str,
        reasoning_chunk: str,
    ) -> str:
        return f"""{C.ROLE_VERIFIER}

Extract SNOMED CT lookup terms from the reasoning chunk.

Rules:
- Return ONLY valid JSON.
- Extract concise clinical concepts, not full sentences or paragraphs.
- Prefer disorders, symptoms, signs, body structures, drugs, procedures, tests,
  organisms, substances, and clinically meaningful findings.
- Include terms from the question/options if they are needed to disambiguate
  the reasoning chunk.
- Do not explain the terms.
- Do not include duplicate terms.

Question:
{question}

Options:
{options_text}

Reasoning chunk:
{reasoning_chunk}

Output format:
{{
    "terms": [
        "term 1",
        "term 2"
    ]
}}
"""


if __name__ == "__main__":
    B = MedicalReasoningPromptBuilder

    print(B.build_reasoning_hypothesis_prompt(
        reasoning_trace="Metformin acts on the liver. It activates AMPK.",
        hypothesis="Metformin lowers blood glucose by acting on the liver.",
        allow_unknown=True,
    ))

    print(B.build_reasoning_hypothesis_snomed_prompt(
        reasoning_trace="Metformin acts on the liver. It activates AMPK.",
        hypothesis="Metformin lowers blood glucose by acting on the liver.",
        snomed_context="Metformin: a biguanide antihyperglycemic agent.",
        allow_unknown=True,
    ))