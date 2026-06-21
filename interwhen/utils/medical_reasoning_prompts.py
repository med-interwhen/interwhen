"""
MedicalReasoningPromptBuilder
"""

from __future__ import annotations

# PROMPT CONFIG 

class PromptConfig:

    # ── Roles ──────────────────────────────────────────────────────────────
    ROLE_WORLD    = "You are a medical reasoning system."
    ROLE_VERIFIER = "You are a medical reasoning verifier."

    # ── Universal rules ────────────────────────────────────────────────────
    RULE_VALID_REASONING = "- Use medically valid reasoning."
    RULE_CONSISTENT      = "- Be logically consistent."
    RETURN_JSON          = "Return ONLY valid JSON."

    # ── Knowledge-restriction rules (reasoning trace) ──────────────────────
    RULE_NO_EXTERNAL_KNOWLEDGE = "- Do NOT use external medical knowledge."
    RULE_TRACE_AS_WORLD        = "- Treat the reasoning trace as complete world knowledge."
    RULE_NO_HALLUCINATE        = "- Do NOT hallucinate unsupported facts."

    # ── TRUE / FALSE definitions — world hypothesis ────────────────────────
    TRUE_WORLD_HYPO    = "- TRUE = correct"
    FALSE_WORLD_HYPO   = "- FALSE = incorrect"

    # ── TRUE / FALSE definitions — reasoning trace hypothesis ──────────────
    TRUE_TRACE_HYPO    = "- TRUE = directly supported or logically inferable from the reasoning trace."
    FALSE_TRACE_HYPO   = "- FALSE = contradicted by the reasoning trace."

    # ── UNKNOWN definitions ────────────────────────────────────────────────
    UNKNOWN_WORLD_HYPO = (
        "- UNKNOWN = information is insufficient to judge medical certainty; "
        "if returning UNKNOWN, include a brief (1–2 sentence) explanation under "
        "'evidence' describing what additional information would resolve the uncertainty."
    )
    UNKNOWN_TRACE_HYPO = (
        "- UNKNOWN = insufficient information in the reasoning trace; "
        "if returning UNKNOWN, include a brief (1–2 sentence) explanation under "
        "'evidence' describing what additional information would resolve the uncertainty."
    )


# BUILDER

C = PromptConfig   


class MedicalReasoningPromptBuilder:

    # Shared helpers

    @staticmethod
    def _allowed_labels(allow_unknown: bool) -> str:
        labels = ["TRUE", "FALSE"]
        if allow_unknown:
            labels.append("UNKNOWN")
        return "/".join(labels)

    # HYPOTHESIS — REASONING TRACE

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


    # HYPOTHESIS — REASONING TRACE + SNOMED (after fetching definitions for an UNKNOWN label)

    @classmethod
    def build_reasoning_hypothesis_snomed_prompt(
        cls,
        *,
        reasoning_trace: str,
        hypothesis: str,
        snomed_context: str,
        allow_unknown: bool = True,
    ) -> str:

        unknown_rule = f"{C.UNKNOWN_WORLD_HYPO}\n" if allow_unknown else ""
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

    # SNOMED TERM EXTRACTION — convert a reasoning chunk into lookup terms

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
