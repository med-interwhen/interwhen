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
    RULE_COMPARE_OPTIONS = "- Compare all provided options before selecting the final answer."
    RETURN_JSON          = "Return ONLY valid JSON."

    # ── Knowledge-restriction rules (reasoning trace) ──────────────────────
    RULE_NO_EXTERNAL_KNOWLEDGE = "- Do NOT use external medical knowledge."
    RULE_TRACE_AS_WORLD        = "- Treat the reasoning trace as complete world knowledge."
    RULE_NO_HALLUCINATE        = "- Do NOT hallucinate unsupported facts."

    # ── Knowledge-restriction rules (propositions) ─────────────────────────
    RULE_PROPS_AS_WORLD        = "- Treat the propositions as complete world knowledge."

    # ── TRUE / FALSE definitions — world, 1-option ─────────────────────────
    TRUE_WORLD_OPTION  = "- TRUE = the option is correct."
    FALSE_WORLD_OPTION = "- FALSE = the option is incorrect."

    # ── TRUE / FALSE definitions — reasoning trace, 1-option ───────────────
    TRUE_TRACE_OPTION  = "- TRUE = the option is supported by the reasoning trace."
    FALSE_TRACE_OPTION = "- FALSE = the option is contradicted by or absent from the reasoning trace."

    # ── TRUE / FALSE definitions — propositions, 1-option ──────────────────
    TRUE_PROPS_OPTION  = "- TRUE = the option is supported by the propositions."
    FALSE_PROPS_OPTION = "- FALSE = the option is contradicted by or absent from the propositions."

    # ── TRUE / FALSE definitions — world hypothesis ────────────────────────
    TRUE_WORLD_HYPO    = "- TRUE = correct"
    FALSE_WORLD_HYPO   = "- FALSE = incorrect"

    # ── TRUE / FALSE definitions — reasoning trace hypothesis ──────────────
    TRUE_TRACE_HYPO    = "- TRUE = directly supported or logically inferable from the reasoning trace."
    FALSE_TRACE_HYPO   = "- FALSE = contradicted by the reasoning trace."

    # ── UNKNOWN definitions ────────────────────────────────────────────────
    UNKNOWN_WORLD_OPTION = "- UNKNOWN = uncertain or insufficient medical certainty."
    UNKNOWN_TRACE_OPTION = "- UNKNOWN = cannot be inferred confidently from the reasoning trace."
    UNKNOWN_PROPS_OPTION = "- UNKNOWN = cannot be inferred confidently from the given propositions."
    UNKNOWN_WORLD_HYPO   = "- UNKNOWN = uncertain or insufficient medical certainty."
    UNKNOWN_TRACE_HYPO   = "- UNKNOWN = insufficient information in the reasoning trace."


# BUILDER

C = PromptConfig   


class MedicalReasoningPromptBuilder:

    # Shared helpers

    @staticmethod
    def _allowed_answers(
        option_labels: list[str],
        allow_unknown: bool,
    ) -> str:
        # 1-option → TRUE/FALSE style (NONE not applicable)
        if len(option_labels) == 1:
            labels = ["TRUE", "FALSE"]
            if allow_unknown:
                labels.append("UNKNOWN")
            return "/".join(labels)

        out = option_labels.copy()
        if allow_unknown:
            out.append("UNKNOWN")
        return "/".join(out)

    @staticmethod
    def _allowed_labels(allow_unknown: bool) -> str:
        labels = ["TRUE", "FALSE"]
        if allow_unknown:
            labels.append("UNKNOWN")
        return "/".join(labels)

    # OPTION SELECTION — WORLD KNOWLEDGE

    @classmethod
    def build_world_option_prompt(
        cls,
        *,
        question: str,
        options_text: str,
        option_labels: list[str],
        allow_unknown: bool = False,
    ) -> str:

        allowed      = cls._allowed_answers(option_labels, allow_unknown)
        unknown_rule = f"{C.UNKNOWN_WORLD_OPTION}\n" if allow_unknown else ""

        # ── 1-option: TRUE / FALSE style ──────────────────────────────────
        if len(option_labels) == 1:
            return f"""{C.ROLE_WORLD}

Evaluate whether the following option is the correct answer to the question,
using medical knowledge and logical reasoning.

Rules:
{C.RULE_VALID_REASONING}
{C.TRUE_WORLD_OPTION}
{C.FALSE_WORLD_OPTION}
{unknown_rule}{C.RULE_CONSISTENT}

- Assign a probability (0.0–1.0) to each allowed answer; probabilities must sum to 1.
{C.RETURN_JSON}

Question:
{question}

Option:
{options_text}

Allowed Answers:
{allowed}

Output format:
{{
  "reasoning": "Step-by-step medical reasoning.",
  "selected_answer": "TRUE",
  "option_probabilities": {{"TRUE": 0.92, "FALSE": 0.08}}
}}
"""

        # ── 2+ options: A/B/C/NONE style ──────────────────────────────────

        return f"""{C.ROLE_WORLD}

Evaluate the following question and options using medical knowledge and logical reasoning.

Rules:
{C.RULE_VALID_REASONING}
{C.RULE_COMPARE_OPTIONS}
{C.RULE_CONSISTENT}
{unknown_rule}
- Assign a probability (0.0–1.0) to each allowed answer; probabilities must sum to 1.
{C.RETURN_JSON}

Question:
{question}

Options:
{options_text}

Allowed Answers:
{allowed}

Output format:
{{
  "reasoning": "Step-by-step medical reasoning and comparison between options.",
  "selected_answer": "A",
  "option_probabilities": {{"A": 0.75, "B": 0.15, "C": 0.07, "D": 0.03}}
}}
"""

    # OPTION SELECTION — REASONING TRACE

    @classmethod
    def build_reasoning_option_prompt(
        cls,
        *,
        reasoning_trace: str,
        question: str,
        options_text: str,
        option_labels: list[str],
        allow_unknown: bool = True,
    ) -> str:

        allowed      = cls._allowed_answers(option_labels, allow_unknown)
        unknown_rule = f"{C.UNKNOWN_TRACE_OPTION}\n" if allow_unknown else ""

        # ── 1-option: TRUE / FALSE style ──────────────────────────────────
        if len(option_labels) == 1:
            return f"""{C.ROLE_VERIFIER}

Evaluate whether the following option is the correct answer to the question,
ONLY using the provided reasoning trace.

Rules:
{C.RULE_NO_EXTERNAL_KNOWLEDGE}
{C.RULE_TRACE_AS_WORLD}
{C.RULE_NO_HALLUCINATE}
{C.TRUE_TRACE_OPTION}
{C.FALSE_TRACE_OPTION}
{unknown_rule}{C.RULE_CONSISTENT}

- Assign a probability (0.0–1.0) to each allowed answer; probabilities must sum to 1.
{C.RETURN_JSON}

Reasoning Trace:
{reasoning_trace}

Question:
{question}

Option:
{options_text}

Allowed Answers:
{allowed}

Output format:
{{
  "reasoning": "Logical reasoning using only the provided reasoning trace.",
  "selected_answer": "TRUE",
  "option_probabilities": {{"TRUE": 0.92, "FALSE": 0.08}}
}}
"""

        # ── 2+ options: A/B/C/NONE style ──────────────────────────────────

        return f"""{C.ROLE_VERIFIER}

Evaluate the following question and options ONLY using the provided reasoning trace.

Rules:
{C.RULE_NO_EXTERNAL_KNOWLEDGE}
{C.RULE_TRACE_AS_WORLD}
{C.RULE_NO_HALLUCINATE}
{C.RULE_COMPARE_OPTIONS}
{C.RULE_CONSISTENT}
{unknown_rule}
- Assign a probability (0.0–1.0) to each allowed answer; probabilities must sum to 1.
{C.RETURN_JSON}

Reasoning Trace:
{reasoning_trace}

Question:
{question}

Options:
{options_text}

Allowed Answers:
{allowed}

Output format:
{{
  "reasoning": "Logical reasoning using only the provided reasoning trace.",
  "selected_answer": "A",
  "option_probabilities": {{"A": 0.75, "B": 0.15, "C": 0.07, "D": 0.03}}
}}
"""

    # OPTION SELECTION — PROPOSITIONS

    @classmethod
    def build_propositions_option_prompt(
        cls,
        *,
        propositions: list[str],
        question: str,
        options_text: str,
        option_labels: list[str],
        allow_unknown: bool = True,
    ) -> str:

        allowed      = cls._allowed_answers(option_labels, allow_unknown)
        props_block  = "\n".join(f"{i+1}. {p}" for i, p in enumerate(propositions))
        unknown_rule = f"{C.UNKNOWN_PROPS_OPTION}\n" if allow_unknown else ""

        # ── 1-option: TRUE / FALSE style ──────────────────────────────────
        if len(option_labels) == 1:
            return f"""{C.ROLE_VERIFIER}

Evaluate whether the following option is the correct answer to the question,
ONLY using the provided propositions.
You are allowed to reorder and combine them, but do NOT use external knowledge.

Rules:
{C.RULE_NO_EXTERNAL_KNOWLEDGE}
{C.RULE_PROPS_AS_WORLD}
{C.RULE_NO_HALLUCINATE}
{C.TRUE_PROPS_OPTION}
{C.FALSE_PROPS_OPTION}
{unknown_rule}{C.RULE_CONSISTENT}

- Assign a probability (0.0–1.0) to each allowed answer; probabilities must sum to 1.
{C.RETURN_JSON}

Propositions:
{props_block}

Question:
{question}

Option:
{options_text}

Allowed Answers:
{allowed}

Output format:
{{
  "reasoning": "Logical reasoning using only the provided propositions.",
  "selected_answer": "TRUE",
  "option_probabilities": {{"TRUE": 0.92, "FALSE": 0.08}}
}}
"""

        # ── 2+ options: A/B/C/NONE style ──────────────────────────────────

        return f"""{C.ROLE_VERIFIER}

Evaluate the following question and options ONLY using the provided propositions.
You are allowed to reorder and combine them, but do NOT use external knowledge.

Rules:
{C.RULE_NO_EXTERNAL_KNOWLEDGE}
{C.RULE_PROPS_AS_WORLD}
{C.RULE_NO_HALLUCINATE}
{C.RULE_COMPARE_OPTIONS}
{C.RULE_CONSISTENT}
{unknown_rule}
- Assign a probability (0.0–1.0) to each allowed answer; probabilities must sum to 1.
{C.RETURN_JSON}

Propositions:
{props_block}

Question:
{question}

Options:
{options_text}

Allowed Answers:
{allowed}

Output format:
{{
  "reasoning": "Logical reasoning using only the provided propositions.",
  "selected_answer": "A",
  "option_probabilities": {{"A": 0.75, "B": 0.15, "C": 0.07, "D": 0.03}}
}}
"""

    # HYPOTHESIS — WORLD KNOWLEDGE

    @classmethod
    def build_world_hypothesis_prompt(
        cls,
        *,
        hypothesis: str,
        allow_unknown: bool = True,
    ) -> str:

        allowed      = cls._allowed_labels(allow_unknown)
        unknown_rule = f"{C.UNKNOWN_WORLD_HYPO}\n" if allow_unknown else ""

        return f"""{C.ROLE_WORLD}

Determine whether the following hypothesis is supported
using medical knowledge and logical reasoning.

Rules:
{C.RULE_VALID_REASONING}
- Select only one label:
{C.TRUE_WORLD_HYPO}
{C.FALSE_WORLD_HYPO}
{unknown_rule}{C.RULE_CONSISTENT}
- Give a short reasoning or explanation for the selected answer.

- Assign a probability (0.0–1.0) to each allowed answer; probabilities must sum to 1.
{C.RETURN_JSON}

Hypothesis:
{hypothesis}

Allowed Labels:
{allowed}

Output format:
{{
    "label": "TRUE/FALSE/UNKNOWN",
    "evidence": [
        "short medical reason"
    ],
    "option_probabilities": {{"TRUE": 0.85, "FALSE": 0.10, "UNKNOWN": 0.05}}
}}
"""

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

- Assign a probability (0.0–1.0) to each allowed answer; probabilities must sum to 1.
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
    ],
    "option_probabilities": {{"TRUE": 0.85, "FALSE": 0.10, "UNKNOWN": 0.05}}
}}
"""

    # PROPOSITION EXTRACTION — REASONING TRACE

    @classmethod
    def build_preposition_hypothesis_prompt(
        cls,
        *,
        reasoning_trace: str,
        hypothesis: str,
    ) -> str:

        return f"""{C.ROLE_VERIFIER}

Evaluate the following hypothesis ONLY using the provided reasoning trace.

Additionally, identify the logical propositions used to reach the conclusion.

Rules:
{C.RULE_NO_EXTERNAL_KNOWLEDGE}
{C.RULE_TRACE_AS_WORLD}
{C.RULE_NO_HALLUCINATE}
- Extract only propositions directly or semantically supported by the reasoning trace.
{C.RULE_CONSISTENT}

- Assign a probability (0.0–1.0) to each allowed answer; probabilities must sum to 1.
{C.RETURN_JSON}

Reasoning Trace:
{reasoning_trace}

Hypothesis:
{hypothesis}

Output format:
{{
    "label": "TRUE/FALSE/UNKNOWN",
    "evidence": [
        "supporting statement from reasoning trace"
    ],
    "propositions_used": [
        "logical proposition 1",
        "logical proposition 2"
    ],
    "option_probabilities": {{"TRUE": 0.85, "FALSE": 0.10, "UNKNOWN": 0.05}}
}}
"""

    # PROPOSITION EXTRACTION — PROPOSITIONS SOURCE

    @classmethod
    def build_propositions_extraction_prompt(
        cls,
        *,
        propositions: list[str],
        hypothesis: str,
    ) -> str:

        props_block = "\n".join(f"{i+1}. {p}" for i, p in enumerate(propositions))

        return f"""{C.ROLE_VERIFIER}

Evaluate the following hypothesis ONLY using the provided propositions.

Additionally, identify which propositions you used to reach the conclusion.
You are allowed to reorder and combine them.

Rules:
{C.RULE_NO_EXTERNAL_KNOWLEDGE}
{C.RULE_PROPS_AS_WORLD}
{C.RULE_NO_HALLUCINATE}
- Extract only propositions directly or semantically supported by the list.
{C.RULE_CONSISTENT}

- Assign a probability (0.0–1.0) to each allowed answer; probabilities must sum to 1.
{C.RETURN_JSON}

Propositions:
{props_block}

Hypothesis:
{hypothesis}

Output format:
{{
    "label": "TRUE/FALSE/UNKNOWN",
    "evidence": [
        "supporting statement from the propositions"
    ],
    "propositions_used": [
        "logical proposition 1",
        "logical proposition 2"
    ],
    "option_probabilities": {{"TRUE": 0.85, "FALSE": 0.10, "UNKNOWN": 0.05}}
}}
"""

    # SNOMED FEEDBACK — STEP 1: GENERATE REASONING

    @classmethod
    def build_generate_reasoning_prompt(
        cls,
        *,
        question: str,
        options_text: str,
    ) -> str:
        """
        Ask the LLM to produce a step-by-step medical reasoning for the question
        WITHOUT selecting a final answer. Used as the first step of the
        SNOMED-feedback experiment so the model reasons freely over all options.

        Output JSON: { "reasoning": "..." }
        """
        return f"""{C.ROLE_WORLD}

Generate a detailed, step-by-step medical reasoning that could be used to answer
the following question. Do NOT select a final answer yet — only reason through the
medical concepts involved.

Rules:
{C.RULE_VALID_REASONING}
{C.RULE_COMPARE_OPTIONS}
- Do NOT reveal which option you think is correct.
{C.RULE_CONSISTENT}
{C.RETURN_JSON}

Question:
{question}

Options:
{options_text}

Output format:
{{
  "reasoning": "Step-by-step medical reasoning covering all options..."
}}
"""

    # SNOMED FEEDBACK — STEP 5: RE-EVALUATE ALL OPTIONS WITH SNOMED CONTEXT

    @classmethod
    def build_reeval_all_prompt(
        cls,
        *,
        question: str,
        options_text: str,
        option_labels: list,
        generated_reasoning: str,
        snomed_context: str,
        allow_unknown: bool = True,
    ) -> str:
        """
        Re-evaluate ALL options (TRUE / FALSE / UNKNOWN each) using both the
        LLM-generated reasoning trace AND SNOMED CT definitions retrieved for
        options that were initially UNKNOWN.

        The SNOMED context is provided only for hard options, but the model is
        explicitly asked to apply it holistically — resolving unknowns may
        cascade corrections into other options.

        Output JSON: { "A": {"answer": ..., "reasoning": ..., "confidence": ...}, ... }
        """
        unknown_rule = f"{C.UNKNOWN_WORLD_OPTION}\n" if allow_unknown else ""
        snomed_section = (
            f"SNOMED CT Definitions for initially-unknown options:\n{snomed_context}"
            if snomed_context.strip()
            else "No SNOMED enrichment was needed (no UNKNOWN options in the initial pass)."
        )

        return f"""{C.ROLE_WORLD}

Re-evaluate every option of the medical question below as TRUE, FALSE, or UNKNOWN.
You are provided with:
  1. A previously generated reasoning trace.
  2. SNOMED CT concept definitions for options that were initially UNKNOWN —
     use this extra context to resolve uncertainty and also to cross-check
     the other options you evaluated earlier.

Rules:
{C.RULE_VALID_REASONING}
{C.RULE_COMPARE_OPTIONS}
- TRUE  = the option is the correct answer to the question.
- FALSE = the option is incorrect.
{unknown_rule}- Exactly ONE option should be TRUE (unless none apply).
- SNOMED context is provided only for the hard options, but apply it holistically —
  it may help you re-assess every option.
{C.RULE_CONSISTENT}
- Assign a confidence (0.0–1.0) to each evaluation.
{C.RETURN_JSON}

Question:
{question}

Options:
{options_text}

Generated Reasoning:
{generated_reasoning}

{snomed_section}

Output format (one key per option label):
{{
  "A": {{"answer": "TRUE",  "reasoning": "...", "confidence": 0.90}},
  "B": {{"answer": "FALSE", "reasoning": "...", "confidence": 0.85}},
  "C": {{"answer": "FALSE", "reasoning": "...", "confidence": 0.80}},
  "D": {{"answer": "FALSE", "reasoning": "...", "confidence": 0.75}}
}}
"""


if __name__ == "__main__":

    B = MedicalReasoningPromptBuilder
    sep = "=" * 70

    print(sep); print("1. World option | 1 option | TRUE/FALSE"); print(sep)
    print(B.build_world_option_prompt(
        question="Which drug is first-line for T2DM?",
        options_text="A. Metformin",
        option_labels=["A"],
        allow_unknown=False,
    ))

    print(sep); print("2. World option | 1 option | TRUE/FALSE/UNKNOWN"); print(sep)
    print(B.build_world_option_prompt(
        question="Which drug is first-line for T2DM?",
        options_text="A. Metformin",
        option_labels=["A"],
        allow_unknown=True,
    ))

    print(sep); print("3. World option | A/B/C/NONE"); print(sep)
    print(B.build_world_option_prompt(
        question="Which drug is first-line for T2DM?",
        options_text="A. Metformin\nB. Insulin glargine\nC. Glipizide",
        option_labels=["A", "B", "C"],
        allow_unknown=False,
    ))

    print(sep); print("4. Reasoning trace option | 1 option | TRUE/FALSE/UNKNOWN"); print(sep)
    print(B.build_reasoning_option_prompt(
        reasoning_trace="Metformin is generally first-line for T2DM.",
        question="Which drug is first-line for T2DM?",
        options_text="A. Metformin",
        option_labels=["A"],
        allow_unknown=True,
    ))

    print(sep); print("5. Reasoning trace option | A/B/NONE/UNKNOWN"); print(sep)
    print(B.build_reasoning_option_prompt(
        reasoning_trace="Metformin is generally first-line for T2DM.",
        question="Which drug should be started?",
        options_text="A. Metformin\nB. Sitagliptin",
        option_labels=["A", "B"],
        allow_unknown=True,
    ))

    print(sep); print("6. World hypothesis | TRUE/FALSE/UNKNOWN"); print(sep)
    print(B.build_world_hypothesis_prompt(
        hypothesis="Metformin reduces hepatic glucose production.",
        allow_unknown=True,
    ))

    print(sep); print("7. Reasoning trace hypothesis | TRUE/FALSE/UNKNOWN"); print(sep)
    print(B.build_reasoning_hypothesis_prompt(
        reasoning_trace="Metformin acts on the liver. It activates AMPK.",
        hypothesis="Metformin lowers blood glucose by acting on the liver.",
        allow_unknown=True,
    ))

    print(sep); print("8. Proposition extraction | reasoning trace"); print(sep)
    print(B.build_preposition_hypothesis_prompt(
        reasoning_trace="Metformin acts on the liver. AMPK suppresses gluconeogenesis.",
        hypothesis="Metformin lowers blood glucose by acting on the liver.",
    ))