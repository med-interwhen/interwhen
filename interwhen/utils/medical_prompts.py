"""
medical_prompts.py  —  Structured reasoning prompt templates.

The solver is required to emit reasoning inside explicit XML-like section tags.
The verifier reads these tags directly — no classification LLM call needed.

Section schema (in order, all required):
  [OBSERVATION] … [/OBSERVATION]        — verbatim facts from the case only
  [INFERENCE] … [/INFERENCE]            — one clinical claim + rationale (repeatable)
  [OPTION_COMPARISON] … [/OPTION_COMPARISON]  — systematic option elimination
  [CONCLUSION] … [/CONCLUSION]          — final answer selection + one-line rationale
"""

# ---------------------------------------------------------------------------
# Primary solver prompt  (structured)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_MEDICAL = """\
# Medical Reasoning Assistant

You are an expert clinical reasoning assistant.
Reason through the medical case using the structured format below, then state
your final answer.

## Required section format

You MUST use these tagged sections, in this order, inside your thinking:

[OBSERVATION]
List ONLY facts explicitly stated in the case: vitals, symptoms, exam findings,
labs, imaging. Do NOT add inferred information, unstated history, or risk factors.
[/OBSERVATION]

[INFERENCE]
State ONE clinical claim (e.g. a differential, mechanism, or pathophysiology).
Cite which observation supports it.
If you are uncertain, write UNKNOWN: <what would resolve it> on its own line.
[/INFERENCE]

(Repeat [INFERENCE]…[/INFERENCE] blocks as needed — one claim per block.)

[OPTION_COMPARISON]
Evaluate each answer option against your observations and inferences.
For each option: support it or rule it out with a reason.
[/OPTION_COMPARISON]

[CONCLUSION]
State the single best answer option letter and why, consistent with your reasoning above.
Do NOT change your answer from what your inferences support.
[/CONCLUSION]

## Final Answer

After your thinking is complete, write:

[FINAL ANSWER]
Selected Option: <exact option letter, e.g. A>
Selected Text: <full text of the chosen option>
Reasoning: <one sentence>
[/FINAL ANSWER]
"""

# ---------------------------------------------------------------------------
# Vanilla baseline  (no structure required — for ablation / comparison)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_VANILLA = """\
You are an expert clinical reasoning assistant.
Reason through the following medical case carefully.
Think through the differential diagnosis and arrive at the most likely diagnosis.
Conclude with a final answer.
"""

# ---------------------------------------------------------------------------
# State extractor  (used for context compression on long traces)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_STATE_EXTRACT = """\
You are a medical reasoning state extractor.
Given a partial clinical reasoning trace, extract the current working state
as a structured JSON object.

Return ONLY valid JSON — no prose, no markdown fences:

{
  "observations": ["<obs1>", ...],
  "inferences": [
    {"claim": "<claim>", "support": "<obs or evidence>", "certainty": "high|medium|low"}
  ],
  "evidence": ["<evidence1>", ...],
  "working_diagnosis": [
    {"diagnosis": "<name>", "icd": "<code or null>", "rank": 1}
  ],
  "ruled_out": ["<option>"],
  "contraindications": ["<item>"]
}

Return an empty array [] for missing sections rather than null.
"""

# ---------------------------------------------------------------------------
# User-facing prompt template
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """\
{case_text}

Please reason through this case using the required structured sections, then
provide your Final Answer.
"""

# ---------------------------------------------------------------------------
# Feedback injection template
# ---------------------------------------------------------------------------

FEEDBACK_TEMPLATE = """\

[FEEDBACK]
{feedback_text}
[/FEEDBACK]

"""

# ---------------------------------------------------------------------------
# Section tag registry  (single source of truth — imported by verifier too)
# ---------------------------------------------------------------------------

# Maps tag name → paragraph type used by the verifier routing logic.
SECTION_TAG_TO_TYPE: dict[str, str] = {
    "OBSERVATION":       "OBSERVATION",
    "INFERENCE":         "INFERENCE",
    "OPTION_COMPARISON": "OPTION_COMPARISON",
    "CONCLUSION":        "CONCLUSION",
}

# All recognised opening tags (used for trigger detection in step_extractor)
ALL_OPEN_TAGS:  tuple[str, ...] = tuple(f"[{t}]"  for t in SECTION_TAG_TO_TYPE)

# All recognised closing tags (used for trigger detection in step_extractor)
ALL_CLOSE_TAGS: tuple[str, ...] = tuple(f"[/{t}]" for t in SECTION_TAG_TO_TYPE)