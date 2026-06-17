"""
Dataset and prompt utilities for the Medical Reasoning (MedReason) integration.

Dataset
-------
UCSC-VLAA/MedReason (https://huggingface.co/datasets/UCSC-VLAA/MedReason)

Schema (single "train" split, 32.7k rows):
    dataset_name : str   - source sub-dataset (medmcqa, etc.)
    id_in_dataset: int   - id within the source sub-dataset
    question     : str   - the medical question / vignette
    answer       : str   - reference answer (+ explanation, free text)
    reasoning    : str   - MedReason's own gold reasoning trace (not used as
                            input — kept only for optional reference/eval)
    options      : str   - "Answer Choices: A. ... B. ... " (empty for non-MCQ)

We only use ``question`` (and ``options`` when present) to build the prompt.
``answer`` / ``reasoning`` are reference data for downstream evaluation; they
are never shown to the model.

No forced reasoning format
---------------------------
Per requirements, the model is NOT instructed to use any particular structure
(no "Observation/Inference/..." sections, etc.).  The ONLY structural
requirement is that the model's reasoning is wrapped in a single
``<think> ... </think>`` block, exactly like every other InterWhen dataset
(Game24, Maze, SpatialMap, ZebraLogic, Verina).  Everything inside the think
block is free-form reasoning; the monitor only cares about line counts inside
that block, not its content.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

import datasets

logger = logging.getLogger(__name__)

HF_DATASET_NAME = "UCSC-VLAA/MedReason"
HF_DATASET_SPLIT = "train"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# System prompt: tells the model to think inside <think>...</think> before
# answering. No instructions about *how* to structure the reasoning itself.
SYSTEM_PROMPT_MEDICAL = """\
You are a careful, knowledgeable medical reasoning assistant.

When answering a medical question, first think through the problem step by \
step inside a single <think>...</think> block. Reason in whatever way is \
natural to you — there is no required structure or section headings inside \
the think block.

After the </think> tag, give your final answer clearly and concisely.
"""

# User prompt template. {options_block} is empty string when the question
# has no multiple-choice options.
USER_PROMPT_TEMPLATE = """\
{question}
{options_block}
Think step by step inside <think> tags, then give your final answer.
"""

# Suffix appended directly to the prompt (outside the chat template content)
# so generation naturally begins inside the think block, matching the
# convention used by Game24 / Maze / SpatialMap / ZebraLogic monitors.
THINK_OPEN_TAG = "<think>\n"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def get_medical_dataset(
    split: str = HF_DATASET_SPLIT,
    dataset_name_filter: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load the MedReason dataset from HuggingFace.

    Args:
        split:                HF split to load (MedReason only has "train").
        dataset_name_filter:  If given, keep only rows whose ``dataset_name``
                               field equals this value (e.g. "medmcqa").
        limit:                If given, truncate to the first N rows after
                               filtering (useful for smoke tests).

    Returns:
        List of problem dicts with keys:
            id           (str)  - f"{dataset_name}-{id_in_dataset}"
            dataset_name (str)
            question     (str)
            options      (str)  - may be ""
            answer       (str)  - reference answer (not shown to model)
            reasoning    (str)  - MedReason gold reasoning (reference only)
    """
    ds = datasets.load_dataset(HF_DATASET_NAME, split=split).to_list()

    if dataset_name_filter:
        ds = [r for r in ds if r.get("dataset_name") == dataset_name_filter]

    problems = []
    for row in ds:
        problems.append({
            "id": f"{row.get('dataset_name', 'unk')}-{row.get('id_in_dataset', '')}",
            "dataset_name": row.get("dataset_name", ""),
            "question": row.get("question", ""),
            "options": row.get("options", "") or "",
            "answer": row.get("answer", ""),
            "reasoning": row.get("reasoning", ""),
        })

    if limit:
        problems = problems[:limit]

    return problems


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_user_prompt(problem: Dict[str, Any]) -> str:
    """Build the user-turn prompt text for a single MedReason problem.

    Args:
        problem: Dict as returned by get_medical_dataset() (must contain at
                  least "question"; "options" is optional).

    Returns:
        Formatted user prompt string.
    """
    options_block = problem.get("options", "") or ""
    return USER_PROMPT_TEMPLATE.format(
        question=problem["question"].strip(),
        options_block=("\n" + options_block.strip() + "\n") if options_block.strip() else "",
    )


def build_prompt(problem: Dict[str, Any], tokenizer) -> str:
    """Build the full generation prompt for a MedReason problem.

    Applies the chat template with SYSTEM_PROMPT_MEDICAL + the user prompt,
    then appends ``<think>\\n`` so the model's free-form reasoning is
    generated directly inside the think block — matching the convention of
    every other InterWhen monitor (Game24, Maze, SpatialMap, ZebraLogic).

    Args:
        problem:   Problem dict (see get_medical_dataset()).
        tokenizer: HuggingFace tokenizer supporting apply_chat_template.

    Returns:
        Prompt string ready to send to the vLLM completions endpoint.
    """
    user_prompt = build_user_prompt(problem)

    chat_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT_MEDICAL},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Force the generation to begin inside <think>...</think>, mirroring the
    # rest of the InterWhen monitors. The dataset itself has no implicit
    # think tag, so we add it explicitly here at the prompt level.
    return chat_prompt + THINK_OPEN_TAG


# ---------------------------------------------------------------------------
# Lightweight final-answer extraction (for logging / quick eval only)
# ---------------------------------------------------------------------------

def extract_post_think_answer(output_text: str) -> str:
    """Return the text generated after the final </think> tag.

    Returns the full output_text unchanged if no </think> tag is present.
    """
    if "</think>" in output_text:
        return output_text.split("</think>", 1)[1].strip()
    return output_text.strip()