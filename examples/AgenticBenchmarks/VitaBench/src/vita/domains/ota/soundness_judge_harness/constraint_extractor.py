"""
Constraint extractor for the NL soundness judge harness.

Preserves natural language descriptions and categorizes constraints by type
(date, price, city, etc.) for LLM-based judgment.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from vita.config import DEFAULT_LLM_EVALUATOR, models
from vita.data_model.message import SystemMessage, UserMessage
from vita.data_model.tasks import Task
from vita.domains.ota.soundness_judge_harness.schema import ExtractedConstraintSet
from vita.domains.ota.verifier.utils import _extract_json
from vita.prompts import get_prompts
from vita.utils.llm_utils import generate

logger = logging.getLogger(__name__)


def extract_constraints(
    task: Task,
    user_message: str,
    llm_model: Optional[str] = None,
    llm_args: Optional[dict] = None,
    language: str = "english",
) -> ExtractedConstraintSet:
    """
    Extract NL constraints from a user's travel request.

    Args:
        task: The Task object (used for metadata: id, environment, user_profile).
        user_message: The user simulation message to extract constraints from.

    Returns an ExtractedConstraintSet with categorized, natural-language constraints.
    """
    if llm_model is None:
        llm_model = DEFAULT_LLM_EVALUATOR

    user_scenario = task.user_scenario
    user_profile = ""
    if user_scenario and user_scenario.user_profile:
        profile = user_scenario.user_profile
        if isinstance(profile, dict):
            user_profile = json.dumps(profile, ensure_ascii=False, indent=2)
        else:
            user_profile = str(profile)

    system_time = ""
    if task.environment and "time" in task.environment:
        system_time = task.environment["time"]

    system_content = get_prompts(language).harness_constraint_extraction_template.format(
        system_time=system_time or "(not specified)",
        user_profile=user_profile or "(none)",
    )

    # Use user_message from user sim file
    messages = [
        SystemMessage(role="system", content=system_content),
        UserMessage(role="user", content=user_message),
    ]

    kwargs = dict(llm_args or {})
    kwargs.setdefault("temperature", 0)

    response = generate(llm_model, messages, enable_think=True, **kwargs)
    raw = response.content if hasattr(response, "content") else str(response)

    parsed = _extract_json(raw)
    if parsed is None:
        raise ValueError(f"Failed to parse constraint extraction output for task {task.id}")

    # Ensure task_id is set
    parsed["task_id"] = task.id

    return ExtractedConstraintSet.model_validate(parsed)
