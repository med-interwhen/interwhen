"""
Soundness Judge Harness — NL-based constraint checking during simulation.

Three-phase architecture:
  1. Constraint Extraction (offline): Extract NL constraints from task instructions
  2. Memory Store (runtime): SLM distills read tool call results into relevant facts
  3. Judgment (runtime): On each write tool call, evaluate constraints against memory

Usage:
    from vita.domains.ota.soundness_judge_harness import create_harness
    harness = create_harness(task, llm_model="claude-sonnet-4.6")

    # During simulation — called by orchestrator:
    # On read tool calls:
    harness.observe_tool_response(tool_name, tool_args, tool_response)
    # On write tool calls:
    feedback = harness.check_soundness(tool_name, tool_args)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from vita.config import models as model_configs
from vita.data_model.tasks import Task
from vita.domains.ota.soundness_judge_harness.judge import ConstraintJudge
from vita.domains.ota.soundness_judge_harness.memory_store import MemoryWriter
from vita.domains.ota.soundness_judge_harness.schema import (
    ConstraintVerdict,
    ExtractedConstraintSet,
    JudgmentResult,
)

logger = logging.getLogger(__name__)


class SoundnessJudgeHarness:
    """
    Orchestrates the three-phase NL soundness checking pipeline.

    Exposes the same check_soundness() interface expected by the Orchestrator,
    plus observe_tool_response() for memory updates on read calls.
    """

    WRITE_PREFIXES = ("create_", "cancel_", "modify_")

    def __init__(
        self,
        constraints: ExtractedConstraintSet,
        memory_writer: MemoryWriter,
        judge: ConstraintJudge,
    ):
        self.constraints = constraints
        self.memory_writer = memory_writer
        self.judge = judge
        self.soundness_call_log: list[dict] = []

    def observe_tool_response(
        self,
        tool_call_name: str,
        tool_call_args: dict,
        tool_response: str,
    ) -> None:
        """
        Called after every tool call execution.
        Updates memory for read calls; no-op for write calls.
        """
        self.memory_writer.process_tool_response(
            tool_call_name, tool_call_args, tool_response
        )

    def check_soundness(
        self,
        tool_call_name: str,
        tool_call_args: dict,
        trajectory: Optional[list] = None,  # unused, kept for interface compat
    ) -> Optional[str]:
        """
        Evaluate a write tool call against constraints using accumulated memory.

        Returns blocking feedback string if violations detected, None otherwise.
        """
        if not any(tool_call_name.startswith(p) for p in self.WRITE_PREFIXES):
            return None
        if tool_call_args.get("override") is True:
            return None

        result = self.judge.judge(
            tool_call_name,
            tool_call_args,
            self.memory_writer.get_memory(),
        )

        # Log every judgment, including current memory snapshot
        self.soundness_call_log.append({
            "tool_call_name": result.tool_call_name,
            "tool_call_args": result.tool_call_args,
            "memory": self.memory_writer.get_memory().model_dump(),
            "judgments": [
                {"constraint_id": j.constraint_id, "verdict": j.verdict.value, "reasoning": j.reasoning}
                for j in result.judgments
            ],
            "has_violation": result.has_violation,
        })

        if not result.has_violation:
            return None

        # Format blocking feedback
        violations = result.violated_constraints
        lines = [
            "[SOUNDNESS CHECK] Order blocked — constraint violations detected:",
        ]
        for v in violations:
            lines.append(f"  - [{v.constraint_id}] {v.reasoning}")
        lines.append(
            "Please review your tool call against the user instructions. "
            "Fix the issue or pass override=true if you believe this is a mistake."
        )
        return "\n".join(lines)


def create_harness(
    task: Task,
    llm_model: Optional[str] = None,
    memory_model: Optional[str] = None,
    language: str = "english",
    constraints_file: Optional[str] = None,
) -> SoundnessJudgeHarness:
    """
    Build a SoundnessJudgeHarness for a task.

    Args:
        task: The task to create the harness for.
        llm_model: Model for judgment.
        memory_model: Model for memory writing (can be smaller/cheaper).
                      Defaults to llm_model if not specified.
        language: Prompt language.
        constraints_file: Path to pre-extracted constraints JSON.
                          Required — no live extraction fallback.
    """
    if llm_model is None:
        from vita.config import DEFAULT_LLM_EVALUATOR
        llm_model = DEFAULT_LLM_EVALUATOR

    if memory_model is None:
        memory_model = llm_model

    # Phase 1: Load pre-extracted constraints (no fallback)
    constraints = _load_constraints(task, constraints_file)
    
    if constraints is None:
        raise ValueError(
            f"Task {task.id}: No constraints found. "
            f"Please provide a valid constraints_file or run the pre-extraction script."
        )

    # Phase 2: Create memory writer (seeded with user profile)
    # Memory writer: no model config passed — keeps thinking disabled, cheap calls
    user_profile = ""
    if task.user_scenario and task.user_scenario.user_profile:
        profile = task.user_scenario.user_profile
        if isinstance(profile, dict):
            user_profile = json.dumps(profile, ensure_ascii=False, indent=2)
        else:
            user_profile = str(profile)

    # memory_llm_args = dict(model_configs.get(memory_model, model_configs.get("default", {})))
    memory_writer = MemoryWriter(
        constraints=constraints,
        llm_model=memory_model,
        # llm_args=memory_llm_args,
        user_profile=user_profile or None,
        language=language,
    )

    # Phase 3: Create judge (pass full model config for thinking etc.)
    judge_llm_args = dict(model_configs.get(llm_model, model_configs.get("default", {})))
    judge = ConstraintJudge(
        constraints=constraints,
        llm_model=llm_model,
        llm_args=judge_llm_args,
        language=language,
    )

    logger.info("Task %s: soundness judge harness created (%s / memory: %s)", task.id, llm_model, memory_model)
    return SoundnessJudgeHarness(constraints, memory_writer, judge)


def _load_constraints(
    task: Task,
    constraints_file: Optional[str],
) -> ExtractedConstraintSet:
    """Load pre-extracted constraints from file. No live extraction fallback."""
    if constraints_file is None:
        return None

    path = Path(constraints_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Task {task.id}: constraints file not found at {constraints_file}. "
            f"Run the pre-extraction script first."
        )

    with open(path) as f:
        data = json.load(f)

    task_data = data.get(task.id)
    if task_data is None:
        raise FileNotFoundError(
            f"Task {task.id} not found in constraints file {constraints_file}. "
            f"Run the pre-extraction script for this task."
        )
    task_data["task_id"] = task.id
    return ExtractedConstraintSet.model_validate(task_data)
