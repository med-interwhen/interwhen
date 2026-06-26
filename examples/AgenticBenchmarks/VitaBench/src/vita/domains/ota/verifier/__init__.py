"""
OTA Verifier — runs during simulation.

Two independent subsystems:
  - Soundness (LLM judge): validates each write tool call BEFORE execution.
  - Completeness (rule-based): checks final order state when the agent stops.

Usage:
    from vita.domains.ota.verifier import create_verifier
    verifier = create_verifier(task, language="english")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from vita.data_model.tasks import Task
from vita.config import models as model_configs
from vita.domains.ota.completeness import check_completeness, CompletenessConstraints
from vita.domains.ota.soundness_judge_llm import SoundnessJudge, SoundnessJudgeConfig
from vita.domains.ota.soundness_judge_harness import SoundnessJudgeHarness

logger = logging.getLogger(__name__)


class OTAVerifier:
    """
    Live verifier with two subsystems:
      - Soundness: LLM judge blocks bad write tool calls.
      - Completeness: rule-based check that all required bookings exist at stop time.
    """

    WRITE_PREFIXES = ("create_", "cancel_", "modify_")

    def __init__(
        self,
        completeness_constraints: Optional[CompletenessConstraints],
        environment: dict,
        soundness_mode: str = "off",
        completeness_mode: str = "on",
        soundness_judge: Optional[SoundnessJudge] = None,
        soundness_harness: Optional[SoundnessJudgeHarness] = None,
    ):
        self.soundness_mode = soundness_mode
        self.completeness_mode = completeness_mode
        self.soundness_judge = soundness_judge
        self.soundness_harness = soundness_harness
        self.environment = environment
        self.soundness_call_log: list[dict] = []
        self.completeness_constraints = completeness_constraints

    def check_soundness(self, tool_call_name: str, tool_call_args: dict, trajectory: Optional[list] = None) -> Optional[str]:
        """
        Validate a write tool call BEFORE execution.
        Dispatches to the appropriate handler based on soundness_mode.
        """
        if not any(tool_call_name.startswith(p) for p in self.WRITE_PREFIXES):
            return None
        if tool_call_args.get("override") is True:
            return None

        if self.soundness_mode == "off":
            return None
        elif self.soundness_mode == "llm":
            return self._check_soundness_llm(tool_call_name, tool_call_args, trajectory)
        elif self.soundness_mode == "harness":
            return self._check_soundness_harness(tool_call_name, tool_call_args, trajectory)
        else:
            logger.warning("Unknown soundness_mode '%s', allowing call", self.soundness_mode)
            return None

    def _check_soundness_llm(self, tool_call_name: str, tool_call_args: dict, trajectory: Optional[list]) -> Optional[str]:
        """LLM judge evaluates the tool call against user instructions + tool history."""
        verdict, reason = self.soundness_judge.judge(tool_call_name, tool_call_args, trajectory or [])
        self.soundness_call_log.append({
            "tool_call_name": tool_call_name,
            "tool_call_args": tool_call_args,
            "verdict": verdict,
            "reason": reason,
        })
        if verdict != "BLOCK":
            return None

        return (
            f"[SOUNDNESS CHECK] Order blocked by LLM judge:\n"
            f"  - {reason}\n"
            f"Please review your tool call against the user instructions and the conversation history.\n"
            f"Fix the issue or pass override=true if you believe this is a mistake."
        )

    def _check_soundness_harness(self, tool_call_name: str, tool_call_args: dict, trajectory: Optional[list]) -> Optional[str]:
        """Delegate soundness check to the NL constraint harness."""
        result = self.soundness_harness.check_soundness(tool_call_name, tool_call_args, trajectory)
        # Merge harness call log into our own
        self.soundness_call_log = self.soundness_harness.soundness_call_log
        return result

    def observe_tool_response(self, tool_call_name: str, tool_call_args: dict, tool_response: str) -> None:
        """Forward tool responses to the harness memory store (no-op if not in harness mode)."""
        if self.soundness_harness:
            self.soundness_harness.observe_tool_response(tool_call_name, tool_call_args, tool_response)

    def check_on_stop(self, trajectory: list, remaining: int = 0, new_orders: list = (), old_orders: list = ()) -> Optional[str]:
        """
        Called when the agent decides to stop. Runs a full completeness check.
        Returns feedback string if bookings are incomplete, None if all good.
        """
        if self.completeness_mode == "off":
            return None
        missing = check_completeness(
            new_orders=new_orders,
            old_orders=old_orders,
            constraints=self.completeness_constraints,
            environment=self.environment,
        )
        if not missing:
            return None

        lines = ["\n\n[COMPLETENESS CHECK]\nSome bookings appear to still be incomplete:"]
        lines.extend(f"  - {m}" for m in missing)
        lines.append(
            "Please review and complete the missing bookings before ending the conversation. "
            f"This message will appear {remaining} more time(s) before your stop is accepted unconditionally."
        )
        return "\n".join(lines)


def create_verifier(
    task: Task,
    llm_model: Optional[str] = None,
    language: str = "english",
    constraints_file: Optional[str] = None,
    soundness_mode: str = "off",
    completeness_mode: str = "on",
    solo_user_message: Optional[str] = None,
) -> OTAVerifier:
    """
    Build an OTAVerifier for a task.

    Args:
        soundness_mode: "llm" for LLM judge, "harness" for NL constraint harness, "off" to disable.
        completeness_mode: "on" to check final order completeness at stop, "off" to disable.
    """
    environment = task.environment or {}
    completeness_constraints = _load_constraints(task.id, llm_model, constraints_file)
    if completeness_constraints is None and completeness_mode != "off":
        raise FileNotFoundError(
            f"Task {task.id}: no pre-computed completeness constraints found. "
            f"Run the pre-extraction script first."
        )

    soundness_judge = None
    soundness_harness = None
    if soundness_mode == "llm":
        try:
            judge_config = SoundnessJudgeConfig(
                llm_model=llm_model,
                llm_args=dict(model_configs.get(llm_model, model_configs.get("default", {}))),
                language=language,
            )
            soundness_judge = SoundnessJudge(
                user_instruction=task.instructions,
                config=judge_config,
            )
        except Exception as e:
            raise RuntimeError(
                f"Task {task.id}: failed to create soundness judge ({llm_model}): {e}"
            ) from e
        logger.info("Task %s: soundness judge enabled (%s)", task.id, llm_model)
    elif soundness_mode == "harness":
        from vita.domains.ota.soundness_judge_harness import create_harness
        soundness_harness = create_harness(task, llm_model=llm_model, language=language)
        logger.info("Task %s: soundness harness enabled (%s)", task.id, llm_model)

    return OTAVerifier(completeness_constraints, environment, soundness_mode=soundness_mode, completeness_mode=completeness_mode, soundness_judge=soundness_judge, soundness_harness=soundness_harness)


def _load_constraints(
    task_id: str,
    llm_model: Optional[str],
    constraints_file: Optional[str],
) -> Optional[CompletenessConstraints]:
    """Load pre-computed completeness constraints from disk."""
    if constraints_file is None:
        return None

    constraints_path = Path(constraints_file)
    if not constraints_path.exists():
        return None

    try:
        with open(constraints_path) as f:
            all_constraints = json.load(f)
        if task_id in all_constraints:
            raw = all_constraints[task_id]
            if "_error" not in raw:
                c = CompletenessConstraints.model_validate(raw)
                logger.info("Task %s: loaded completeness constraints", task_id)
                return c
    except Exception as e:
        logger.warning("Failed to load completeness constraints: %s", e)
    return None
