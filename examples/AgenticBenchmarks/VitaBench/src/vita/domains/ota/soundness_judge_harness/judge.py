"""
Judgment module for the NL soundness judge harness.

Given a write tool call, evaluates each relevant constraint individually
against the accumulated memory. One LLM call per constraint, results
ANDed in code (any violation → overall violation).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from vita.data_model.message import SystemMessage, UserMessage
from vita.domains.ota.soundness_judge_harness.schema import (
    Constraint,
    ConstraintJudgment,
    ConstraintVerdict,
    EntityType,
    ExtractedConstraintSet,
    JudgmentResult,
    TaskMemory,
)
from vita.prompts import get_prompts
from vita.utils.llm_utils import generate

logger = logging.getLogger(__name__)

# Map tool call name prefixes to entity types
_TOOL_TO_ENTITY = {
    "hotel": EntityType.HOTEL,
    "flight": EntityType.FLIGHT,
    "train": EntityType.TRAIN,
    "attraction": EntityType.ATTRACTION,
}


def _entity_type_from_tool(tool_call_name: str) -> Optional[EntityType]:
    """Infer entity type from tool call name."""
    for key, etype in _TOOL_TO_ENTITY.items():
        if key in tool_call_name:
            return etype
    return None


def _parse_single_judgment(text: str) -> Optional[dict]:
    """Parse a single JSON verdict from LLM response."""
    text = text.split("</think>")[-1].strip()

    # Try fenced blocks
    fence_matches = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    for candidate in reversed(fence_matches):
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    # Try raw JSON object
    brace_start = text.find("{")
    if brace_start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(brace_start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class ConstraintJudge:
    """
    Evaluates write tool calls against extracted constraints using memory context.

    Makes one LLM call per relevant constraint, then ANDs results:
    any single VIOLATED → overall violation.
    """

    def __init__(
        self,
        constraints: ExtractedConstraintSet,
        llm_model: str,
        llm_args: Optional[dict] = None,
        language: str = "english",
    ):
        self.constraints = constraints
        self.llm_model = llm_model
        self.llm_args = llm_args or {}
        self.language = language

    def _get_relevant_constraints(self, tool_call_name: str) -> list[Constraint]:
        """Filter constraints to those relevant to the tool call's entity type."""
        entity_type = _entity_type_from_tool(tool_call_name)
        relevant = []
        for c in self.constraints.constraints:
            # Include if: constraint has no entity_type (cross-cutting),
            # or matches the tool call's entity type
            if c.entity_type is None or c.entity_type == entity_type:
                relevant.append(c)
        return relevant

    def judge(
        self,
        tool_call_name: str,
        tool_call_args: dict,
        memory: TaskMemory,
    ) -> JudgmentResult:
        """
        Judge a write tool call against relevant constraints.

        One LLM call per constraint, results ANDed in code.
        """
        relevant = self._get_relevant_constraints(tool_call_name)

        if not relevant:
            return JudgmentResult(
                tool_call_name=tool_call_name,
                tool_call_args=tool_call_args,
                judgments=[],
            )

        memory_text = memory.render()
        current_call = f"{tool_call_name}({json.dumps(tool_call_args, ensure_ascii=False)})"

        judgments: list[ConstraintJudgment] = []
        for constraint in relevant:
            judgment = self._judge_single(constraint, current_call, memory_text)
            judgments.append(judgment)

        return JudgmentResult(
            tool_call_name=tool_call_name,
            tool_call_args=tool_call_args,
            judgments=judgments,
        )

    def _judge_single(
        self,
        constraint: Constraint,
        current_call: str,
        memory_text: str,
    ) -> ConstraintJudgment:
        """Evaluate a single constraint against the tool call."""
        cat = constraint.category.value
        etype = f" [{constraint.entity_type.value}]" if constraint.entity_type else ""
        constraint_text = f"[{constraint.id}] ({cat}{etype}) {constraint.description}"

        user_content = (
            f"## Constraint\n{constraint_text}\n\n"
            f"## Memory (observed facts from prior tool calls)\n{memory_text}\n\n"
            f"## Write Tool Call to Evaluate\n{current_call}"
        )

        messages = [
            SystemMessage(role="system", content=get_prompts(self.language).harness_soundness_judge_template),
            UserMessage(role="user", content=user_content),
        ]

        kwargs = dict(self.llm_args)
        kwargs.setdefault("temperature", 0)

        try:
            response = generate(self.llm_model, messages, enable_think=True, **kwargs)
            raw = response.content if hasattr(response, "content") else str(response)
            return self._parse_response(constraint.id, raw)
        except Exception as e:
            logger.warning("Judge call failed for constraint %s: %s", constraint.id, e)
            return ConstraintJudgment(
                constraint_id=constraint.id,
                verdict=ConstraintVerdict.UNDETERMINED,
                reasoning=f"Judge call failed: {e}",
            )

    def _parse_response(self, constraint_id: str, raw: str) -> ConstraintJudgment:
        """Parse single-constraint LLM response."""
        parsed = _parse_single_judgment(raw)
        if parsed is None:
            logger.warning("Failed to parse judge response for %s", constraint_id)
            return ConstraintJudgment(
                constraint_id=constraint_id,
                verdict=ConstraintVerdict.UNDETERMINED,
                reasoning="Parse failure — could not interpret judge response",
            )

        verdict_str = parsed.get("verdict", "undetermined").lower()
        reasoning = parsed.get("reasoning", "")

        try:
            verdict = ConstraintVerdict(verdict_str)
        except ValueError:
            verdict = ConstraintVerdict.UNDETERMINED

        return ConstraintJudgment(
            constraint_id=constraint_id,
            verdict=verdict,
            reasoning=reasoning,
        )
