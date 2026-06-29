"""
LLM-based soundness judge for OTA verifier.

Uses an LLM to evaluate whether a write tool call (create/pay/cancel/modify) is
consistent with the user's instructions, given the tool call history so far.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from vita.data_model.message import SystemMessage, UserMessage
from vita.prompts import get_prompts
from vita.utils.llm_utils import generate
from vita.domains.ota.verifier.utils import _extract_tool_history, _extract_json

logger = logging.getLogger(__name__)


@dataclass
class SoundnessJudgeConfig:
    """Configuration for the LLM-based soundness judge."""

    llm_model: str
    """Model name to use for judging (e.g. 'claude-sonnet-4.6')."""

    llm_args: dict = field(default_factory=dict)
    """Extra kwargs forwarded to the generate() call (temperature, max_tokens, etc.)."""

    language: str = "english"
    """Language for the prompt template ('english' or 'chinese')."""

    max_response_len: int = 2000
    """Max characters per tool response shown to the judge (longer responses are truncated)."""


def _format_tool_trace(tool_trace: list[dict], max_response_len: int = 2000) -> str:
    """Format tool trace into a readable string for the LLM."""
    if not tool_trace:
        return "(no prior tool calls)"

    lines = []
    for i, tc in enumerate(tool_trace, 1):
        lines.append(f"[{i}] {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})")
        resp = tc["response"]
        if len(resp) > max_response_len:
            resp = resp[:max_response_len] + "... (truncated)"
        lines.append(f"    → {resp}")
    return "\n".join(lines)


class SoundnessJudge:
    """
    LLM-based judge that evaluates whether a write tool call
    is consistent with the user's original instructions.
    """

    def __init__(
        self,
        user_instruction: str,
        config: SoundnessJudgeConfig,
    ):
        self.user_instruction = user_instruction
        self.config = config

        prompts = get_prompts(config.language)
        self.system_prompt = prompts.soundness_judge_template

    def judge(
        self,
        tool_call_name: str,
        tool_call_args: dict,
        trajectory: list,
    ) -> tuple[str, Optional[str]]:
        """
        Judge whether a write tool call should be allowed or blocked.

        Returns a (verdict, reason) tuple where verdict is 'ALLOW' or 'BLOCK',
        and reason is None when not applicable.
        """
        # Reuse the shared tool history extractor
        tool_trace = _extract_tool_history(trajectory)
        trace_str = _format_tool_trace(tool_trace, self.config.max_response_len)

        current_call = f"{tool_call_name}({json.dumps(tool_call_args, ensure_ascii=False)})"

        user_content = (
            f"## User Instruction\n{self.user_instruction}\n\n"
            f"## Tool Call History\n{trace_str}\n\n"
            f"## Current Tool Call (to judge)\n{current_call}"
        )

        messages = [
            SystemMessage(role="system", content=self.system_prompt),
            UserMessage(role="user", content=user_content),
        ]

        try:
            response = generate(
                model=self.config.llm_model,
                messages=messages,
                **self.config.llm_args,
            )
        except Exception as e:
            logger.warning("Soundness judge failed, allowing call: %s", e)
            return "ALLOW", None

        raw = response.content or ""
        return _parse_verdict(raw)


def _parse_verdict(raw: str) -> tuple[str, Optional[str]]:
    """Parse the LLM's JSON verdict. Defaults to ALLOW on parse failure."""

    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        logger.warning("Failed to parse LLM judge response, defaulting to ALLOW: %s", raw[:200])
        return "ALLOW", None

    verdict = parsed.get("verdict", "ALLOW").upper()
    reason = parsed.get("reason") or None
    if verdict not in ("ALLOW", "BLOCK"):
        verdict = "ALLOW"
    return verdict, reason
