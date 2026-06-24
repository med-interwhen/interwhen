"""
Runtime memory store for the NL soundness judge harness.

After each read-type tool call, an SLM distills the response into
facts relevant to the extracted constraints. This accumulated memory
is used during judgment of write tool calls.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from vita.config import models
from vita.data_model.message import SystemMessage, UserMessage
from vita.domains.ota.soundness_judge_harness.schema import (
    Constraint,
    ExtractedConstraintSet,
    TaskMemory,
)
from vita.prompts import get_prompts
from vita.utils.llm_utils import generate

logger = logging.getLogger(__name__)

WRITE_PREFIXES = ("create_", "cancel_", "modify_", "pay_")


class MemoryWriter:
    """
    Manages the per-task memory store during simulation.

    On each non-write tool call, asks an SLM to distill relevant facts
    from the tool response into the memory.
    """

    def __init__(
        self,
        constraints: ExtractedConstraintSet,
        llm_model: str,
        llm_args: Optional[dict] = None,
        max_response_len: int = 3000,
        user_profile: Optional[str] = None,
        language: str = "english",
    ):
        self.constraints = constraints
        self.llm_model = llm_model
        self.llm_args = llm_args or {}
        self.max_response_len = max_response_len
        self.memory = TaskMemory()
        self.language = language

        # Seed memory with user profile
        if user_profile:
            self.memory.append(source_tool="user_profile", summary=user_profile)

        # Pre-render constraints text for the prompt
        self._constraints_text = self._render_constraints()

    def _render_constraints(self) -> str:
        lines = []
        for c in self.constraints.constraints:
            cat = c.category.value
            etype = f" [{c.entity_type.value}]" if c.entity_type else ""
            lines.append(f"- [{c.id}] ({cat}{etype}) {c.description}")
        return "\n".join(lines)

    def is_read_call(self, tool_call_name: str) -> bool:
        """Returns True if this tool call is a read (non-write) operation."""
        return not any(tool_call_name.startswith(p) for p in WRITE_PREFIXES)

    def process_tool_response(
        self,
        tool_call_name: str,
        tool_call_args: dict,
        tool_response: str,
    ) -> None:
        """
        Process a read tool call response and update memory if relevant.

        Should be called for every non-write tool call after execution.
        """
        if not self.is_read_call(tool_call_name):
            return

        # Truncate long responses
        response_text = tool_response
        if len(response_text) > self.max_response_len:
            response_text = response_text[: self.max_response_len] + "... (truncated)"

        system_content = get_prompts(self.language).harness_memory_writer_template.format(
            constraints_text=self._constraints_text,
        )

        user_content = (
            f"## Tool Call\n"
            f"{tool_call_name}({json.dumps(tool_call_args, ensure_ascii=False)})\n\n"
            f"## Response\n{response_text}"
        )

        messages = [
            SystemMessage(role="system", content=system_content),
            UserMessage(role="user", content=user_content),
        ]

        kwargs = dict(self.llm_args)
        kwargs.setdefault("temperature", 0)

        try:
            response = generate(self.llm_model, messages, enable_think=False, **kwargs)
            summary = response.content if hasattr(response, "content") else str(response)
            summary = summary.strip()

            if summary and summary != "NOTHING_RELEVANT":
                self.memory.append(source_tool=tool_call_name, summary=summary)
                logger.debug("Memory updated from %s: %s", tool_call_name, summary[:80])
        except Exception as e:
            logger.warning("Memory writer failed for %s: %s", tool_call_name, e)

    def get_memory(self) -> TaskMemory:
        """Return the current memory state."""
        return self.memory

    def render_memory(self) -> str:
        """Render memory as text for the judge."""
        return self.memory.render()
