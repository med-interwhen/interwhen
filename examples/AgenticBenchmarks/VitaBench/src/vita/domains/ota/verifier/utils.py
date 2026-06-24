"""Shared utility functions for the OTA verifier."""

import json
import re


def _extract_json(text: str) -> dict | list | None:
    """Extract JSON object/array from text that may have thinking prefix.

    When the LLM emits multiple JSON blocks (e.g. it reconsiders and outputs a
    corrected version), we take the *last* valid JSON object.
    """
    # Strip <think>...</think> blocks
    text = text.split("</think>")[-1].strip()

    # Try all ```json ... ``` fences, take the last one that parses
    fence_matches = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    for candidate in reversed(fence_matches):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Fallback: find balanced top-level { } blocks, take the last valid one
    candidates: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start != -1:
                candidates.append(text[start:i + 1])
                start = -1

    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def _extract_tool_history(trajectory: list) -> list[dict]:
    """
    Extract tool calls paired with their responses from a trajectory.
    """
    tool_calls_by_id: dict[str, dict] = {}

    for msg in trajectory:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_by_id[tc.id] = {
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "tool_call_id": tc.id,
                    "response": "",
                }

        if hasattr(msg, "role") and getattr(msg, "role", None) == "tool":
            msg_id = getattr(msg, "id", "")
            if msg_id in tool_calls_by_id:
                tool_calls_by_id[msg_id]["response"] = getattr(msg, "content", "") or ""

        if hasattr(msg, "tool_messages"):
            for tm in msg.tool_messages:
                tm_id = getattr(tm, "id", "")
                if tm_id in tool_calls_by_id:
                    tool_calls_by_id[tm_id]["response"] = getattr(tm, "content", "") or ""

    return list(tool_calls_by_id.values())
