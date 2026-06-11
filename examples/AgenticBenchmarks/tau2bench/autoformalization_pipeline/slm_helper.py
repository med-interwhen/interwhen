"""
SLM helper - thin wrapper around a small language model for extracting
structured facts from conversation history during verification.

At verification time the verifier may need to know things like:
  "Did the user explicitly confirm this action?"
  "What reason did the user give for cancellation?"
  "How many passengers did the user mention?"

These are hard to extract with regex but trivial for a small LM.
The SLM is called with a focused prompt + the recent conversation and
returns a short structured answer.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SLM client – uses the same vLLM / OpenAI‑compatible endpoint as the agent
# but with a small, fast model.  Falls back to the main model if no separate
# SLM endpoint is configured.
# ---------------------------------------------------------------------------

_SLM_BASE = os.environ.get("SLM_API_BASE", os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"))
_SLM_KEY = os.environ.get("SLM_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy"))
_SLM_MODEL = os.environ.get("SLM_MODEL", os.environ.get("OPENAI_MODEL", ""))

_resolved_model: str | None = None


def _get_model() -> str:
    """Resolve the SLM model name, auto-detecting from the endpoint if needed."""
    global _resolved_model
    if _resolved_model:
        return _resolved_model
    if _SLM_MODEL:
        _resolved_model = _SLM_MODEL
        return _resolved_model
    # Auto-detect from vLLM /v1/models endpoint
    try:
        import requests
        base = _SLM_BASE.rstrip("/")
        if base.endswith("/v1"):
            models_url = base + "/models"
        else:
            models_url = base + "/v1/models"
        resp = requests.get(models_url, timeout=5)
        data = resp.json()
        if "data" in data and data["data"]:
            _resolved_model = data["data"][0]["id"]
            logger.info("SLM auto-detected model: %s", _resolved_model)
            return _resolved_model
    except Exception as e:
        logger.warning("SLM model auto-detect failed: %s", e)
    _resolved_model = "default"
    return _resolved_model


def _get_client():
    """Lazy-init an OpenAI client pointed at the SLM endpoint."""
    from openai import OpenAI
    return OpenAI(base_url=_SLM_BASE, api_key=_SLM_KEY)


def _parse_slm_answer(raw: str) -> str:
    """
    Parse the SLM's raw output to extract the actual answer.

    Thinking models (Qwen3, etc.) may output reasoning text before the answer.
    This function handles:
    - <think>...</think> tags
    - Multi-line reasoning ending with the actual answer on the last line(s)
    """
    import re

    text = raw.strip()

    # 1. Strip <think>...</think> blocks
    if "<think>" in text:
        parts = text.split("</think>")
        if len(parts) > 1:
            text = parts[-1].strip()
        else:
            text = text.split("<think>")[-1].strip()

    # 2. If the result is short enough, return as-is
    if len(text) <= 30:
        return text

    # 3. For longer outputs (reasoning models), try to find the actual answer
    lines = text.strip().split("\n")

    # Check last few lines for a clean yes/no or short answer
    for line in reversed(lines[-5:]):
        clean = line.strip().lower().rstrip(".")
        if clean in ("yes", "no"):
            return clean

    # Check for yes/no/value after common markers
    for marker in ["answer:", "result:", "final answer:", "**answer**:", "**"]:
        idx = text.lower().rfind(marker)
        if idx >= 0:
            after = text[idx + len(marker):].strip().strip("*").strip()
            if after:
                # Take first line/word
                first_line = after.split("\n")[0].strip()
                if len(first_line) <= 50:
                    return first_line

    # 4. Fallback: return the last non-empty line
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped

    return text


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks so the SLM only sees user-visible text."""
    import re
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned.startswith("<think>"):
        cleaned = ""
    return cleaned


def slm_extract(question: str, conversation: list[dict], max_tokens: int = 256) -> str:
    """
    Ask the SLM a yes/no or short-answer question about the conversation.

    Parameters
    ----------
    question : str
        A focused extraction question, e.g.
        "Did the user explicitly say 'yes' to confirm the action?"
    conversation : list[dict]
        The recent message history (list of {role, content} dicts).
    max_tokens : int
        Cap on the SLM response length.

    Returns
    -------
    str  –  The SLM's answer (stripped).
    """
    # Build a compact transcript for the SLM
    transcript_lines = []
    for msg in conversation[-30:]:  # last 30 messages to match orchestrator window
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if content:
            # Strip thinking traces so SLM only sees user-visible text
            clean = _strip_thinking(str(content))
            if clean:
                transcript_lines.append(f"[{role}]: {clean[:500]}")
    transcript = "\n".join(transcript_lines)

    system_prompt = (
        "You are a precise information extractor. Given a conversation transcript "
        "and a question, answer the question as concisely as possible. "
        "If the answer is yes/no, reply with ONLY 'yes' or 'no'. "
        "If the answer is a value, reply with ONLY the value. "
        "Do not explain or add extra text."
    )

    try:
        client = _get_client()
        model = _get_model()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Conversation:\n{transcript}\n\nQuestion: {question}"},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        raw_answer = resp.choices[0].message.content.strip()
        answer = _parse_slm_answer(raw_answer)
        logger.debug("SLM extract Q=%s  A=%s (raw_len=%d)", question, answer, len(raw_answer))
        return answer
    except Exception as e:
        logger.warning("SLM extraction failed: %s", e)
        return ""

def slm_extract_json(question: str, conversation: list[dict], max_tokens: int = 256) -> Any:
    """Same as slm_extract but parses the answer as JSON."""
    raw = slm_extract(question + " Reply in valid JSON.", conversation, max_tokens)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
