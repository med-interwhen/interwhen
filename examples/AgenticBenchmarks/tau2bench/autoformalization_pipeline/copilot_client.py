"""Thin synchronous wrapper around the GitHub Copilot CLI.

Override the model via ``TAU2_COPILOT_MODEL`` (default ``gpt-5``).
Override the binary via ``TAU2_COPILOT_BIN``.
Set ``TAU2_COPILOT_DRY_RUN=1`` to skip the call and return a stub.

"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Optional

_INLINE_PROMPT_LIMIT = 60_000

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("TAU2_COPILOT_MODEL", "claude-opus-4.7")
COPILOT_BIN = os.environ.get("TAU2_COPILOT_BIN", "copilot")

def _format_messages(messages: list[dict]) -> str:
    """Flatten OpenAI-style messages into a single tagged prompt.

    The CLI takes a single ``-p`` string rather than a structured chat
    history, so we render the role-tagged messages into one text block.
    """
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        parts.append(f"<<<{role}>>>\n{content}\n<<</{role}>>>")
    return "\n\n".join(parts)


def _resolve_bin() -> str:
    path = shutil.which(COPILOT_BIN)
    if not path:
        raise RuntimeError(
            f"Could not find '{COPILOT_BIN}' on PATH. Install the Copilot CLI "
            "via 'curl -fsSL https://gh.io/copilot-install | bash' and ensure "
            "~/.local/bin is in PATH."
        )
    return path


def chat(
    messages: list[dict],
    *,
    model: Optional[str] = None,
    timeout_s: float = 600.0,
) -> str:
    """Run a one-shot chat turn through the Copilot CLI.

    Returns the assistant's stdout text.  Blocking; spawns a subprocess
    per call. 
    """
    if os.environ.get("TAU2_COPILOT_DRY_RUN") == "1":
        logger.warning("TAU2_COPILOT_DRY_RUN=1 — returning stub response")
        return "[dry-run stub]"

    prompt = _format_messages(messages)
    model = model or DEFAULT_MODEL
    binary = _resolve_bin()

    base_cmd = [
        binary,
        "--model", model,
        "--allow-all-tools",
        "--deny-tool", "shell",
        "--deny-tool", "write",
    ]

    stdin_input: Optional[str] = None
    if len(prompt) <= _INLINE_PROMPT_LIMIT:
        cmd = base_cmd + ["-p", prompt]
    else:
        # Pipe the prompt via stdin and use an empty -p so the CLI runs non-interactive.
        cmd = base_cmd + ["-p", ""]
        stdin_input = prompt
        logger.info("copilot.chat: prompt %d chars → piping via stdin",
                    len(prompt))

    logger.info("copilot.chat → model=%s prompt_chars=%d", model, len(prompt))
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            input=stdin_input,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"copilot CLI timed out after {timeout_s}s") from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"copilot CLI failed (exit {proc.returncode}):\n"
            f"--- stderr ---\n{proc.stderr}\n"
            f"--- stdout ---\n{proc.stdout}"
        )
    return proc.stdout


__all__ = ["chat", "DEFAULT_MODEL", "COPILOT_BIN"]
