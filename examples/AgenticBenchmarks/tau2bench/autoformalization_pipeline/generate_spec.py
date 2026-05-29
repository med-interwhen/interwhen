"""Auto mode.

We hand the agent a Lean project, the policy/tools inputs, and a single
prompt; it writes ``PolicyChecker.lean``, runs ``lake build`` until
green, and emits ``manifest.json``. 
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from copilot_client import COPILOT_BIN, DEFAULT_MODEL, _resolve_bin
from lean_project import LeanProject
from manifest import Manifest

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class AutoInputs:
    policy_md: Path
    tools_py: Path
    user_tools_py: Optional[Path] = None
    workflow_md: Optional[Path] = None
    db_schema: Optional[Path] = None


def _stage_inputs(proj: LeanProject, inputs: AutoInputs) -> Path:
    """Copy the policy/tools/workflow inputs under ``<proj>/inputs/``
    so the agent can read them from one well-known location.
    """
    in_dir = proj.root / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(inputs.policy_md, in_dir / "policy.md")
    shutil.copy(inputs.tools_py, in_dir / "tools.py")
    if inputs.user_tools_py and inputs.user_tools_py.exists():
        shutil.copy(inputs.user_tools_py, in_dir / "user_tools.py")
    if inputs.workflow_md and inputs.workflow_md.exists():
        shutil.copy(inputs.workflow_md, in_dir / "workflow.md")
    if inputs.db_schema and inputs.db_schema.exists():
        # Preserve original extension so the agent knows how to read it.
        suffix = inputs.db_schema.suffix or ".txt"
        shutil.copy(inputs.db_schema, in_dir / f"db_schema{suffix}")
    return in_dir


def _build_prompt(proj: LeanProject) -> str:
    auto = (_PROMPTS_DIR / "auto.md").read_text()
    spec_rules = (_PROMPTS_DIR / "spec_initial.md").read_text()
    spec_post = (_PROMPTS_DIR / "spec_post.md").read_text()
    return (
        auto
        .replace("{{PROJECT_DIR}}", str(proj.root))
        .replace("{{SPEC_RULES}}", spec_rules)
        .replace("{{SPEC_POST}}", spec_post)
    )


def generate_spec_auto(
    proj: LeanProject,
    inputs: AutoInputs,
    *,
    model: Optional[str] = None,
    timeout_s: float = 7200.0,
) -> Manifest:
    """Run the agent loop. Blocks until the agent exits."""
    proj.bootstrap()
    _stage_inputs(proj, inputs)
    prompt = _build_prompt(proj)
    binary = _resolve_bin()
    model = model or DEFAULT_MODEL

    # Write the long prompt to a file the agent can read with its own
    # tools; pass only a short bootstrap instruction via -p.  This
    # leaves stdin free for the agent's interactive tooling and avoids
    # the ARG_MAX limit on `-p`.
    prompt_path = proj.root / "AGENT_PROMPT.md"
    prompt_path.write_text(prompt)

    bootstrap = (
        f"Read {prompt_path} and execute it end to end. Edit files and "
        f"run `lake build` from {proj.root} until everything is green. "
        f"When done, the project directory must contain a working "
        f"PolicyChecker.lean and manifest.json."
    )

    cmd = [
        binary,
        "-p", bootstrap,
        "--model", model,
        "--allow-all-tools",
        "--add-dir", str(proj.root),
        "-C", str(proj.root),
    ]
    logger.warning("auto: launching copilot agent → model=%s, project=%s",
                   model, proj.root)
    logger.warning("auto: prompt at %s; tail %s/PolicyChecker.lean for progress",
                   prompt_path, proj.root)

    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            timeout=timeout_s,
            stdout=None,
            stderr=None,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"copilot agent timed out after {timeout_s}s") from exc

    if proc.returncode != 0:
        raise RuntimeError(f"copilot agent exited {proc.returncode}")

    if not proj.manifest_path.exists():
        raise RuntimeError(
            f"agent finished but {proj.manifest_path} was not created. "
            "Inspect the project dir and the conversation transcript."
        )
    manifest = Manifest.load(proj.manifest_path)
    logger.warning(
        "auto complete: %d rules, %d stuck",
        len(manifest.rules), manifest.stuck_count(),
    )
    return manifest


__all__ = ["generate_spec_auto", "AutoInputs"]
