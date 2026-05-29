"""Stage 2 — render ``LeanMain.lean`` from ``manifest.json``."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from copilot_client import COPILOT_BIN, DEFAULT_MODEL, _resolve_bin
from lean_project import LeanProject
from manifest import Manifest

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _camel_to_snake(name: str) -> str:
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    env.globals["camel_to_snake"] = _camel_to_snake
    return env


def render_runner(manifest: Manifest) -> str:
    env = _env()
    template = env.get_template("LeanMain.lean.j2")
    enum_names = {e.name for e in manifest.enums}
    return template.render(manifest=manifest, enum_names=enum_names)


def write_runner(proj: LeanProject, manifest: Manifest) -> Path:
    src = render_runner(manifest)
    proj.write_runner(src)
    logger.warning("LeanMain.lean rendered (%d bytes)", len(src))
    return proj.runner_path


_FIX_PROMPT = (
    "LeanMain.lean fails to build against PolicyChecker.lean. "
    "Read both files plus manifest.json (which describes what fields and "
    "types the spec exposes), then make `lake build` pass. "
    "PREFER fixing LeanMain.lean. Do NOT modify manifest.json, "
    "lakefile.toml, or lean-toolchain. "
    "However, if the build errors are clearly inside PolicyChecker.lean "
    "(e.g. `PolicyChecker.lean:NNN:` lines pointing at theorem proofs, "
    "tactic failures, or missing instances on spec-side definitions), "
    "you MAY edit PolicyChecker.lean to fix them — the prior auto stage "
    "sometimes ships broken proofs. When patching proofs, prefer "
    "minimal tactic rewrites (e.g. replace `cases h : expr with` + "
    "`rw [h] at hyp` with `generalize hX : expr = v; cases v`) over "
    "rewriting the whole theorem. Never introduce `sorry` or "
    "`native_decide`; if a proof is unsalvageable, replace the rule "
    "with the stuck-rule stub (spec = True, check = true, trivial "
    "iff). "
    "Common LeanMain issues to check: `String.ofByteArray` API may now "
    "require `(bytes, validateUtf8)`; `Hyp` may need a default "
    "constructor or an Inhabited instance; helpers like `parseToolCall` "
    "may be missing — define them inline. "
    "When the build is green, run `lake build` ONE FINAL TIME from a "
    "fresh shell to confirm exit code 0, then write a one-line summary."
)


def _agent_fix_runner(
    proj: LeanProject, *, model: str | None, timeout_s: float = 1800.0,
) -> bool:
    """Hand the failing runner to the Copilot agent for repair.

    Returns True iff `lake build` is green after the agent's session.
    """
    binary = _resolve_bin()
    model = model or DEFAULT_MODEL
    cmd = [
        binary,
        "-p", _FIX_PROMPT,
        "--model", model,
        "--allow-all-tools",
        "--add-dir", str(proj.root),
        "-C", str(proj.root),
    ]
    logger.warning("LeanMain.lean: handing off to copilot agent for repair")
    try:
        proc = subprocess.run(cmd, check=False, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        logger.error("agent runner-repair timed out after %ds", int(timeout_s))
        return False
    if proc.returncode != 0:
        logger.error("agent exited %d during runner repair", proc.returncode)
        return False
    res = proj.build()
    if res.ok:
        logger.warning("LeanMain.lean builds after agent repair")
    else:
        logger.error("LeanMain.lean still red after agent repair:\n%s",
                     res.short_error())
    return res.ok


def generate_runner(
    proj: LeanProject,
    *,
    build: bool = True,
    auto_fix: bool = True,
    model: str | None = None,
) -> Path:
    manifest = Manifest.load(proj.manifest_path)
    path = write_runner(proj, manifest)
    if not build:
        return path
    res = proj.build()
    if res.ok:
        logger.warning("LeanMain.lean builds")
        return path
    logger.error("LeanMain.lean build failed:\n%s", res.short_error())
    if auto_fix:
        _agent_fix_runner(proj, model=model)
    return path


__all__ = ["generate_runner", "render_runner", "write_runner"]
