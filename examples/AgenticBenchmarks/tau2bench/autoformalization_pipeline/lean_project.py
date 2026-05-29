"""Lean project bootstrapping and ``lake build`` driver.

Stage 1 needs a clean Lean project with mathlib already cached, plus an
in-memory representation of ``PolicyChecker.lean`` it can append to and
rebuild after each LLM turn.  This module owns the project on-disk and
exposes a small API:

* :func:`bootstrap`          — create ``lakefile.toml`` / ``lean-toolchain``.
* :class:`LeanProject.build` — run ``lake build`` and return stderr on failure.
* :class:`LeanProject.write_spec` — atomically replace ``PolicyChecker.lean``.

We deliberately reuse the existing verifier's ``lakefile.toml`` /
``lean-toolchain`` settings so that the mathlib cache is shared.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


_LAKEFILE_TEMPLATE = """\
name = "policychecker"
version = "0.1.0"
defaultTargets = ["policychecker"]

[leanOptions]
pp.unicode.fun = true
relaxedAutoImplicit = false
maxSynthPendingDepth = 3

[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "v4.30.0-rc2"

[[lean_lib]]
name = "PolicyChecker"
roots = ["PolicyChecker"]

[[lean_exe]]
name = "policychecker"
root = "LeanMain"
supportInterpreter = true
"""

_LEAN_TOOLCHAIN = "leanprover/lean4:v4.30.0-rc2\n"


@dataclass
class BuildResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int

    def short_error(self, max_chars: int = 4000) -> str:
        msg = self.stderr or self.stdout
        if len(msg) <= max_chars:
            return msg
        return msg[:max_chars] + "\n...[truncated]..."


class LeanProject:
    """Owns a Lean project directory and runs ``lake build`` on demand."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.spec_path = self.root / "PolicyChecker.lean"
        self.runner_path = self.root / "LeanMain.lean"
        self.manifest_path = self.root / "manifest.json"

    # ---- Bootstrap ---------------------------------------------------

    def bootstrap(self, seed_spec: str = "", seed_runner: str = "") -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "lakefile.toml").write_text(_LAKEFILE_TEMPLATE)
        (self.root / "lean-toolchain").write_text(_LEAN_TOOLCHAIN)
        if not self.spec_path.exists():
            self.spec_path.write_text(seed_spec or "import Mathlib\n")
        if not self.runner_path.exists():
            self.runner_path.write_text(
                seed_runner or "def main : IO Unit := pure ()\n"
            )

    # ---- Spec IO -----------------------------------------------------

    def read_spec(self) -> str:
        return self.spec_path.read_text() if self.spec_path.exists() else ""

    def write_spec(self, src: str) -> None:
        tmp = self.spec_path.with_suffix(".lean.tmp")
        tmp.write_text(src)
        tmp.replace(self.spec_path)

    def write_runner(self, src: str) -> None:
        tmp = self.runner_path.with_suffix(".lean.tmp")
        tmp.write_text(src)
        tmp.replace(self.runner_path)

    # Build

    def cache_mathlib(self, timeout_s: float = 1800.0) -> BuildResult:
        """One-time prefetch of mathlib oleans (~10 min if cold)."""
        return self._run(["lake", "exe", "cache", "get"], timeout_s)

    def build(self, timeout_s: float = 1200.0) -> BuildResult:
        return self._run(["lake", "build"], timeout_s)

    def _run(self, cmd: list[str], timeout_s: float) -> BuildResult:
        logger.info("LeanProject: running %s (cwd=%s)", cmd, self.root)
        try:
            cp = subprocess.run(
                cmd,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            return BuildResult(
                ok=False, stdout=e.stdout or "", stderr=f"timeout after {timeout_s}s",
                returncode=-1,
            )
        except FileNotFoundError as e:
            return BuildResult(
                ok=False, stdout="", stderr=f"lake binary not found: {e!r}",
                returncode=-1,
            )
        return BuildResult(
            ok=(cp.returncode == 0), stdout=cp.stdout, stderr=cp.stderr,
            returncode=cp.returncode,
        )


def bootstrap_from_existing(
    target_root: Path,
    *,
    reference_root: Path | None = None,
) -> LeanProject:
    """Convenience: create the project, optionally copying lake/manifest from a
    sibling verifier checkout so the mathlib build cache is shared."""
    proj = LeanProject(target_root)
    proj.bootstrap()
    if reference_root is not None:
        for shared in ("lake-manifest.json",):
            src = reference_root / shared
            if src.exists():
                shutil.copy2(src, target_root / shared)
    return proj
