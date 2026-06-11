"""End-to-end pipeline CLI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from generate_glue import generate_glue
from generate_runner import generate_runner
from generate_spec import AutoInputs, generate_spec_auto
from lean_project import LeanProject
from manifest import Manifest

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Policy → Lean spec + glue pipeline")
    p.add_argument("--policy", required=True, type=Path)
    p.add_argument("--tools", required=True, type=Path)
    p.add_argument("--user-tools", type=Path, default=None,
                   help="Optional file with user/device-facing helper tools or "
                        "tool result formats. Staged into inputs/user_tools.py "
                        "for auto spec generation.")
    p.add_argument("--workflow", type=Path, default=None)
    p.add_argument("--db-schema", type=Path, default=None,
                   help="Optional file describing the runtime DB schema "
                        "(.md/.json/.py/.txt). Staged into inputs/ so the "
                        "agent can match Lean field names + json keys to "
                        "the actual DB columns.")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Lean project directory (manifest.json + .lean files)")
    p.add_argument("--glue-out", type=Path, default=None,
                   help="Where to write the rendered glue .py (default: skip)")
    p.add_argument("--skip-spec", action="store_true",
                   help="Reuse existing PolicyChecker.lean + manifest.json")
    p.add_argument("--skip-runner", action="store_true")
    p.add_argument("--skip-glue", action="store_true")
    p.add_argument("--skip-post", action="store_true",
                   help="Skip the POST-rule phase in spec generation")
    p.add_argument("--retries", type=int, default=5,
                   help="Max retries per build-fix loop")
    p.add_argument("--batch-size", type=int, default=3,
                   help="Rules per LLM call in the per-rule phase (default 3; "
                        "set to 1 to disable batching)")
    p.add_argument("--no-auto-fix", action="store_true",
                   help="Do not hand a failing LeanMain.lean to the agent; "
                        "just leave it red and exit.")
    p.add_argument("--model", type=str, default=None,
                   help="Override Copilot model (default: env TAU2_COPILOT_MODEL or gpt-5)")
    p.add_argument("--verbose", "-v", action="count", default=0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(level=max(level, logging.DEBUG),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    proj = LeanProject(args.out_dir)
    proj.bootstrap()

    if not args.skip_spec:
        auto_inputs = AutoInputs(
            policy_md=args.policy,
            tools_py=args.tools,
            user_tools_py=args.user_tools,
            workflow_md=args.workflow,
            db_schema=args.db_schema,
        )
        manifest = generate_spec_auto(proj, auto_inputs, model=args.model)
    else:
        if not proj.manifest_path.exists():
            logger.error("--skip-spec set but %s missing", proj.manifest_path)
            return 2
        manifest = Manifest.load(proj.manifest_path)
        logger.warning("re-using existing manifest (%d rules, %d stuck)",
                       len(manifest.rules), manifest.stuck_count())

    if not args.skip_runner:
        generate_runner(
            proj,
            build=True,
            auto_fix=not args.no_auto_fix,
            model=args.model,
        )

    if not args.skip_glue:
        if args.glue_out is None:
            logger.warning("no --glue-out specified; skipping glue render")
        else:
            generate_glue(proj.manifest_path, args.glue_out)

    logger.warning(
        "pipeline complete: rules=%d, stuck=%d, manifest=%s",
        len(manifest.rules), manifest.stuck_count(), proj.manifest_path,
    )
    return 0


if __name__ == "__main__": 
    sys.exit(main())
