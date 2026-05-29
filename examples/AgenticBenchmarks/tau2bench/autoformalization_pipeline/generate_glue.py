"""Stage 3 — render the per-domain ``<domain>_glue_spec.py`` from
``manifest.json``."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

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


def render_glue(manifest: Manifest) -> str:
    env = _env()
    template = env.get_template("glue_spec.py.j2")
    return template.render(manifest=manifest)


def generate_glue(manifest_path: Path, out_path: Path) -> Path:
    manifest = Manifest.load(manifest_path)
    src = render_glue(manifest)
    out_path.write_text(src)
    logger.warning("glue file rendered → %s (%d bytes)", out_path, len(src))
    return out_path


__all__ = ["generate_glue", "render_glue"]
