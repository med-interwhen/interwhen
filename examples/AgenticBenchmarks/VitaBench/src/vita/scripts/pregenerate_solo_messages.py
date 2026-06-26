#!/usr/bin/env python3
"""
Pregenerate the first user message for solo-agent tasks and store them in a JSON file.

The output file maps task_id -> pregenerated user message string.  Pass it to the
benchmark via --solo-user-mode=file --solo-user-file=<path>.

The output path is built automatically as:
    data/user_sim_<model>_<domain>_<language>.json

Usage:
    python -m vita.scripts.pregenerate_solo_messages --domain ota --llm claude-sonnet-4-5

    # Chinese tasks
    python -m vita.scripts.pregenerate_solo_messages --domain ota --language chinese --llm claude-sonnet-4-5

    # Force-regenerate entries already present in the output file
    python -m vita.scripts.pregenerate_solo_messages --domain ota --llm claude-sonnet-4-5 --force
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from vita.data_model.tasks import Task
from vita.user.user_simulator import DummyUser
from vita.utils.utils import get_task_file_path, DATA_DIR
from vita.orchestrator.orchestrator import get_default_first_agent_message
from vita.config import models


def build_output_path(llm: str, domain: str, language: str, resolved: bool = False) -> Path:
    safe_model = llm.replace("/", "_")
    suffix = "_resolved" if resolved else ""
    return DATA_DIR / f"user_sim_{safe_model}_{domain}_{language}{suffix}.json"


def pregenerate(
        domain: str,
        language: str,
        llm: str,
        llm_args: dict,
        output: Path,
        task_ids: list[str] = None,
        max_concurrency: int = 1,
        force: bool = False,
        resolved_instructions_file: str = None,
):
    task_path = get_task_file_path(domain, language)
    with open(task_path, "r", encoding="utf-8") as fp:
        raw_tasks = json.load(fp)

    tasks = [Task.model_validate(t) for t in raw_tasks]

    # Load resolved instructions if provided
    resolved = {}
    if resolved_instructions_file:
        with open(resolved_instructions_file, "r", encoding="utf-8") as fp:
            resolved = json.load(fp)
        logger.info(f"Loaded {len(resolved)} resolved instructions from {resolved_instructions_file}")

    if task_ids is not None:
        tasks = [t for t in tasks if t.id in task_ids]
        if len(tasks) != len(task_ids):
            missing = set(task_ids) - {t.id for t in tasks}
            raise ValueError(f"Task IDs not found: {missing}")

    # Load existing output file so we can skip already-generated entries
    messages: dict = {}
    if output.exists():
        with open(output, "r", encoding="utf-8") as fp:
            messages = json.load(fp)

    needs_update = [t for t in tasks if force or t.id not in messages]
    logger.info(f"{len(needs_update)} / {len(tasks)} tasks need solo_user_message generation")

    if not needs_update:
        logger.info("Nothing to do.")
        return

    # The orchestrator sends this greeting before the first user turn
    greeting = get_default_first_agent_message(language)
    lock = threading.Lock()

    def _generate(task: Task):
        if resolved:
            entry = resolved.get(task.id)
            if entry is None:
                raise ValueError(f"Task {task.id} not found in resolved instructions file")
            ri = entry.get("resolved_instructions")
            if not ri:
                raise ValueError(f"Task {task.id} has no resolved_instructions in resolved file")
            instructions = ri
        else:
            instructions = str(task.instructions)
        dummy = DummyUser(
            instructions=instructions,
            persona=str(task.user_scenario.user_profile),
            llm=llm,
            llm_args=llm_args,
            language=language,
        )
        state = dummy.get_init_state()
        user_msg, _ = dummy.generate_next_message(greeting, state)
        logger.info(f"  [{task.id}] -> {user_msg.content[:120]!r}...")
        with lock:
            messages[task.id] = user_msg.content

    logger.info(f"Generating with concurrency={max_concurrency} ...")
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        list(executor.map(_generate, needs_update))

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fp:
        json.dump(messages, fp, indent=2, ensure_ascii=False)

    logger.info(f"Done. {len(needs_update)} messages written to {output}")


def main():
    parser = argparse.ArgumentParser(description="Pregenerate solo user messages for tasks.")
    parser.add_argument("--domain", required=True, help="Domain name (e.g. ota, delivery, instore)")
    parser.add_argument("--language", default="english", choices=["english", "chinese"])
    parser.add_argument("--llm", required=True, help="LLM model name to use for generation")
    parser.add_argument("--task-ids", nargs="+", default=None, help="Only generate for these task IDs.")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Number of tasks to generate in parallel. Default is 1.")
    parser.add_argument("--force", action="store_true", help="Regenerate even if already present in the output file")
    parser.add_argument("--resolved-instructions", default=None, help="Path to resolved instructions JSON (from preresolve_dates.py)")
    args = parser.parse_args()

    output = build_output_path(args.llm, args.domain, args.language, resolved=bool(args.resolved_instructions))
    logger.info(f"Output path: {output}")

    pregenerate(
        domain=args.domain,
        language=args.language,
        llm=args.llm,
        llm_args=models.get(args.llm, {}),
        output=output,
        task_ids=args.task_ids,
        max_concurrency=args.max_concurrency,
        force=args.force,
        resolved_instructions_file=args.resolved_instructions,
    )


if __name__ == "__main__":
    main()
