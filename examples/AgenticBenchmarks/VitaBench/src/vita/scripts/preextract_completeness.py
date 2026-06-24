"""
Pre-extract completeness constraints for all OTA tasks and save to a JSON file.

Run this ONCE before test time so the verifier can load pre-computed
completeness constraints instead of calling the LLM at runtime.

Usage:
    python src/vita/scripts/preextract_completeness.py --model <model>
    python src/vita/scripts/preextract_completeness.py --model <model> --max-concurrency 16
    python src/vita/scripts/preextract_completeness.py --model <model> --task-ids D0811005 D0811006
    python src/vita/scripts/preextract_completeness.py --model <model> --num-tasks 10
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from vita.data_model.tasks import Task
from vita.domains.ota.completeness import extract_completeness_constraints


def load_tasks(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract OTA completeness constraints for all tasks"
    )
    parser.add_argument(
        "--tasks-file",
        default="data/vita/domains/ota/tasks_en.json",
        help="Path to tasks JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path (default: data/vita/domains/ota/completeness_{model}.json)",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Only extract specific task IDs",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Only extract first N tasks",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LLM model name",
    )
    parser.add_argument(
        "--language",
        default="english",
        choices=["english", "chinese"],
        help="Prompt language (default: english)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skipping already-extracted tasks",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Max parallel extraction tasks (default: 1)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/vita/domains/ota/completeness_{args.model}.json"

    raw_tasks = load_tasks(args.tasks_file)
    print(f"Loaded {len(raw_tasks)} tasks from {args.tasks_file}")

    # Filter
    if args.task_ids:
        raw_tasks = [t for t in raw_tasks if t.get("id") in args.task_ids]
        if not raw_tasks:
            print(f"No tasks found matching IDs: {args.task_ids}")
            sys.exit(1)
    elif args.num_tasks is not None:
        raw_tasks = raw_tasks[: args.num_tasks]

    # Load existing results if resuming
    existing: dict[str, dict] = {}
    if args.resume:
        try:
            with open(args.output) as f:
                existing = json.load(f)
            print(f"Resuming: {len(existing)} tasks already extracted")
        except FileNotFoundError:
            pass

    results: dict[str, dict] = dict(existing)
    successes = 0
    failures = 0
    lock = threading.Lock()

    # Filter out already-extracted tasks
    pending: list[tuple[int, dict]] = []
    for i, raw in enumerate(raw_tasks):
        task_id = raw.get("id", f"unknown_{i}")
        if task_id in existing:
            print(f"[{i+1}/{len(raw_tasks)}] {task_id} — skipped (already extracted)")
            successes += 1
        else:
            pending.append((i, raw))

    total = len(raw_tasks)

    def _extract_one(idx: int, raw: dict) -> None:
        nonlocal successes, failures
        task_id = raw.get("id", f"unknown_{idx}")
        print(f"[{idx+1}/{total}] {task_id} — extracting...", flush=True)
        t0 = time.time()

        raw.setdefault("environment", {})
        raw.setdefault("user_scenario", {"user_profile": {}})
        task = Task(**raw)

        try:
            constraints = extract_completeness_constraints(
                task,
                llm_model=args.model,
                language=args.language,
            )
            result = constraints.model_dump(exclude_none=True)
            elapsed = time.time() - t0
            n_items = (
                len(constraints.hotel)
                + len(constraints.flight)
                + len(constraints.train)
                + len(constraints.attraction)
                + len(constraints.cancel)
                + len(constraints.modify)
                + len(constraints.conditional)
            )
            print(f"[{idx+1}/{total}] {task_id} — OK ({n_items} constraints, {elapsed:.1f}s)")
            with lock:
                results[task_id] = result
                successes += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{idx+1}/{total}] {task_id} — FAILED ({elapsed:.1f}s): {e}")
            with lock:
                results[task_id] = {"task_id": task_id, "_error": str(e)}
                failures += 1

        # Save after each task so progress isn't lost
        with lock:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    max_workers = max(1, args.max_concurrency)
    if max_workers == 1:
        for idx, raw in pending:
            _extract_one(idx, raw)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_extract_one, idx, raw): idx for idx, raw in pending}
            for fut in as_completed(futures):
                fut.result()

    print(f"\nDone: {successes} succeeded, {failures} failed")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
