"""
Pre-resolve relative dates in OTA task instructions.

Replaces relative date expressions ("next Saturday", "next month on the 1st",
"the day after Qixi Festival", etc.) with absolute dates using an LLM, given
the simulated environment time.

Produces a JSON mapping {task_id: resolved_instructions} that can be loaded
at runtime to replace the original instructions.

Usage:
    python src/vita/scripts/preresolve_dates.py --model <model>
    python src/vita/scripts/preresolve_dates.py --model <model> --tasks-file data/vita/domains/ota/tasks_en.json
    python src/vita/scripts/preresolve_dates.py --model <model> --language english
    python src/vita/scripts/preresolve_dates.py --model <model> --task-ids D0812002 70812007
    python src/vita/scripts/preresolve_dates.py --model <model> --resume
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from vita.config import DEFAULT_LLM_EVALUATOR, models
from vita.data_model.message import SystemMessage, UserMessage
from vita.domains.ota.verifier.utils import _extract_json
from vita.prompts import get_prompts
from vita.utils.llm_utils import generate
from vita.utils.utils import get_weekday


WEEKDAY_NAMES = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday",
    3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday",
}


def _verify_day(date_str: str, claimed_day: str) -> bool:
    """Return True if claimed day-of-week matches for a single date, a range, or a month."""
    # Month-only: YYYY-MM
    if re.fullmatch(r"\d{4}-\d{2}", date_str):
        return True  # nothing to verify for month-level
    # Range: YYYY-MM-DD to YYYY-MM-DD
    range_match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})", date_str)
    if range_match:
        days = [d.strip() for d in claimed_day.split("-")]
        if len(days) != 2:
            return False
        return _verify_single(range_match.group(1), days[0]) and _verify_single(range_match.group(2), days[1])
    # Multi-date with "and": YYYY-MM-DD and YYYY-MM-DD [and ...]
    and_match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})(?:\s+and\s+\d{4}-\d{2}-\d{2})+", date_str)
    if and_match:
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", date_str)
        days = [d.strip() for d in claimed_day.split(" and ")]
        if len(dates) != len(days):
            return False
        return all(_verify_single(d, w) for d, w in zip(dates, days))
    # Single date
    return _verify_single(date_str, claimed_day)


def _verify_single(date_str: str, claimed_day: str) -> bool:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return False
    return WEEKDAY_NAMES[dt.weekday()].lower() == claimed_day.lower().strip()


def load_tasks(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _build_correction_message(dates: list[dict]) -> str | None:
    """Build a correction prompt for any dates that failed verification.
    Returns None if all dates are verified."""
    errors = []
    for d in dates:
        if not d["verified"]:
            date_str = d["date"]
            claimed = d["day"]
            # Compute actual day(s)
            range_match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})", date_str)
            if range_match:
                days_claimed = [x.strip() for x in claimed.split("-")]
                parts = []
                for ds, dc in zip([range_match.group(1), range_match.group(2)], days_claimed):
                    try:
                        actual = WEEKDAY_NAMES[datetime.strptime(ds, "%Y-%m-%d").weekday()]
                    except ValueError:
                        actual = "?"
                    if actual.lower() != dc.lower():
                        parts.append(f"{ds} is actually {actual}, not {dc}")
                if parts:
                    errors.append(f'- "{d["phrase"]}": {"; ".join(parts)}')
            elif re.fullmatch(r"\d{4}-\d{2}", date_str):
                continue  # month-level, nothing to verify
            else:
                try:
                    actual = WEEKDAY_NAMES[datetime.strptime(date_str, "%Y-%m-%d").weekday()]
                except ValueError:
                    actual = "?"
                errors.append(f'- "{d["phrase"]}": {date_str} is actually {actual}, not {claimed}')
    if not errors:
        return None
    return (
        "Some of your resolved dates have incorrect days of the week:\n"
        + "\n".join(errors)
        + "\n\nPlease fix the errors and output the corrected JSON in the same format."
    )


def resolve_dates(
    task_id: str,
    instructions: str,
    system_time: str,
    language: str,
    llm_model: str,
    llm_args: dict,
    max_retries: int = 2,
) -> dict:
    """Call the LLM to extract relative date phrases, resolve them, and
    return the replaced instructions.

    Returns dict with keys "resolved_dates" and "resolved_instructions".
    """
    weekday = get_weekday(system_time, language)

    prompts = get_prompts(language)
    system_prompt = prompts.date_resolution_template.format(
        system_time=system_time,
        weekday=weekday,
        instructions=instructions,
    )

    messages = [
        SystemMessage(role="system", content=system_prompt),
        UserMessage(role="user", content=instructions),
    ]

    for attempt in range(1 + max_retries):
        response = generate(model=llm_model, messages=messages, enable_think=True, **llm_args)

        raw_content = (response.content or "").strip()
        if not raw_content:
            raise ValueError(f"Empty response for task {task_id}")

        parsed = _extract_json(raw_content)
        if parsed is None:
            raise ValueError(f"No JSON found in response for {task_id}: {raw_content[:200]}")

        if not isinstance(parsed, dict):
            raise ValueError(f"Expected dict, got {type(parsed).__name__} for {task_id}")

        tuples = parsed.get("resolved_dates", [])
        resolved_instructions = parsed.get("resolved_instructions", "")

        if not resolved_instructions:
            raise ValueError(f"Missing resolved_instructions for {task_id}")

        dates = []
        for item in tuples:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                phrase, date_str, day = item
                dates.append({
                    "phrase": phrase,
                    "date": date_str,
                    "day": day,
                    "verified": _verify_day(date_str, day),
                })

        # Check if correction is needed
        correction = _build_correction_message(dates)
        if correction is None or attempt == max_retries:
            break

        # Feed back errors for correction
        messages.append({"role": "assistant", "content": raw_content})
        messages.append(UserMessage(role="user", content=correction))

    return {"system_time": system_time, "resolved_dates": dates, "resolved_instructions": resolved_instructions}


def main():
    parser = argparse.ArgumentParser(
        description="Pre-resolve relative dates in OTA task instructions"
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
        help="Output path (default: derived from model name)",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Only resolve specific task IDs",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Only resolve first N tasks",
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
        help="Resume from existing output file, skipping already-resolved tasks",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=16,
        help="Max parallel resolution tasks (default: 16)",
    )
    args = parser.parse_args()

    # Derive output path from model name if not specified
    if args.output is None:
        model_slug = args.model.rsplit("/", 1)[-1]
        args.output = f"data/vita/domains/ota/resolved_instructions_{model_slug}.json"

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
    existing: dict[str, str] = {}
    if args.resume:
        try:
            with open(args.output) as f:
                existing = json.load(f)
            print(f"Resuming: {len(existing)} tasks already resolved")
        except FileNotFoundError:
            pass

    # LLM config
    llm_args = dict(models.get(args.model, models.get("default", {})))
    llm_args["max_tokens"] = max(llm_args.get("max_tokens", 0), 4096)

    results: dict[str, str] = dict(existing)
    successes = 0
    failures = 0
    lock = threading.Lock()

    # Filter out already-resolved tasks
    pending: list[tuple[int, dict]] = []
    for i, raw in enumerate(raw_tasks):
        task_id = raw.get("id", f"unknown_{i}")
        if task_id in existing:
            print(f"[{i+1}/{len(raw_tasks)}] {task_id} — skipped (already resolved)")
            successes += 1
        else:
            pending.append((i, raw))

    total = len(raw_tasks)

    def _resolve_one(idx: int, raw: dict) -> None:
        nonlocal successes, failures
        task_id = raw.get("id", f"unknown_{idx}")
        env = raw.get("environment", {})
        system_time = env.get("time", "")
        instructions = raw.get("instructions", "")

        if not system_time:
            print(f"[{idx+1}/{total}] {task_id} — SKIPPED (no env time)")
            return
        if not instructions:
            print(f"[{idx+1}/{total}] {task_id} — SKIPPED (no instructions)")
            return

        print(f"[{idx+1}/{total}] {task_id} — resolving...", flush=True)
        t0 = time.time()

        try:
            result = resolve_dates(
                task_id=task_id,
                instructions=instructions,
                system_time=system_time,
                language=args.language,
                llm_model=args.model,
                llm_args=llm_args,
            )
            elapsed = time.time() - t0
            n_dates = len(result["resolved_dates"])
            print(f"[{idx+1}/{total}] {task_id} — OK ({n_dates} dates, {elapsed:.1f}s)")
            with lock:
                results[task_id] = result
                successes += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{idx+1}/{total}] {task_id} — FAILED ({elapsed:.1f}s): {e}")
            with lock:
                results[task_id] = {"_error": str(e)}
                failures += 1

        # Save after each task
        with lock:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    max_workers = max(1, args.max_concurrency)
    if max_workers == 1:
        for idx, raw in pending:
            _resolve_one(idx, raw)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_resolve_one, idx, raw): idx for idx, raw in pending}
            for fut in as_completed(futures):
                fut.result()

    print(f"\nDone: {successes} succeeded, {failures} failed")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
