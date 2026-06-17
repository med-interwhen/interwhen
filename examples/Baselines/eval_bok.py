"""
Best-of-K / generate-test evaluation over a precomputed run (prints; writes nothing).

Two orthogonal axes:

  --method  bok      Best-of-K: with K samples available, is the selected sample
                     correct? ("is any sample correct")
            gentest  Generate-and-test: draw samples one at a time and stop at the
                     first the judge accepts. Reports how many samples that took.

  --judge   gt       A sample is "correct" iff it is ground-truth-correct.
            critic   A sample is "correct" iff the critic judged it correct
                     (requires critic_judge.py to have been run; --critic_model).

In every case selection stops at the first judge-accepted sample (lowest solver
index first; fall back to sample 0 if none is accepted), and the FINAL score is
always computed against ground truth.

Ground truth is reused from the stored per-sample fields wherever possible:
    game24 / maze / spatialmap -> ``is_correct``
    verina_code                -> ``all_tests_pass``
    verina_spec                -> ``full_spec_correct``
    zebralogic                 -> recomputed from ``problem`` + ``output_text``

Usage:
    python eval_bok.py --task maze --run_dir <run> --method bok --judge gt
    python eval_bok.py --task maze --run_dir <run> --method gentest --judge critic --critic_model google/gemma-4-E4B-it
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

TASKS = ["game24", "maze", "spatialmap", "zebralogic", "verina_code", "verina_spec"]


def model_short_name(model_name: str) -> str:
    return model_name.split("/")[-1].replace(" ", "_").replace(":", "-")


def query_key(task, record):
    """Stable identifier for the underlying query (shared across solver files)."""
    if task == "zebralogic":
        return record.get("problem_id")
    if task in ("verina_code", "verina_spec"):
        return (record.get("idx"), record.get("data_id"))
    return record.get("idx")


# (n_houses, n_features) grids that make up the zebra "x-large" complexity bucket.
_ZEBRA_XLARGE = {(5, 5), (6, 4), (5, 6), (6, 5), (6, 6)}


def is_zebra_xlarge(record):
    """True iff a zebralogic record's puzzle falls in the x-large bucket."""
    problem = record.get("problem") or {}
    return (problem.get("n_houses"), problem.get("n_features")) in _ZEBRA_XLARGE


# ============================================================================
# Ground-truth correctness (reused from stored fields; zebra is recomputed)
# ============================================================================

_CLOSE_THINK = None  # think-close tag of the solver model (set in main)


def is_gt_correct(task, record):
    if task in ("game24", "maze", "spatialmap"):
        return bool(record.get("is_correct"))
    if task == "verina_code":
        return bool(record.get("all_tests_pass"))
    if task == "verina_spec":
        return bool(record.get("full_spec_correct"))
    if task == "zebralogic":
        return _zebra_gt_correct(record)
    raise ValueError(f"Unsupported task: {task}")


def _zebra_gt_correct(record):
    from interwhen.utils.zebralogic_helper import extract_last_json, zebra_correctness

    problem = record.get("problem")
    output_text = record.get("output_text", "")
    if not problem or not output_text:
        return False
    try:
        candidate = extract_last_json(output_text, close_think=_CLOSE_THINK)
        if not candidate:
            return False
        c, s, m, t = zebra_correctness(problem, candidate)
        return t > 0 and c == t
    except Exception:
        return False


# ============================================================================
# Loading
# ============================================================================

def load_solver_groups(paths, task):
    """Return {query_key: [(solver_idx, record), ...]} preserving solver order."""
    groups = defaultdict(list)
    for path in paths:
        m = re.search(r"_solver_(\d+)\.jsonl$", path)
        solver_idx = int(m.group(1)) if m else 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    groups[query_key(task, record)].append((solver_idx, record))
    for k in groups:
        groups[k].sort(key=lambda x: x[0])
    return groups


# ============================================================================
# Selection & evaluation
# ============================================================================

def judge_accepts(judge, task, record):
    if judge == "gt":
        return is_gt_correct(task, record)
    return bool(record.get("critic_correct"))


def sample_tokens(record):
    """Generated token count for a sample, or None if not recorded."""
    t = record.get("generated_tokens")
    return t if isinstance(t, (int, float)) else None


def add_zebra_token_counts(groups, model_name):
    """Populate ``generated_tokens`` for zebra records by re-tokenizing output_text.

    Zebra runs store only ``output_text``; recompute token counts with the solver
    tokenizer (same approach as the notebooks) so token cost can be reported.
    No-op if the tokenizer cannot be loaded.
    """
    if not model_name:
        return
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return
    for samples in groups.values():
        for _, rec in samples:
            if "generated_tokens" not in rec:
                rec["generated_tokens"] = len(tok.encode(rec.get("output_text", ""), add_special_tokens=False))


def select_sample(samples, judge, task):
    """Scan samples in solver order; stop at the first the judge accepts.

    Returns (selected_record, n_samples_drawn, accepted).
    Falls back to the last sample (all K drawn) when none is accepted.
    """
    for i, (_, rec) in enumerate(samples):
        if judge_accepts(judge, task, rec):
            return rec, i + 1, True
    return samples[-1][1], len(samples), False


def token_cost(groups, drawn):
    """Cost of drawing ``drawn[key]`` samples per query, relative to drawing one.

    Returns the ratio (drawn tokens / single-sample baseline), or None if any
    relevant sample lacks ``generated_tokens``.
    """
    total = baseline = 0
    for key, samples in groups.items():
        first = sample_tokens(samples[0][1])
        if first is None:
            return None
        baseline += first
        for _, rec in samples[: drawn[key]]:
            t = sample_tokens(rec)
            if t is None:
                return None
            total += t
    return total / baseline if baseline else None


def eval_bok(groups, judge, task):
    """Is the judge-selected sample ground-truth-correct?

    Returns (solved, total, judge_accepted, judge_correct, drawn).
    """
    solved = judge_accepted = judge_correct = 0
    drawn = {}
    for key, samples in groups.items():
        rec, _, ok = select_sample(samples, judge, task)
        drawn[key] = len(samples)  # bok draws all K
        gt = is_gt_correct(task, rec)
        solved += gt
        if ok:
            judge_accepted += 1
            judge_correct += gt
    return solved, len(groups), judge_accepted, judge_correct, drawn


def eval_gentest(groups, judge, task):
    """Generate-and-test: stop at first judge-accepted sample, track samples drawn.

    Returns (solved, total, draws_dist, draws_total, judge_accepted, judge_correct, drawn).
      draws_dist     {n_samples_drawn: n_queries}
      judge_accepted #queries where the judge accepted a sample
      judge_correct  #of those where the accepted sample is gt-correct (for precision)
      drawn          {query_key: n_samples_drawn}
    """
    solved = draws_total = judge_accepted = judge_correct = 0
    draws_dist = defaultdict(int)
    drawn = {}
    for key, samples in groups.items():
        rec, n, ok = select_sample(samples, judge, task)
        drawn[key] = n
        draws_total += n
        draws_dist[n] += 1
        gt = is_gt_correct(task, rec)
        solved += gt
        if ok:
            judge_accepted += 1
            judge_correct += gt
    return solved, len(groups), draws_dist, draws_total, judge_accepted, judge_correct, drawn


# ============================================================================
# Reporting
# ============================================================================

def report(groups, task, method, judge, critic_model, run_dir, label=None):
    """Run the chosen method over ``groups`` and print the results."""
    k = max((len(v) for v in groups.values()), default=0)
    head = f"task: {task}" + (f" ({label})" if label else "")

    if method == "bok":
        solved, total, judge_accepted, judge_correct, drawn = eval_bok(groups, judge, task)
        print(f"{head} | method: bok | judge: {judge} | run: {run_dir}")
        if judge == "critic":
            print(f"critic: {critic_model}")
        print(f"queries: {total} | K: {k}")
        print(f"accuracy: {solved}/{total} = {100 * solved / total:.2f}%" if total else "no queries")
        if judge == "critic" and total:
            print(f"judge precision: {judge_correct}/{judge_accepted} = {100 * judge_correct / judge_accepted:.2f}%" if judge_accepted else "judge precision: n/a")
        ratio = token_cost(groups, drawn)
        print(f"token cost vs 1 sample: {ratio:.3f}x" if ratio is not None else "token cost vs 1 sample: n/a (generated_tokens not recorded)")
        return

    # gentest
    solved, total, draws_dist, draws_total, judge_accepted, judge_correct, drawn = eval_gentest(groups, judge, task)
    print(f"{head} | method: gentest | judge: {judge} | run: {run_dir}")
    # if judge == "critic":
    #     print(f"critic: {critic_model}")
    # print(f"queries: {total} | K: {k}")
    print(f"accuracy: {solved}/{total} = {100 * solved / total:.2f}%" if total else "no queries")
    if total:
        # print(f"avg samples drawn: {draws_total / total:.3f}")
        # print("samples-drawn distribution:")
        # for n in range(1, k + 1):
        #     c = draws_dist.get(n, 0)
        #     print(f"  {n}: {c} ({100 * c / total:.2f}%)")
        # if judge == "critic":
        #     print(f"judge precision: {judge_correct}/{judge_accepted} = {100 * judge_correct / judge_accepted:.2f}%" if judge_accepted else "judge precision: n/a")
        ratio = token_cost(groups, drawn)
        print(f"token cost vs 1 sample: {ratio:.3f}x" if ratio is not None else "token cost vs 1 sample: n/a (generated_tokens not recorded)")


# ============================================================================
# Main
# ============================================================================

def main():
    global _CLOSE_THINK

    parser = argparse.ArgumentParser(description="Best-of-K / generate-test evaluation over a precomputed run")
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--run_dir", required=True, help="Precomputed run directory")
    parser.add_argument("--method", required=True, choices=["bok", "gentest"])
    parser.add_argument("--judge", required=True, choices=["gt", "critic"])
    parser.add_argument("--critic_model", help="Critic model name (required for --judge critic)")
    parser.add_argument("--max_k", type=int, default=None, help="Cap samples per query (first K in solver order)")
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"run_dir not found: {args.run_dir}")

    solver_model = None
    if args.task == "zebralogic":
        from interwhen.utils.llm import get_think_tags
        with open(os.path.join(args.run_dir, "args.json"), "r", encoding="utf-8") as f:
            solver_model = json.load(f)["solver_lm"]
        _CLOSE_THINK = get_think_tags(solver_model)["close"]

    # Load samples: critic judge needs the critic_judgements copies, gt the raw outputs.
    if args.judge == "critic":
        if not args.critic_model:
            raise SystemExit("--critic_model is required for --judge critic")
        critic_dir = os.path.join(args.run_dir, "critic_judgements", model_short_name(args.critic_model))
        paths = sorted(glob.glob(os.path.join(critic_dir, "critic_solver_*.jsonl")))
        if not paths:
            raise SystemExit(f"No critic_solver_*.jsonl found in {critic_dir} (run critic_judge.py first)")
    else:
        paths = sorted(glob.glob(os.path.join(args.run_dir, "outputs_solver_*.jsonl")))
        if not paths:
            raise SystemExit(f"No outputs_solver_*.jsonl found in {args.run_dir}")

    groups = load_solver_groups(paths, args.task)
    if args.task == "zebralogic":
        add_zebra_token_counts(groups, solver_model)
    if args.max_k is not None:
        if args.max_k < 1:
            raise SystemExit("--max_k must be >= 1")
        groups = {key: samples[: args.max_k] for key, samples in groups.items()}

    report(groups, args.task, args.method, args.judge, args.critic_model, args.run_dir)

    # For zebralogic also report the x-large complexity bucket on its own.
    if args.task == "zebralogic":
        xl = {key: samples for key, samples in groups.items()
              if is_zebra_xlarge(samples[0][1])}
        if xl:
            print()
            report(xl, args.task, args.method, args.judge, args.critic_model, args.run_dir, label="x-large")


if __name__ == "__main__":
    main()
