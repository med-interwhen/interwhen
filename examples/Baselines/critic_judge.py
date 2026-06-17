"""
Critic judgement baseline (post-hoc, downstream of generation).

Given a *precomputed* run directory (the per-solver ``outputs_solver_{i}.jsonl``
files produced by the TTSwithVerification example scripts), this script asks an
LLM "critic" to judge each generated sample as CORRECT / INCORRECT.

It only computes per-sample critic judgements; it does NOT generate samples and
does NOT do best-of-K selection (that is a separate, cheap downstream step over
these judgements). Generation and selection are fully decoupled, so the critic
can be run as many times as desired over the same fixed set of samples.

The sample being judged is always the stored ``output_text`` field; runs that
predate that field must be regenerated.

The task/domain is passed as ``--task``. Supported domains:
    game24, maze, spatialmap, zebralogic, verina_code, verina_spec

Critic prompts are ported from the upstream ``bestofk_baseline.py``.

Usage:
    python critic_judge.py \
        --task verina_spec \
        --run_dir ../../Outputs_TTS/VerinaSpecResults/<run> \
        --critic_model google/gemma-4-E4B-it \
        --port 8000 \
        --processes 32
"""

import argparse
import glob
import json
import logging
import os
import re
from collections import defaultdict
from multiprocessing import Pool

import requests
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

TASKS = ["game24", "maze", "spatialmap", "zebralogic", "verina_code", "verina_spec"]

CRITIC_SYSTEM = "You are a strict academic verifier."

# Module-level state shared with worker processes (inherited via fork).
ARGS = None
DATASET = None          # task dataset (maze/spatial: HF dataset; verina: list)
VERINA_BY_ID = None     # data_id -> BenchmarkData (verina only)
TOKENIZER = None        # critic-model tokenizer (for chat-template application)


# ============================================================================
# Critic prompt builders (ported from upstream bestofk_baseline.py)
# ============================================================================

def build_game24_critic_prompt(nums, reasoning_output):
    return f"""You are a math verifier. Evaluate the following Game of 24 solution.

Numbers: {nums}
Target: 24

Student's reasoning and answer:
{reasoning_output}

Verify:
1. Does it use ALL four numbers exactly once?
2. Does each step follow correct arithmetic?
3. Does the final expression evaluate to exactly 24?

Respond in the following format:
VERDICT: CORRECT or INCORRECT
REASONING: Your detailed explanation

If CORRECT, briefly explain why.
If INCORRECT, explain what went wrong and how to fix it.
"""


def build_zebralogic_critic_prompt(task_description, reasoning_output):
    return f"""You are an expert logic puzzle verifier. Evaluate the following ZebraLogic solution.

Task:
{task_description}

Student's reasoning and answer:
{reasoning_output}

Verify:
1. Does the solution assign exactly one value per feature per house?
2. Are all constraints/clues satisfied?
3. Is the JSON output well-formed and complete?

Respond in the following format:
VERDICT: CORRECT or INCORRECT
REASONING: Your detailed explanation

If CORRECT, briefly explain why.
If INCORRECT, explain what went wrong and suggest corrections.
"""


def build_mcq_critic_prompt(task, task_description, reasoning_output):
    task_name = "Maze" if task == "maze" else "Spatial Reasoning"
    return f"""You are an expert {task_name} verifier. Evaluate the following solution.

Task:
{task_description}

Student's reasoning and answer:
{reasoning_output}

Verify the correctness of the step-by-step reasoning and final answer.

Respond in the following format:
VERDICT: CORRECT or INCORRECT
REASONING: Your detailed explanation

If CORRECT, briefly explain why.
If INCORRECT, explain what went wrong and suggest the correct approach.
"""


def build_verina_code_critic_prompt(data, reasoning_output):
    from interwhen.utils.verina_spec_example_utils import render_param_list

    signature = data.signature
    func_name = signature.get("name", "solution")
    return_type = signature.get("return_type", "Bool")
    param_list = render_param_list(signature)

    precond = data.lean_data.get("precond", "True").strip()
    postcond = data.lean_data.get("postcond", "").strip()

    return f"""You are an expert Lean 4 code verifier. Evaluate the following code generation attempt.

## Task Description
{data.description}

## Function Signature
```lean4
def {func_name} {param_list} (h_precond : {func_name}_precond ...) : {return_type}
```

## Precondition
```lean4
{precond}
```

## Postcondition
```lean4
{postcond}
```

## Student's Reasoning and Generated Code
{reasoning_output}

Verify:
1. Is the generated code syntactically valid Lean 4?
2. Does it match the expected function signature and return type ({return_type})?
3. Does the logic appear to satisfy the postcondition given the precondition?
4. Are there any obvious bugs, infinite loops, or incorrect base cases?

Respond in the following format:
VERDICT: CORRECT or INCORRECT
REASONING: Your detailed explanation

If CORRECT, briefly explain why
If INCORRECT, explain what went wrong and suggest how to fix it.
"""


def build_verina_spec_critic_prompt(data, reasoning_output):
    """Critic for Lean 4 specification generation (precond + postcond).

    Upstream bestofk_baseline.py only ships a verina *code* critic; this is the
    analogous critic for the spec-generation task, judging soundness and
    completeness of the generated precondition/postcondition.
    """
    from interwhen.utils.verina_spec_example_utils import render_param_list

    signature = data.signature
    func_name = signature.get("name", "solution")
    return_type = signature.get("return_type", "Bool")
    param_list = render_param_list(signature)

    return f"""You are an expert Lean 4 specification verifier. Evaluate the following specification generation attempt (precondition and postcondition).

## Task Description
{data.description}

## Function Signature
```lean4
def {func_name} {param_list} : {return_type}
```

## Student's Reasoning and Generated Specification
{reasoning_output}

Verify:
1. Is the generated precondition valid Lean 4 and as permissive as possible while ensuring correct execution?
2. Is the postcondition sound: does it reject all incorrect outputs?
3. Is the postcondition complete: does it accept every correct output?
4. Does the specification fully and correctly capture the input/output relationship described in the task?

Respond in the following format:
VERDICT: CORRECT or INCORRECT
REASONING: Your detailed explanation

If CORRECT, briefly explain why.
If INCORRECT, explain what went wrong and suggest how to fix it.
"""


# ============================================================================
# Per-domain problem-statement sources
# ============================================================================

def remove_last_paragraph(s: str) -> str:
    """Strip the trailing answer-format instruction (matches example scripts)."""
    return s[:-143] if len(s) > 143 else s


def build_zebralogic_task_description(problem: dict) -> str:
    from interwhen.utils.zebralogic_helper import USER_PROMPT_TEMPLATE

    problem_text = problem.get("puzzle_clean") or problem.get("puzzle", "")
    return USER_PROMPT_TEMPLATE.format(problem_text=problem_text)


def get_verina_data(record):
    """Resolve the BenchmarkData for a verina record (by data_id, then idx)."""
    data_id = record.get("data_id")
    if data_id and VERINA_BY_ID and data_id in VERINA_BY_ID:
        return VERINA_BY_ID[data_id]
    return DATASET[int(record["idx"])]


def get_critic_prompt(task, record, sample_text):
    if task == "game24":
        return build_game24_critic_prompt(record["numbers"], sample_text)
    if task == "zebralogic":
        desc = build_zebralogic_task_description(record["problem"])
        return build_zebralogic_critic_prompt(desc, sample_text)
    if task in ("maze", "spatialmap"):
        example = DATASET[int(record["idx"])]
        desc = remove_last_paragraph(str(example.get("prompt")))
        return build_mcq_critic_prompt(task, desc, sample_text)
    if task == "verina_code":
        return build_verina_code_critic_prompt(get_verina_data(record), sample_text)
    if task == "verina_spec":
        return build_verina_spec_critic_prompt(get_verina_data(record), sample_text)
    raise ValueError(f"Unsupported task: {task}")


def get_sample_text(task, solver_idx, record):
    """Return the generated trace to judge (the stored ``output_text``)."""
    return record.get("output_text") or ""


# ============================================================================
# Critic LLM call (simple synchronous vLLM completions request)
# ============================================================================

def call_critic(prompt: str) -> str:
    # Apply the chat template locally so we can control generation flags such as
    # enable_thinking, then hit the raw completions endpoint.
    full_prompt = TOKENIZER.apply_chat_template(
        [
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    url = f"http://localhost:{ARGS.port}/v1/completions"
    payload = {
        "model": ARGS.critic_model,
        "prompt": full_prompt,
        "max_tokens": ARGS.max_tokens,
    }
    try:
        resp = requests.post(url, json=payload, timeout=300)
        if resp.status_code >= 400:
            logger.warning("HTTP %s from critic: %s", resp.status_code, resp.text[:200])
            return ""
        result = resp.json()
        return result["choices"][0].get("text", "") or ""
    except Exception as e:  # noqa: BLE001
        logger.warning("Critic request failed: %s", e)
        return ""


def parse_verdict(critic_output: str):
    """Return (is_correct, reasoning).

    Robustly reads the ``VERDICT:`` line. (The upstream `"CORRECT" in text`
    check is buggy because "INCORRECT" contains "CORRECT"; this fixes that.)
    """
    up = critic_output.upper()
    m = re.search(r"VERDICT\s*:\s*(INCORRECT|CORRECT)", up)
    if m:
        is_correct = m.group(1) == "CORRECT"
    else:
        is_correct = ("INCORRECT" not in up) and ("CORRECT" in up)

    reasoning = ""
    if "REASONING:" in critic_output:
        reasoning = critic_output.split("REASONING:", 1)[1].strip()
    elif "VERDICT:" not in critic_output:
        reasoning = critic_output
    return is_correct, reasoning


# ============================================================================
# Worker
# ============================================================================

def judge_one(item):
    """item = (solver_idx, row, record).

    Returns a copy of the original ``record`` with the critic judgement fields
    appended (the source solver files are never modified). ``row`` is the
    0-based line position of this record within ``outputs_solver_{solver_idx}``.
    """
    solver_idx, row, record = item

    out = dict(record)  # copy; never mutate the original
    out["solver_idx"] = solver_idx
    out["row"] = row

    sample_text = get_sample_text(ARGS.task, solver_idx, record)
    if not sample_text.strip():
        out["critic_correct"] = False
        out["critic_feedback"] = ""
        out["critic_raw"] = ""
        out["critic_skipped"] = True
        return out

    prompt = get_critic_prompt(ARGS.task, record, sample_text)
    critic_output = call_critic(prompt)
    is_correct, reasoning = parse_verdict(critic_output)

    out["critic_correct"] = bool(is_correct)
    out["critic_feedback"] = reasoning
    out["critic_raw"] = critic_output
    out["critic_skipped"] = False
    return out


# ============================================================================
# Dataset loading
# ============================================================================

def load_task_dataset(task):
    if task in ("game24", "zebralogic"):
        return None  # problem statement is inline in the jsonl records
    if task == "maze":
        from datasets import load_dataset
        return load_dataset("microsoft/VISION_LANGUAGE", "maze_text_only", split="val")
    if task == "spatialmap":
        from datasets import load_dataset
        return load_dataset("microsoft/VISION_LANGUAGE", "spatial_map_text_only", split="val")
    if task in ("verina_code", "verina_spec"):
        from interwhen.utils.verina_spec_example_utils import load_verina_dataset
        return load_verina_dataset()
    raise ValueError(f"Unsupported task: {task}")


def model_short_name(model_name: str) -> str:
    return model_name.split("/")[-1].replace(" ", "_").replace(":", "-")


# ============================================================================
# Main
# ============================================================================

def main():
    global ARGS, DATASET, VERINA_BY_ID, TOKENIZER

    parser = argparse.ArgumentParser(description="Critic judgement baseline for a precomputed run")
    parser.add_argument("--task", required=True, choices=TASKS, help="Domain to judge")
    parser.add_argument("--run_dir", required=True, help="Precomputed run directory (contains outputs_solver_*.jsonl)")
    parser.add_argument("--critic_model", required=True, help="Critic model name served by vLLM")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--processes", "-p", type=int, default=16, help="Parallel worker processes")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Critic generation max tokens")
    parser.add_argument("--debug", "-d", action="store_true", help="Single-process + debug logging")
    ARGS = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if ARGS.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not os.path.isdir(ARGS.run_dir):
        raise SystemExit(f"run_dir not found: {ARGS.run_dir}")

    solver_files = sorted(glob.glob(os.path.join(ARGS.run_dir, "outputs_solver_*.jsonl")))
    if not solver_files:
        raise SystemExit(f"No outputs_solver_*.jsonl found in {ARGS.run_dir}")

    logger.info("Task: %s | Critic: %s | %d solver file(s)", ARGS.task, ARGS.critic_model, len(solver_files))

    TOKENIZER = AutoTokenizer.from_pretrained(ARGS.critic_model, trust_remote_code=True)

    DATASET = load_task_dataset(ARGS.task)
    if ARGS.task in ("verina_code", "verina_spec"):
        VERINA_BY_ID = {d.data_id: d for d in DATASET}

    # Build the flat list of (solver_idx, row, record) work items.
    work = []
    for path in solver_files:
        m = re.search(r"outputs_solver_(\d+)\.jsonl$", path)
        solver_idx = int(m.group(1)) if m else 0
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        for row, r in enumerate(records):
            work.append((solver_idx, row, r))

    logger.info("Judging %d samples total", len(work))

    if ARGS.debug or ARGS.processes <= 1:
        results = [judge_one(item) for item in tqdm(work, desc="Critic")]
    else:
        with Pool(processes=ARGS.processes) as pool:
            results = list(tqdm(pool.imap_unordered(judge_one, work), total=len(work), desc="Critic"))

    # Group judgements by solver and write out.
    out_dir = os.path.join(ARGS.run_dir, "critic_judgements", model_short_name(ARGS.critic_model))
    os.makedirs(out_dir, exist_ok=True)

    by_solver = defaultdict(list)
    for r in results:
        by_solver[r["solver_idx"]].append(r)

    for solver_idx in sorted(by_solver):
        rows = sorted(by_solver[solver_idx], key=lambda r: r["row"])
        out_path = os.path.join(out_dir, f"critic_solver_{solver_idx}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n = len(rows)
        n_correct = sum(1 for r in rows if r["critic_correct"])
        n_skipped = sum(1 for r in rows if r.get("critic_skipped"))
        logger.info(
            "solver_%d: critic_correct %d/%d (%.2f%%), skipped=%d -> %s",
            solver_idx, n_correct, n, 100 * n_correct / n if n else 0.0, n_skipped, out_path,
        )


if __name__ == "__main__":
    main()
