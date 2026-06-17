"""
Game of 24 experiment with thinking-phase step verification.

Uses ThinkingPhaseStepVerifierGame24Monitor which:
  - Verifies the model's intermediate expressions during the think-open tag via side-streams
  - Injects expression extraction after the think-close tag
  - Verifies the final \\boxed{} expression for correctness
"""

import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from multiprocessing import Pool

import numpy as np

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from interwhen import stream_completion
from interwhen.monitors import ThinkingPhaseStepVerifierGame24Monitor
from interwhen.utils.llm import init_llm_server, get_think_tags

# ============== MODEL CONFIGURATION ==============
MAIN_MODEL = "Qwen/QwQ-32B"
# =================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Walk up to find the repo root (contains pyproject.toml), output into it
_dir = _SCRIPT_DIR
while _dir != os.path.dirname(_dir) and not os.path.isfile(os.path.join(_dir, "pyproject.toml")):
    _dir = os.path.dirname(_dir)
_OUTPUT_ROOT = _dir

# Module-level objects shared with worker processes (inherited via fork)
tokenizer = None


def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from the model path."""
    short_name = model_name.split("/")[-1]
    short_name = short_name.replace(" ", "_").replace(":", "-")
    return short_name


def save_prompt(idx, prompt_with_answer, reason_dir):
    filename = os.path.join(reason_dir, f"reason_{idx}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_with_answer)

logger = logging.getLogger(__name__)


def build_prompt(nums):
    a, b, c, d = nums
    boxed = r"\boxed{}"
    base_prompt = f"""
    You are solving the Game of 24.
    
    You are given four numbers: {a}, {b}, {c}, {d}
    
    Your job is to produce a valid arithmetic expression using:
    - ALL four numbers exactly once
    - ONLY +, -, *, /
    - The expression must evaluate to exactly 24.
    
    Please reason step by step, and put your final answer containing only the expression within {boxed}.""".strip()
    return base_prompt


def count_tokens(text: str, tokenizer) -> int:
    """Count the total number of tokens in the generated text using the tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def extract_solution(text, open_think="<think>", close_think="</think>"):
    
    # Only search for \boxed{} AFTER the think-close tag to avoid grabbing unverified
    # expressions from inside the thinking trace.
    # If model opened the think tag but never closed it (hit token limit), there is
    # no final answer — return None.
    if close_think in text:
        search_text = text[text.rfind(close_think):]
    elif open_think in text:
        # Model started thinking but never finished — no verified answer
        return None
    else:
        search_text = text

    # Use a more robust extraction that handles nested braces in \boxed{}
    # Find \boxed{ and then match braces properly
    boxed_pattern = r"\\boxed\{"
    matches = list(re.finditer(boxed_pattern, search_text))
    if not matches:
        return None
    
    # Get the last \boxed{} content by matching braces
    last_match = matches[-1]
    start = last_match.end()  # Position right after \boxed{
    brace_count = 1
    end = start
    while end < len(search_text) and brace_count > 0:
        if search_text[end] == '{':
            brace_count += 1
        elif search_text[end] == '}':
            brace_count -= 1
        end += 1
    
    expr = search_text[start:end-1].strip()  # -1 to exclude the closing brace

    # Skip empty \boxed{} (e.g., from verifier feedback "Wrap in \boxed{}.")
    if not expr:
        return None

    # 1. Convert \frac{a}{b} to (a/b)
    frac_pattern = r"\\frac\{([^{}]+)\}\{([^{}]+)\}"
    while re.search(frac_pattern, expr):
        expr = re.sub(frac_pattern, r"(\1/\2)", expr)

    # 2. Replace LaTeX operators
    replacements = {
        r"\times": "*",
        r"\cdot": "*",
        r"\div": "/",
    }
    for latex, op in replacements.items():
        expr = expr.replace(latex, op)

    # 2b. Replace Unicode math operators (QwQ frequently uses these)
    expr = expr.replace('\u00d7', '*').replace('\u00f7', '/').replace('\u2212', '-')
    expr = expr.replace('\u2013', '-').replace('\u2014', '-')  # en-dash, em-dash

    # 3. Cleanup (remove LaTeX formatting artifacts)
    expr = expr.replace(r"\,", "").replace(r"\ ", "")
    expr = expr.replace(r"\left", "").replace(r"\right", "")

    # 3b. Strip trailing "= <number>" (e.g., "10 - 8/8 * 1 = 24" -> "10 - 8/8 * 1")
    expr = re.sub(r'\s*=\s*[\d.]+\s*$', '', expr)

    # 4. Handle implicit multiplication (e.g., "(11+1)(1+1)" -> "(11+1)*(1+1)")
    # Insert * between: )( , )number, number(, )(
    expr = re.sub(r'\)\s*\(', ')*(', expr)  # )( -> )*(
    expr = re.sub(r'\)\s*(\d)', r')*\1', expr)  # )number -> )*number
    expr = re.sub(r'(\d)\s*\(', r'\1*(', expr)  # number( -> number*(

    return expr

def extract_numbers_from_expr(expr):
    """Extract all numbers (including decimals) from an expression."""
    # Match integers and decimals
    numbers = re.findall(r'\d+\.?\d*', expr)
    return [int(float(n)) if float(n).is_integer() else float(n) for n in numbers]

def validate_numbers_used(expr, expected_nums):
    """Check if the expression uses exactly the given numbers (each exactly once)."""
    used_nums = extract_numbers_from_expr(expr)
    # Sort both lists to compare
    return sorted(used_nums) == sorted(expected_nums)

def evaluate_expression(expr, expected_nums=None):
    try:
        # First check if expression uses exactly the given numbers
        if expected_nums is not None:
            if not validate_numbers_used(expr, expected_nums):
                return False
        
        value = eval(expr, {"__builtins__": None}, {})
        return abs(value - 24) < 1e-6
    except Exception:
        return False

def evaluate_game24_answer(answer, nums, open_think="<think>", close_think="</think>"):
    """
    Evaluate a Game24 answer and return (is_correct, expr, error_message).
    
    Args:
        answer: Raw model output
        nums: Expected numbers to use
        
    Returns:
        Tuple of (is_correct, extracted_expression, error_message)
    """
    expr = extract_solution(answer, open_think, close_think)
    if not expr:
        return False, None, "No expression found"
    if evaluate_expression(expr, expected_nums=nums):
        return True, expr, "Correct solution (evaluates to 24 using exactly the given numbers)"
    else:
        used_nums = extract_numbers_from_expr(expr)
        if sorted(used_nums) != sorted(nums):
            return False, expr, f"Incorrect: Expression uses {used_nums}, expected {nums}"
        else:
            return False, expr, "Expression does not evaluate to 24"

def run_one(task):
    """Process a single Game24 example. Designed to run in a worker process.

    ``task`` is a tuple of ``(args, idx, nums, reason_dir)``.  Returns a result
    dict that the main process aggregates.
    """
    args, idx, nums, reason_dir = task
    main_model = args.model
    think_tags = get_think_tags(main_model)

    llm_server = init_llm_server(main_model, port=args.port)

    prompt = build_prompt(nums)
    full_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    if args.monitor:
        monitors = (ThinkingPhaseStepVerifierGame24Monitor(
            name="game24_verifier",
            original_numbers=nums,
            llm_server=llm_server,
            prompt=full_prompt,
            newline_threshold=args.newline_threshold,
            max_corrections=args.max_corrections,
            answer_start_token=think_tags['close'],
            warmup_newlines=args.warmup,
        ),)
    else:
        monitors = ()

    try:
        answer = asyncio.run(stream_completion(
            full_prompt,
            llm_server=llm_server,
            monitors=monitors,
            add_delay=False,
            termination_requires_validation=False,
            async_execution=True,
            tokenizer=tokenizer,
        ))
    except Exception as e:
        logger.error(f"Error running example {idx}: {e}")
        return None

    save_prompt(idx, answer, reason_dir)

    generated_tokens = count_tokens(answer, tokenizer)
    is_correct, expr, message = evaluate_game24_answer(
        answer, nums, think_tags['open'], think_tags['close']
    )
    gave_no_solution = (expr is not None and "no solution" in expr.strip().lower())
    no_expr_found = (expr is None)
    attempted = not (gave_no_solution or no_expr_found)

    return {
        "idx": int(idx),
        "numbers": list(nums),
        "expr": expr,
        "is_correct": bool(is_correct),
        "attempted": bool(attempted),
        "generated_tokens": int(generated_tokens),
        "message": message,
        "output_text": answer,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Game of 24 step-by-step solver with monitors")
    parser.add_argument("--num_examples", "-n", type=int, default=1362, help="Number of examples to run")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logs and single-process mode")
    parser.add_argument("--newline_threshold", type=int, default=20, help="Number of newlines in thinking before forcing step verification")
    parser.add_argument("--max_corrections", type=int, default=3, help="Maximum number of correction attempts per example")
    parser.add_argument("--warmup", type=int, default=4, help="Number of \\n to skip before starting side-chain verification")
    parser.add_argument("--model", type=str, default=MAIN_MODEL, help="Main model to use for generation")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--monitor", "-m", action="store_true", help="Enable thinking-phase step verification monitor (default: vanilla CoT)")
    parser.add_argument("--n_processes", "-p", type=int, default=16, help="Number of parallel worker processes")
    parser.add_argument("--n_exps", type=int, default=1, help="Number of independent sampling runs (produces outputs_solver_{i}.jsonl)")
    parser.add_argument("--extra", type=str, default="", help="Extra text description for the output directory")
    args = parser.parse_args()

    main_model = args.model
    N = args.num_examples

    # ---- Unique, timestamped run directory ----
    model_short = get_model_short_name(main_model)
    mode = "monitor" if args.monitor else "solveronly"
    run_name = f"{model_short}_{mode}"
    if args.monitor:
        run_name += f"_maxcorr{args.max_corrections}_nl{args.newline_threshold}"
    run_name += f"_nexps{args.n_exps}"
    if args.debug:
        run_name += "_debug"

    output_dir = os.path.join(
        _OUTPUT_ROOT, "Outputs_TTS", "Gameof24results",
        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}-{run_name}',
    )
    if args.extra:
        output_dir += f"-{args.extra}"
    reason_dir = os.path.join(output_dir, "Reasoning_output")
    os.makedirs(reason_dir, exist_ok=True)

    logfile = os.path.join(output_dir, "run.log")

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(logfile, mode="w")],
        force=True,
    )

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    logger.info(f"Main model: {main_model}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Monitor: {args.monitor} | Examples: {N} | Processes: {args.n_processes}")

    dataset = load_dataset("nlile/24-game", split="train")

    tokenizer = AutoTokenizer.from_pretrained(main_model, trust_remote_code=True)

    indices = np.linspace(0, len(dataset) - 1, N, dtype=int)

    for exp_i in range(args.n_exps):
        if args.n_exps > 1:
            print(f"\n=== Run {exp_i + 1}/{args.n_exps} ===")

        run_reason_dir = os.path.join(reason_dir, f"solver_{exp_i}")
        os.makedirs(run_reason_dir, exist_ok=True)
        outputs_file = os.path.join(output_dir, f"outputs_solver_{exp_i}.jsonl")
        results_file = os.path.join(output_dir, f"results_solver_{exp_i}.txt")

        tasks = [
            (args, int(idx), dataset[int(idx)]["numbers"], run_reason_dir)
            for idx in indices
        ]

        if args.debug:
            results = [run_one(t) for t in tqdm(tasks, desc="Game24")]
        else:
            with Pool(processes=args.n_processes) as pool:
                results = list(tqdm(
                    pool.imap_unordered(run_one, tasks),
                    total=len(tasks),
                    desc="Game24",
                ))

        results = [r for r in results if r is not None]

        with open(outputs_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        num_correct = sum(r["is_correct"] for r in results)
        num_attempted = sum(r["attempted"] for r in results)
        num_excluded = len(results) - num_attempted
        generated_token_counts = [r["generated_tokens"] for r in results]
        total_generated_tokens = sum(generated_token_counts)

        avg_generated_tokens = total_generated_tokens / N if N > 0 else 0
        accuracy = num_correct / N if N > 0 else 0
        soundness = num_correct / num_attempted if num_attempted > 0 else 0

        with open(results_file, "w") as f:
            f.write("Game of 24 Evaluation Results\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Model: {main_model}\n")
            f.write(f"Number of Examples: {N}\n\n")
            f.write("Results:\n")
            f.write("---------\n")
            f.write(f"Correct: {num_correct}/{N}\n")
            f.write(f"Accuracy: {accuracy:.2%}\n")
            f.write(f"Soundness: {num_correct}/{num_attempted} = {soundness:.2%}\n")
            f.write(f"Excluded from soundness (no solution / token budget exceeded): {num_excluded}\n\n")
            f.write("Generated Token Statistics:\n")
            f.write("---------------------------\n")
            f.write(f"Total Generated Tokens: {total_generated_tokens}\n")
            f.write(f"Average Generated Tokens: {avg_generated_tokens:.2f}\n")
            if generated_token_counts:
                f.write(f"Min Generated Tokens: {min(generated_token_counts)}\n")
                f.write(f"Max Generated Tokens: {max(generated_token_counts)}\n")
                f.write(f"Std Dev: {np.std(generated_token_counts):.2f}\n")

        logger.info(
            f"Run {exp_i}: Accuracy={accuracy:.2%} Soundness={soundness:.2%} "
            f"Results saved to {results_file}"
        )
