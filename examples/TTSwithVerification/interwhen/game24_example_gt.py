"""
Game of 24 experiment with thinking-phase step verification.

Uses ThinkingPhaseStepVerifierGame24Monitor which:
  - Verifies the model's intermediate expressions during <think> via side-streams
  - Injects expression extraction after </think>
  - Verifies the final \\boxed{} expression for correctness
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import numpy as np
import csv

from io import StringIO
from multiprocessing.pool import ThreadPool
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from interwhen import stream_completion
from interwhen.monitors import ThinkingPhaseStepVerifierGame24Monitor
from interwhen.monitors.thinkingPhaseVerifierGame24 import verify_expression

# ============== MODEL CONFIGURATION ==============
MAIN_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
# =================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Walk up to find the repo root (contains pyproject.toml), output to its parent
_dir = _SCRIPT_DIR
while _dir != os.path.dirname(_dir) and not os.path.isfile(os.path.join(_dir, "pyproject.toml")):
    _dir = os.path.dirname(_dir)
_OUTPUT_ROOT = os.path.dirname(_dir)

def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from the model path."""
    short_name = model_name.split("/")[-1]
    short_name = short_name.replace(" ", "_").replace(":", "-")
    return short_name

def get_output_dirs(main_model: str, base_dir: str = None):
    """Create and return output directory paths based on model name."""
    if base_dir is None:
        base_dir = os.path.join(_OUTPUT_ROOT, "Outputs_TTS", "Gameof24results_2")
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    
    dirs = {
        "base": output_base,
        "reasoning": os.path.join(output_base, "Reasoning_output"),
        "csv_saved": os.path.join(output_base, "csv_saved"),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def get_log_filename(main_model: str, num_examples: int, base_dir: str = None) -> str:
    """Generate log filename based on model name."""
    if base_dir is None:
        base_dir = os.path.join(_OUTPUT_ROOT, "Outputs_TTS", "Gameof24results")
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    os.makedirs(output_base, exist_ok=True)
    return os.path.join(output_base, f"EAT_{num_examples}examples.log")

def save_prompt(idx, prompt_with_answer, reason_dir):
    filename = os.path.join(reason_dir, f"reason_{idx}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_with_answer)

logger = logging.getLogger(__name__)

_real_stderr = sys.stderr


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def init_llm_server(modelname, max_tokens=32768, port=8000):
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": modelname,
        "max_tokens": max_tokens,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "do_sample": True,
        "temperature": 0.6,
        "stream": True,
        "logprobs": 20,
        "use_beam_search": False,
        "prompt_cache": True,
        "seed": 42
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}


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


def extract_solution(text):
    
    # Only search for \boxed{} AFTER </think> to avoid grabbing unverified
    # expressions from inside the thinking trace.
    # If model opened <think> but never closed it (hit token limit), there is
    # no final answer — return None.
    if '</think>' in text:
        search_text = text[text.rfind('</think>'):]
    elif '<think>' in text:
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

def evaluate_game24_answer(answer, nums):
    """
    Evaluate a Game24 answer and return (is_correct, expr, error_message).
    
    Args:
        answer: Raw model output
        nums: Expected numbers to use
        
    Returns:
        Tuple of (is_correct, extracted_expression, error_message)
    """
    expr = extract_solution(answer)
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Game of 24 step-by-step solver with monitors")
    parser.add_argument("--num_examples", "-n", type=int, default=1362, help="Number of examples to run")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logs")
    parser.add_argument("--newline_threshold", type=int, default=20, help="Number of newlines in thinking before forcing step verification")
    parser.add_argument("--max_corrections", type=int, default=3, help="Maximum number of correction attempts per example")
    parser.add_argument("--warmup", type=int, default=4, help="Number of \\n to skip before starting side-chain verification")
    parser.add_argument("--model", type=str, default=MAIN_MODEL, help="Main model to use for generation")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--k_runs", type=int, default=1, help="Best-of-K: sequential attempts per example (stop on first verifier pass)")
    parser.add_argument("--processes", "-p", type=int, default=1, help="Number of examples to process in parallel")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (attempt j uses seed+j)")
    parser.add_argument("--base_dir", type=str, default=None, help="Override base output directory")
    parser.add_argument("--summary_file", type=str, default="summary.json", help="Summary filename")
    args = parser.parse_args()

    main_model = args.model

    output_dirs = get_output_dirs(main_model, base_dir=args.base_dir)
    logfile = get_log_filename(main_model, args.num_examples, base_dir=args.base_dir)
    reason_dir = output_dirs["reasoning"]

    log_level = logging.DEBUG if args.debug else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logfile, mode="w"),
        ],
        force=True,
    )

    # Silence stdout/stderr globally; tqdm writes to _real_stderr so progress bar still shows.
    # All prints get redirected into the log file alongside logging output.
    _stdout_log = open(logfile, "a", buffering=1)
    sys.stdout = _stdout_log
    sys.stderr = _stdout_log

    logger.info(f"Main model: {main_model}")
    logger.info(f"Output directory: {output_dirs['base']}")
    logger.info(f"k_runs: {args.k_runs}, processes: {args.processes}, seed: {args.seed}")

    dataset = load_dataset("nlile/24-game", split="train")

    llm_server = init_llm_server(main_model, port=args.port)

    logger.info(f"Loading tokenizer for {main_model}...")
    tokenizer = AutoTokenizer.from_pretrained(main_model, trust_remote_code=True)
    logger.info("Tokenizer loaded successfully.")

    N = args.num_examples
    indices = np.linspace(0, len(dataset)-1, N, dtype=int)

    def process_example(idx):
        example = dataset[int(idx)]
        nums = example["numbers"]
        prompt = build_prompt(nums)
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        example_log_dir = os.path.join(reason_dir, f"example_{idx}")
        os.makedirs(example_log_dir, exist_ok=True)

        def run_attempt(j):
            attempt_seed = args.seed + j
            attempt_log = StringIO()
            attempt_log.write(f"=== Attempt {j} (seed={attempt_seed}) ===\n")

            attempt_llm = dict(llm_server)
            attempt_llm["payload"] = dict(llm_server["payload"])
            attempt_llm["payload"]["seed"] = attempt_seed

            monitor_final_answer = ThinkingPhaseStepVerifierGame24Monitor(
                name="game24_verifier",
                original_numbers=nums,
                llm_server=attempt_llm,
                prompt=full_prompt,
                newline_threshold=args.newline_threshold,
                max_corrections=args.max_corrections,
                answer_start_token="</think>",
                warmup_newlines=args.warmup,
            )

            monitor = ThinkingPhaseStepVerifierGame24Monitor(
                name="game24_verifier",
                original_numbers=nums,
                llm_server=attempt_llm,
                prompt=full_prompt,
                newline_threshold=args.newline_threshold,
                max_corrections=args.max_corrections,
                answer_start_token="</think>",
                warmup_newlines=args.warmup,
            )

            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = attempt_log
            sys.stderr = attempt_log
            try:
                answer = asyncio.run(stream_completion(
                    full_prompt,
                    llm_server=attempt_llm,
                    monitors=[monitor],
                    add_delay=False,
                    termination_requires_validation=False,
                    async_execution=True,
                ))
                attempt_log.write(f"\nANSWER:\n{answer}\n")
            except Exception as e:
                attempt_log.write(f"\nERROR: {e}\n")
                answer = ""
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

            expr_v = extract_solution(answer) if answer else None
            if expr_v:
                status, is_valid, _errors, _unused = verify_expression(expr_v, [float(x) for x in nums])
                v_passed = (is_valid and status == "complete")
            else:
                v_passed = False

            is_correct_a, expr_a, message_a = (
                evaluate_game24_answer(answer, nums) if answer else (False, None, "No answer")
            )
            generated_tokens = count_tokens(answer, tokenizer) if answer else 0
            gave_no_solution = (expr_a is not None and "no solution" in expr_a.strip().lower())
            no_expr_found = (expr_a is None)
            attempted = not (gave_no_solution or no_expr_found)

            attempt_log.write(f"\nExpr: {expr_v}, Verifier passed: {v_passed}\n")

            with open(os.path.join(example_log_dir, f"attempt_{j}.txt"), "w") as f:
                f.write(attempt_log.getvalue())
            save_prompt(int(idx), answer, example_log_dir)

            return {
                "j": j,
                "output": answer,
                "verifier_passed": v_passed,
                "final_correct": bool(is_correct_a),
                "expr": expr_a,
                "attempted": attempted,
                "message": message_a,
                "generated_tokens": generated_tokens,
            }

        # Sequential best-of-K with early-stop on verifier pass.
        attempts = []
        for j in range(args.k_runs):
            a = run_attempt(j)
            attempts.append(a)
            if a["verifier_passed"]:
                break
        num_attempts = len(attempts)

        # Pick a "final" attempt: first verifier-pass if any, else the last attempt
        chosen = next((a for a in attempts if a["verifier_passed"]), attempts[-1])

        return {
            "idx": int(idx),
            "nums": nums,
            "num_attempts": num_attempts,
            "verifier_passed": chosen["verifier_passed"],
            "final_correct": chosen["final_correct"],
            "attempts": attempts,
            "is_correct": chosen["final_correct"],
            "attempted": chosen["attempted"],
            "expr": chosen["expr"],
            "message": chosen["message"],
            "generated_tokens": chosen["generated_tokens"],
        }

    # Run in parallel across examples
    results = []
    num_correct = 0
    num_attempted = 0
    total_examples = 0

    with ThreadPool(processes=args.processes) as pool:
        for result in tqdm(
            pool.imap_unordered(process_example, indices),
            total=len(indices),
            desc="Processing examples",
            unit="example",
            file=_real_stderr,
        ):
            total_examples += 1
            if result["is_correct"]:
                num_correct += 1
            if result["attempted"]:
                num_attempted += 1
            results.append(result)

    # Compute stats
    accuracy = num_correct / total_examples if total_examples else 0
    soundness = num_correct / num_attempted if num_attempted else 0
    num_excluded = total_examples - num_attempted
    total_tokens = sum(r["generated_tokens"] for r in results)
    avg_tokens = total_tokens / total_examples if total_examples else 0
    avg_attempts = np.mean([r["num_attempts"] for r in results]) if results else 0

    # Save CSV
    results_csv = os.path.join(output_dirs["base"], "game24_results.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "nums", "num_attempts", "is_correct", "attempted", "expr", "message", "generated_tokens"])
        for r in results:
            writer.writerow([r["idx"], r["nums"], r["num_attempts"], r["is_correct"], r["attempted"], r["expr"], r["message"], r["generated_tokens"]])

    # Save summary JSON
    summary = {
        "model": main_model,
        "k_runs": args.k_runs,
        "seed": args.seed,
        "processes": args.processes,
        "total_examples": total_examples,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "num_attempted": num_attempted,
        "soundness": soundness,
        "num_excluded": num_excluded,
        "avg_tokens": avg_tokens,
        "total_tokens": total_tokens,
        "avg_attempts": float(avg_attempts),
        "results": results,
    }

    summary_path = os.path.join(output_dirs["base"], args.summary_file)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*50}", file=_real_stderr)
    print(f"FINAL RESULTS - Game of 24", file=_real_stderr)
    print(f"{'='*50}", file=_real_stderr)
    print(f"Model: {main_model}", file=_real_stderr)
    print(f"k_runs: {args.k_runs}", file=_real_stderr)
    print(f"Total examples: {total_examples}", file=_real_stderr)
    print(f"Accuracy: {num_correct}/{total_examples} ({accuracy:.2%})", file=_real_stderr)
    print(f"Soundness: {num_correct}/{num_attempted} ({soundness:.2%})", file=_real_stderr)
    print(f"Excluded: {num_excluded}", file=_real_stderr)
    print(f"Avg tokens: {avg_tokens:.2f}", file=_real_stderr)
    print(f"Avg attempts: {avg_attempts:.2f}", file=_real_stderr)
    print(f"Results: {results_csv}", file=_real_stderr)
    print(f"Summary: {summary_path}", file=_real_stderr)
    print(f"Logs: {reason_dir}/", file=_real_stderr)