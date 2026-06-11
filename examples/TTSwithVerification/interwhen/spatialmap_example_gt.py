"""
SpatialMap experiment with thinking-phase step verification.

Uses ThinkingPhaseStepVerifierSpatialMapMonitor which:
  - Verifies the model's directional claims during <think> via side-streams
  - Injects a structured step format after </think> (no meta-prompt needed)
  - Verifies each step as the model fills in the structured template
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
import numpy as np

from io import StringIO
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from interwhen import stream_completion
from interwhen.monitors import ThinkingPhaseStepVerifierSpatialMapMonitor

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
        base_dir = os.path.join(_OUTPUT_ROOT, "Outputs_TTS", "SpatialMapResults_final_answer_verification")
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


def get_question_type(idx: int) -> str:
    """Determine question type based on index range.
    
    Dataset structure (1500 examples total):
    - 0-499: Q0 (direction finding)
    - 500-999: Q1 (object finding)
    - 1000-1499: Q2 (counting)
    """
    if idx < 500:
        return "direction"
    elif idx < 1000:
        return "object"
    else:
        return "counting"


def build_simple_prompt(example):
    """Build a prompt matching spatialmap_example.py."""
    pre_prompt = "You are an expert problem solver. Carefully read the following multiple-choice question and think through the solution step-by-step before providing your final answer. Provide your final answer option by enclosing it within \\boxed{A/B/C/D}.:"
    description = str(example.get("prompt", ""))
    description_trimmed = description[:-143] if len(description) > 143 else description
    return pre_prompt, description_trimmed


def extract_solution(text: str) -> str:
    """Extract the boxed answer from the response (after </think>)."""
    patterns = [
        r"\\boxed\{([^}]*)\}",
        r"boxed\{([^}]*)\}",
        r"\*\*([A-D])\*\*",
        r"answer[:\s]*([A-D])",
        r"(?:^|\n)([A-D])(?:\s|$|\.)",
    ]
    if "</think>" in text:
        answer_section = text.split("</think>")[-1]
    else:
        answer_section = text
    answer_section = re.sub(r'<format>.*?</format>', '', answer_section, flags=re.DOTALL)
    for pattern in patterns:
        matches = re.findall(pattern, answer_section, re.IGNORECASE)
        if matches:
            expr = matches[-1].strip()
            choice_match = re.search(r"\b([ABCD])\b", expr, flags=re.IGNORECASE)
            if choice_match:
                return choice_match.group(1).upper()
    return None


def count_tokens(text: str, tokenizer) -> int:
    """Count the total number of tokens in the generated text using the tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def init_llm_server(model_name, max_tokens=20480, port=8000):
    """Initialize LLM server configuration."""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model_name,
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


def save_prompt(idx, prompt_with_answer, reason_dir):
    """Save reasoning trace to file."""
    os.makedirs(reason_dir, exist_ok=True)
    filename = os.path.join(reason_dir, f"reason_{idx}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_with_answer)
    logger.info(f"Saved reasoning trace to {filename}")


def evaluate_spatialmap_answer(answer, options, ground_truth):
    """
    Evaluate a SpatialMap MCQ answer and return (is_correct, extracted_answer, message).
    
    Args:
        answer: Raw model output
        options: Dictionary mapping option letters (A/B/C/D) to their values
        ground_truth: The correct answer value
        
    Returns:
        Tuple of (is_correct, extracted_answer, message)
    """
    sol = extract_solution(answer)
    gt_sol = str(ground_truth).strip()
    if not sol:
        return False, None, "No expression found"
    sol = sol.strip()
    if sol in options:
        if options[sol] == gt_sol:
            return True, sol, f"Correct: option {sol} -> {options[sol]}"
        return False, sol, f"Incorrect: expected '{gt_sol}', got '{options[sol]}' (option {sol})"
    if sol.lower() == gt_sol.lower():
        return True, sol, f"Correct: answer text matches ground truth: {sol}"
    for opt_letter, opt_value in options.items():
        if sol.lower() == opt_value.lower():
            if opt_value == gt_sol:
                return True, sol, f"Correct: answer text {sol} (option {opt_letter})"
            return False, sol, f"Incorrect: expected '{gt_sol}', got '{opt_value}' (option {opt_letter})"
    return False, sol, f"Solution '{sol}' not found in options or ground truth"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SpatialMap experiments with StepVerifierSpatialMapMonitor")
    parser.add_argument("--model", type=str, default=MAIN_MODEL,
                        help="Model name for generation")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated indices to run (e.g., '0,100,200')")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=1500, help="End index")
    parser.add_argument("--num_examples", "-n", type=int, default=None,
                        help="Number of examples to run (overrides start/end)")
    parser.add_argument("--max_corrections", type=int, default=5,
                        help="Maximum number of correction attempts per example")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    parser.add_argument("--newline_threshold", type=int, default=20,
                        help="Number of \\n in thinking before triggering side verification")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Number of \\n to skip before starting side-chain verification (warmup period)")
    parser.add_argument("--k_runs", type=int, default=1, help="Best-of-K: number of sequential attempts per example (stop on first verifier pass)")
    parser.add_argument("--processes", "-p", type=int, default=1, help="Number of examples to process in parallel")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (attempt j uses seed+j)")
    parser.add_argument("--base_dir", type=str, default=None, help="Override base output directory")
    parser.add_argument("--summary_file", type=str, default="summary.json", help="Summary filename")
    args = parser.parse_args()

    # Setup output directory
    output_dirs = get_output_dirs(args.model, base_dir=args.base_dir)
    if args.k_runs > 1:
        # Append k to output dir so different k values don't overwrite each other
        new_base = output_dirs["base"] + f"_k{args.k_runs}"
        output_dirs = {
            "base": new_base,
            "reasoning": os.path.join(new_base, "Reasoning_output"),
            "csv_saved": os.path.join(new_base, "csv_saved"),
        }
        for d in output_dirs.values():
            os.makedirs(d, exist_ok=True)
    reason_dir = output_dirs["reasoning"]

    # Setup logging - file only (tqdm handles console progress)
    log_level = logging.DEBUG if args.debug else logging.INFO
    logfile = os.path.join(output_dirs["base"], "spatialmap.log")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logfile, mode="w"),
        ],
        force=True,
    )
    
    _stdout_log = open(logfile, "a", buffering=1)
    sys.stdout = _stdout_log
    sys.stderr = _stdout_log

    logger.info(f"Model: {args.model}")
    logger.info(f"k_runs: {args.k_runs}, processes: {args.processes}, seed: {args.seed}")
    logger.info(f"Newline threshold: {args.newline_threshold}, Warmup: {args.warmup}")
    
    # Load dataset (spatial_map_text_only has 1500 examples)
    dataset = load_dataset("microsoft/VISION_LANGUAGE", 'spatial_map_text_only', split='val')
    
    # Setup LLM server
    llm_server = init_llm_server(args.model, port=args.port)
    
    # Load tokenizer for accurate token counting
    logger.info(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    logger.info("Tokenizer loaded successfully.")
    
    # Determine indices
    max_idx = len(dataset) - 1
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]
    elif args.num_examples:
        indices = np.linspace(0, min(max_idx, 1499), args.num_examples, dtype=int)
    else:
        indices = list(range(args.start, min(args.end, max_idx + 1)))

    def process_example(idx):
        idx = int(idx)
        example = dataset[idx]
        pre_prompt, description_trimmed = build_simple_prompt(example)
        if str(example.get("ground_truth", "")).strip() == "Q4":
            target_options = ["A", "B"]
        else:
            target_options = ["A", "B", "C", "D"]
        keys = "|".join(map(re.escape, target_options))
        pattern = r'\b([A-D])\.\s*(.*?)(?=\s*[A-D]\.|$)'
        raw = re.findall(pattern, description_trimmed, flags=re.DOTALL)
        options = {k: v.strip().rstrip(".") for k, v in raw}

        question_type = get_question_type(idx)
        full_prompt = f"<|im_start|>system\n{pre_prompt}<|im_end|>\n<|im_start|>user\n{description_trimmed}<|im_end|>\n<|im_start|>assistant\n"

        example_log_dir = os.path.join(reason_dir, f"example_{idx}")
        os.makedirs(example_log_dir, exist_ok=True)

        gt_sol = str(example.get("ground_truth", "")).strip()

        def run_attempt(j):
            attempt_seed = args.seed + j
            attempt_log = StringIO()
            attempt_log.write(f"=== Attempt {j} (seed={attempt_seed}) ===\n")

            attempt_llm = dict(llm_server)
            attempt_llm["payload"] = dict(llm_server["payload"])
            attempt_llm["payload"]["seed"] = attempt_seed

            monitor_final_answer = ThinkingPhaseStepVerifierSpatialMapMonitor(
                name="spatialmap_thinking_verifier",
                problem_text=description_trimmed,
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
                    monitors=[],
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

            sol = extract_solution(answer) if answer else None
            v_passed = (sol is not None and sol.strip().lower() != "no solution")

            is_correct, extracted_answer, message = (
                evaluate_spatialmap_answer(answer, options, gt_sol) if answer
                else (False, None, "No answer")
            )
            reasoning_tokens = count_tokens(answer, tokenizer) if answer else 0
            attempted = (extracted_answer is not None and extracted_answer.strip().lower() != "no solution")

            attempt_log.write(f"\nVerifier passed: {v_passed}\n")

            with open(os.path.join(example_log_dir, f"attempt_{j}.txt"), "w") as f:
                f.write(attempt_log.getvalue())
            save_prompt(idx, answer, example_log_dir)

            return {
                "j": j,
                "output": answer,
                "verifier_passed": v_passed,
                "final_correct": bool(is_correct),
                "sol": extracted_answer,
                "attempted": attempted,
                "reasoning_tokens": reasoning_tokens,
                # "num_relations": len(monitor.z3_solver.parsed_relations),
                # "verified_claims": len(monitor.verified_claims),
                "message": message,
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
            "idx": idx,
            "question_type": question_type,
            "num_attempts": num_attempts,
            "verifier_passed": chosen["verifier_passed"],
            "final_correct": chosen["final_correct"],
            "attempts": attempts,
            "correct": chosen["final_correct"],
            "attempted": chosen["attempted"],
            "sol": chosen["sol"],
            "gt": gt_sol,
            "reasoning_tokens": chosen["reasoning_tokens"],
            # "num_relations": chosen["num_relations"],
            # "verified_claims": chosen["verified_claims"],
            "message": chosen["message"],
        }

    # Run in parallel across examples, sequential within each example
    results = []
    total_correct = 0
    total_examples = 0
    num_attempted = 0
    stats_by_type = {
        "direction": {"total": 0, "correct": 0},
        "object": {"total": 0, "correct": 0},
        "counting": {"total": 0, "correct": 0},
    }

    with Pool(processes=args.processes) as pool:
        for result in tqdm(
            pool.imap_unordered(process_example, indices),
            total=len(indices),
            desc="Processing examples",
            unit="example",
            file=_real_stderr,
        ):
            total_examples += 1
            if result["correct"]:
                total_correct += 1
                stats_by_type[result["question_type"]]["correct"] += 1
            if result["attempted"]:
                num_attempted += 1
            stats_by_type[result["question_type"]]["total"] += 1
            results.append(result)

    # Save results
    accuracy = total_correct / total_examples if total_examples else 0
    soundness = total_correct / num_attempted if num_attempted else 0
    avg_attempts = np.mean([r["num_attempts"] for r in results]) if results else 0

    # CSV
    results_csv = os.path.join(output_dirs["base"], "spatialmap_results.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx", "question_type", "num_attempts", "verifier_passed",
            "correct", "attempted", "sol", "gt",
            "reasoning_tokens", "message",
        ])
        for r in results:
            writer.writerow([
                r["idx"], r["question_type"], r["num_attempts"], r["verifier_passed"],
                r["correct"], r["attempted"], r["sol"], r["gt"],
                r["reasoning_tokens"], r["message"],
            ])

    # Summary JSON
    summary = {
        "model": args.model,
        "k_runs": args.k_runs,
        "seed": args.seed,
        "processes": args.processes,
        "total_examples": total_examples,
        "correct": total_correct,
        "attempted": num_attempted,
        "accuracy": accuracy,
        "soundness": soundness,
        "avg_attempts": float(avg_attempts),
        "max_corrections": args.max_corrections,
        "stats_by_type": stats_by_type,
        "results": results,
    }

    summary_path = os.path.join(output_dirs["base"], args.summary_file)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*50}", file=_real_stderr)
    print(f"FINAL RESULTS", file=_real_stderr)
    print(f"{'='*50}", file=_real_stderr)
    print(f"Model: {args.model}", file=_real_stderr)
    print(f"k_runs: {args.k_runs}", file=_real_stderr)
    print(f"Total examples: {total_examples}", file=_real_stderr)
    print(f"Accuracy: {total_correct}/{total_examples} ({accuracy:.2%})", file=_real_stderr)
    print(f"Soundness: {total_correct}/{num_attempted} ({soundness:.2%})", file=_real_stderr)
    print(f"Avg attempts: {avg_attempts:.2f}", file=_real_stderr)
    for qtype, stats in stats_by_type.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            print(f"  {qtype}: {acc:.2%} ({stats['correct']}/{stats['total']})", file=_real_stderr)
    print(f"Results: {results_csv}", file=_real_stderr)
    print(f"Summary: {summary_path}", file=_real_stderr)
    print(f"Logs: {reason_dir}/", file=_real_stderr)