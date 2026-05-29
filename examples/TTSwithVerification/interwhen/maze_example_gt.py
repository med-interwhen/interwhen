"""
Maze experiment with thinking-phase step verification.

Uses ThinkingPhaseStepVerifierMazeMonitor which:
  - Verifies the model's traced path during <think> via side-streams
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
from multiprocessing.pool import ThreadPool
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from interwhen import stream_completion
from interwhen.monitors import ThinkingPhaseStepVerifierMazeMonitor
from interwhen.utils.maze_verifier import parse_maze_from_prompt

logging.basicConfig(level=logging.INFO, format='%(message)s')
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
        base_dir = os.path.join(_OUTPUT_ROOT, "Outputs_TTS", "MazeResults_final_answer_verification")
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

def remove_last_paragraph(s: str) -> str:
    return s[:-143]

def build_prompt_from_example(example): #(original prompt config)

    pre_prompt = "You are an expert problem solver. Carefully read the following multiple-choice question and think through the solution step-by-step before providing your final answer. Provide your final answer option by enclosing it within \\boxed{A/B/C/D}.:"
    description = example.get("prompt")
    description = str(description)
    description = remove_last_paragraph(description)
    return pre_prompt, description


def extract_solution_mcq(text):
    """Extract MCQ solution from model output."""
    patterns = [
        r"\\boxed\{([^}]*)\}",
        r"boxed\{([^}]*)\}",
        r"\*\*([A-D])\*\*",
        r"answer[:\s]*([A-D])",
        r"(?:^|\n)([A-D])(?:\s|$|\.)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
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


def save_prompt(idx, prompt_with_answer, reason_dir):
    """Save reasoning trace to file."""
    os.makedirs(reason_dir, exist_ok=True)
    filename = os.path.join(reason_dir, f"reason_{idx}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_with_answer)
    logger.info(f"Saved reasoning trace to {filename}")


def get_log_filename(main_model: str, num_examples: int, base_dir: str = None) -> str:
    """Generate log filename based on model name."""
    if base_dir is None:
        base_dir = os.path.join(_OUTPUT_ROOT, "Outputs_TTS", "MazeResults_final_answer_verification")
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    os.makedirs(output_base, exist_ok=True)
    return os.path.join(output_base, f"EAT_{num_examples}examples.log")


def evaluate_mcq_answer(answer, options, ground_truth):
    sol = extract_solution_mcq(answer)
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
    parser = argparse.ArgumentParser(description="Run maze experiments with step verification")
    parser.add_argument("--model", type=str, default=MAIN_MODEL,
                        help="Model name for generation")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated indices to run (e.g., '3000,3500,4000')")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=10, help="End index")
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
    parser.add_argument("--k_runs", type=int, default=1,
                        help="Number of best-of-K attempts to run per example (in parallel)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; attempt j uses seed = args.seed + j")
    parser.add_argument("--processes", type=int, default=1,
                        help="Number of examples to process in parallel")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Override output base directory")
    parser.add_argument("--summary_file", type=str, default="summary.json",
                        help="Filename for the summary JSON written under the output base dir")
    args = parser.parse_args()

    logger.info(f"Thinking-phase verification: always on")
    logger.info(f"  Newline threshold: {args.newline_threshold}")
    logger.info(f"  Warmup: {args.warmup}")
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load dataset
    dataset = load_dataset("microsoft/VISION_LANGUAGE", 'maze_text_only', split='val')
    
    # Setup LLM server
    llm_server = init_llm_server(args.model, port=args.port)
    
    # Load tokenizer for accurate token counting
    logger.info(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    logger.info("Tokenizer loaded successfully.")
    
    # Setup output directory
    output_dirs = get_output_dirs(args.model, base_dir=args.base_dir)
    reason_dir = output_dirs["reasoning"]

    # Setup logging - file only (tqdm handles console progress)
    log_level = logging.DEBUG if args.debug else logging.INFO
    logfile = os.path.join(output_dirs["base"], "maze.log")
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

    # Determine indices
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]
    elif args.num_examples:
        # Use 1499 as endpoint (1500 is out of bounds since dataset size is 1500)
        indices = np.linspace(0, 1499, args.num_examples, dtype=int)
    else:
        indices = range(args.start, 1500)
    
    # Stats tracking
    results = []
    total_correct = 0
    total_examples = 0
    total_reasoning_tokens = 0
    num_attempted = 0  # examples where a \boxed{} answer was produced
    reasoning_token_counts = []
    per_example_results = []  # list of dicts for CSV

    def process_example(idx):
        """Process a single example: run K attempts in parallel, choose first verifier-pass."""
        idx_int = int(idx)
        try:
            example = dataset[idx_int]
            pre_prompt, user_prompt = build_prompt_from_example(example)
            if str(example.get("ground_truth", "")).strip() == "Q4":
                target_options = ["A", "B"]
            else:
                target_options = ["A", "B", "C", "D"]
            keys = "|".join(map(re.escape, target_options))
            pattern = rf'\b({keys})\.\s*([A-Za-z0-9]+)\b'
            options = dict(re.findall(pattern, user_prompt))

            full_prompt = (
                f"<|im_start|>system\n{pre_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            # Parse maze from prompt
            grid, start_pos, exit_pos = parse_maze_from_prompt(user_prompt)
            if not grid or not start_pos or not exit_pos:
                logger.error(f"Could not parse maze for example {idx_int}")
                return None

            # Detect question type from prompt (auto-detection)
            question_type = ThinkingPhaseStepVerifierMazeMonitor.detect_question_type(user_prompt)
            gt_sol = str(example.get("ground_truth", "")).strip()

            example_log_dir = os.path.join(reason_dir, f"example_{idx_int}")
            os.makedirs(example_log_dir, exist_ok=True)

            def run_attempt(j):
                attempt_seed = args.seed + j
                attempt_log = StringIO()
                attempt_log.write(f"=== Attempt {j} (seed={attempt_seed}) ===\n")
                attempt_log.write(
                    f"Maze: S={start_pos}, E={exit_pos}, "
                    f"grid={len(grid)}x{len(grid[0]) if grid else 0}, "
                    f"qtype={question_type}\n"
                )

                attempt_llm = dict(llm_server)
                attempt_llm["payload"] = dict(llm_server["payload"])
                attempt_llm["payload"]["seed"] = attempt_seed

                monitor = ThinkingPhaseStepVerifierMazeMonitor(
                    name="maze_thinking_verifier",
                    grid=grid,
                    start_pos=start_pos,
                    exit_pos=exit_pos,
                    llm_server=attempt_llm,
                    prompt=full_prompt,
                    question_type=question_type,
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
                    attempt_log.write(f"\nERROR running example {idx_int} attempt {j}: {e}\n")
                    import traceback
                    traceback.print_exc(file=attempt_log)
                    answer = ""
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr


                sol = extract_solution_mcq(answer) if answer else None
                v_passed = (sol is not None and sol.strip().lower() != "no solution")

                # Ground-truth evaluation
                is_correct, extracted_answer, message = (
                    evaluate_mcq_answer(answer, options, gt_sol) if answer
                    else (False, None, "No answer")
                )
                reasoning_tokens = count_tokens(answer, tokenizer) if answer else 0
                attempted = (extracted_answer is not None
                             and extracted_answer.strip().lower() != "no solution")

                attempt_log.write(
                    f"\nResult: sol={extracted_answer}, gt={gt_sol}, "
                    f"correct={is_correct}, attempted={attempted}, "
                    f"verifier_passed={v_passed}\n{message}\n"
                )

                with open(os.path.join(example_log_dir, f"attempt_{j}.txt"), "w") as f:
                    f.write(attempt_log.getvalue())
                save_prompt(idx_int, answer, example_log_dir)

                return {
                    "j": j,
                    "output": answer,
                    "verifier_passed": v_passed,
                    "final_correct": bool(is_correct),
                    # Task-specific extras (not required by analysis tooling).
                    "sol": extracted_answer if extracted_answer else "",
                    "attempted": attempted,
                    "tokens": int(reasoning_tokens),
                    "message": message,
                }

            # Run attempts sequentially with early-stop on verifier pass.
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
                "idx": idx_int,
                "question_type": question_type,
                "num_attempts": num_attempts,
                "verifier_passed": chosen["verifier_passed"],
                "final_correct": chosen["final_correct"],
                "attempts": attempts,
                "correct": chosen["final_correct"],
                "attempted": chosen["attempted"],
                "sol": chosen["sol"],
                "gt": gt_sol,
                "tokens": chosen["tokens"],
                "message": chosen["message"],
            }
        except Exception as e:
            logger.error(f"FATAL ERROR in example {idx_int}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Run examples in parallel; each example runs K attempts in parallel.
    with ThreadPool(processes=args.processes) as pool:
        for result in tqdm(
            pool.imap_unordered(process_example, indices),
            total=len(indices),
            desc="Processing examples",
            unit="example",
            file=_real_stderr,
        ):
            if result is None:
                continue

            total_examples += 1
            if result["correct"]:
                total_correct += 1
            if result["attempted"]:
                num_attempted += 1
            total_reasoning_tokens += result["tokens"]
            reasoning_token_counts.append(result["tokens"])

            results.append({
                "idx": result["idx"],
                "question_type": result["question_type"],
                "num_attempts": result.get("num_attempts", 1),
                "verifier_passed": result.get("verifier_passed", False),
                "final_correct": result.get("final_correct", result["correct"]),
                "attempts": result.get("attempts", []),
                "correct": result["correct"],
                "attempted": result["attempted"],
                "sol": result["sol"],
                "gt": result["gt"],
                "reasoning_tokens": result["tokens"],
            })
            per_example_results.append({
                "index": result["idx"],
                "question_type": result["question_type"],
                "correct": result["correct"],
                "attempted": result["attempted"],
                "sol": result["sol"],
                "gt": result["gt"],
                "tokens": result["tokens"],
                "message": result["message"],
            })
    
    # Compute final metrics
    accuracy = total_correct / total_examples if total_examples > 0 else 0
    soundness = total_correct / num_attempted if num_attempted > 0 else 0  # correct / attempted
    avg_reasoning_tokens = total_reasoning_tokens / total_examples if total_examples > 0 else 0
    avg_attempts = (
        float(np.mean([r.get("num_attempts", 1) for r in results])) if results else 0.0
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total examples: {total_examples}")
    logger.info(f"Correct: {total_correct}")
    logger.info(f"Attempted (produced \\boxed answer): {num_attempted}/{total_examples}")
    logger.info(f"Accuracy: {accuracy:.4f} ({total_correct}/{total_examples})")
    logger.info(f"Soundness: {soundness:.4f} ({total_correct}/{num_attempted})")
    logger.info(f"Total reasoning tokens: {total_reasoning_tokens}")
    logger.info(f"Avg reasoning tokens: {avg_reasoning_tokens:.1f}")
    
    print(f"\n{'='*50}", file=_real_stderr)
    print(f"FINAL RESULTS", file=_real_stderr)
    print(f"{'='*50}", file=_real_stderr)
    print(f"Model: {args.model}", file=_real_stderr)
    print(f"k_runs: {args.k_runs}", file=_real_stderr)
    print(f"Total examples: {total_examples}", file=_real_stderr)
    print(f"Accuracy: {total_correct}/{total_examples} ({accuracy:.2%})", file=_real_stderr)
    print(f"Soundness: {total_correct}/{num_attempted} ({soundness:.2%})", file=_real_stderr)
    print(f"Avg attempts: {avg_attempts:.2f}", file=_real_stderr)
    print(f"Avg reasoning tokens: {avg_reasoning_tokens:.2f}", file=_real_stderr)
    print(f"Total reasoning tokens: {total_reasoning_tokens}", file=_real_stderr)
    
    # Save per-example CSV
    csv_file = os.path.join(output_dirs["csv_saved"], f"results_{total_examples}examples.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["index", "question_type", "correct", "attempted", "sol", "gt", "tokens", "message"])
        writer.writeheader()
        writer.writerows(per_example_results)
    logger.info(f"Per-example CSV saved to {csv_file}")
    
    # Save summary
    summary = {
        'model': args.model,
        'k_runs': args.k_runs,
        'seed': args.seed,
        'processes': args.processes,
        'total_examples': total_examples,
        'correct': total_correct,
        'attempted': num_attempted,
        'accuracy': accuracy,
        'soundness': soundness,
        'total_reasoning_tokens': total_reasoning_tokens,
        'avg_reasoning_tokens': avg_reasoning_tokens,
        'avg_attempts': avg_attempts,
        'max_corrections': args.max_corrections,
        'results': results,
    }

    summary_path = os.path.join(output_dirs["base"], args.summary_file)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nSaved summary to {summary_path}")
    
    # Save results summary to a text file
    results_file = os.path.join(output_dirs["base"], f"EAT_{total_examples}examples_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Maze Step Verification Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Number of Examples: {total_examples}\n")
        f.write(f"Max Corrections: {args.max_corrections}\n")
        f.write(f"Newline Threshold: {args.newline_threshold}\n")
        f.write(f"Warmup: {args.warmup}\n")
        f.write(f"\n")
        f.write(f"Results:\n")
        f.write(f"---------\n")
        f.write(f"Correct: {total_correct}/{total_examples}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Attempted (produced \\boxed answer): {num_attempted}/{total_examples}\n")
        f.write(f"Soundness (correct/attempted): {soundness:.2%}\n\n")
        f.write(f"Token Statistics:\n")
        f.write(f"---------------------------\n")
        f.write(f"Total Tokens: {total_reasoning_tokens}\n")
        f.write(f"Average Tokens: {avg_reasoning_tokens:.2f}\n")
        if reasoning_token_counts:
            f.write(f"Median Tokens: {float(np.median(reasoning_token_counts)):.0f}\n")
            f.write(f"Min Tokens: {min(reasoning_token_counts)}\n")
            f.write(f"Max Tokens: {max(reasoning_token_counts)}\n")
            f.write(f"Std Dev: {np.std(reasoning_token_counts):.2f}\n")
    
    logger.info(f"Results saved to {results_file}")
    print(f"Results: {results_file}", file=_real_stderr)
    print(f"Summary: {summary_path}", file=_real_stderr)
    print(f"CSV: {csv_file}", file=_real_stderr)