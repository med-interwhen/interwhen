"""
SpatialMap experiment with thinking-phase step verification.

Uses ThinkingPhaseStepVerifierSpatialMapMonitor which:
  - Verifies the model's directional claims during the think-open tag via side-streams
  - Injects a structured step format after the think-close tag (no meta-prompt needed)
  - Verifies each step as the model fills in the structured template
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
from interwhen.monitors import ThinkingPhaseStepVerifierSpatialMapMonitor
from interwhen.utils.llm import init_llm_server, get_think_tags

logger = logging.getLogger(__name__)

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
dataset = None
reason_dir = None


def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from the model path."""
    short_name = model_name.split("/")[-1]
    short_name = short_name.replace(" ", "_").replace(":", "-")
    return short_name


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


def extract_solution(text: str, close_think: str = "</think>") -> str:
    """Extract the boxed answer from the response (after the think-close tag)."""
    patterns = [
        r"\\boxed\{([^}]*)\}",
        r"boxed\{([^}]*)\}",
        r"\*\*([A-D])\*\*",
        r"answer[:\s]*([A-D])",
        r"(?:^|\n)([A-D])(?:\s|$|\.)",
    ]
    if close_think in text:
        answer_section = text.split(close_think)[-1]
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


def save_prompt(idx, prompt_with_answer, reason_dir):
    """Save reasoning trace to file."""
    os.makedirs(reason_dir, exist_ok=True)
    filename = os.path.join(reason_dir, f"reason_{idx}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_with_answer)


def evaluate_spatialmap_answer(answer, options, ground_truth, close_think="</think>"):
    """
    Evaluate a SpatialMap MCQ answer and return (is_correct, extracted_answer, message).
    
    Args:
        answer: Raw model output
        options: Dictionary mapping option letters (A/B/C/D) to their values
        ground_truth: The correct answer value
        
    Returns:
        Tuple of (is_correct, extracted_answer, message)
    """
    sol = extract_solution(answer, close_think)
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


def run_one(task):
    """Process a single SpatialMap example in a worker process.

    ``task`` is ``(args, idx)``.  Returns a result dict that the main process
    aggregates, or ``None`` on failure.
    """
    args, idx = task
    main_model = args.model
    think_tags = get_think_tags(main_model)

    llm_server = init_llm_server(main_model, context_length=20480, port=args.port)

    example = dataset[idx]
    pre_prompt, description_trimmed = build_simple_prompt(example)

    pattern = r'\b([A-D])\.\s*(.*?)(?=\s*[A-D]\.|$)'
    raw = re.findall(pattern, description_trimmed, flags=re.DOTALL)
    options = {k: v.strip().rstrip(".") for k, v in raw}

    question_type = get_question_type(idx)

    full_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": pre_prompt}, {"role": "user", "content": description_trimmed}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    num_relations = 0
    verified_claims = 0
    if args.monitor:
        monitor = ThinkingPhaseStepVerifierSpatialMapMonitor(
            name="spatialmap_thinking_verifier",
            problem_text=description_trimmed,
            llm_server=llm_server,
            prompt=full_prompt,
            newline_threshold=args.newline_threshold,
            max_corrections=args.max_corrections,
            answer_start_token=think_tags['close'],
            warmup_newlines=args.warmup,
        )
        monitors = (monitor,)
    else:
        monitor = None
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

    save_prompt(int(idx), answer, reason_dir)

    generated_tokens = count_tokens(answer, tokenizer)
    gt_sol = str(example.get("ground_truth", "")).strip()
    is_correct, extracted_answer, message = evaluate_spatialmap_answer(
        answer, options, gt_sol, think_tags['close']
    )
    attempted = (extracted_answer is not None and extracted_answer.strip().lower() != "no solution")

    if monitor is not None:
        num_relations = len(monitor.z3_solver.parsed_relations)
        verified_claims = len(monitor.verified_claims)

    return {
        "idx": int(idx),
        "question_type": question_type,
        "sol": extracted_answer,
        "gt": gt_sol,
        "is_correct": bool(is_correct),
        "attempted": bool(attempted),
        "generated_tokens": int(generated_tokens),
        "num_relations": int(num_relations),
        "verified_claims": int(verified_claims),
        "message": message,
        "output_text": answer,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SpatialMap step-by-step solver with monitors")
    parser.add_argument("--num_examples", "-n", type=int, default=1500, help="Number of examples to run")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logs and single-process mode")
    parser.add_argument("--newline_threshold", type=int, default=20, help="Number of newlines in thinking before forcing step verification")
    parser.add_argument("--max_corrections", type=int, default=5, help="Maximum number of correction attempts per example")
    parser.add_argument("--warmup", type=int, default=0, help="Number of \\n to skip before starting side-chain verification")
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
        _OUTPUT_ROOT, "Outputs_TTS", "SpatialMapResults",
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

    dataset = load_dataset("microsoft/VISION_LANGUAGE", "spatial_map_text_only", split="val")

    tokenizer = AutoTokenizer.from_pretrained(main_model, trust_remote_code=True)

    # Dataset has 1500 examples; 1499 is the last valid index
    indices = np.linspace(0, len(dataset) - 1, N, dtype=int)

    for exp_i in range(args.n_exps):
        if args.n_exps > 1:
            print(f"\n=== Run {exp_i + 1}/{args.n_exps} ===")

        reason_dir = os.path.join(output_dir, "Reasoning_output", f"solver_{exp_i}")
        os.makedirs(reason_dir, exist_ok=True)
        outputs_file = os.path.join(output_dir, f"outputs_solver_{exp_i}.jsonl")
        results_file = os.path.join(output_dir, f"results_solver_{exp_i}.txt")

        tasks = [(args, int(idx)) for idx in indices]

        if args.debug:
            results = [run_one(t) for t in tqdm(tasks, desc="SpatialMap")]
        else:
            with Pool(processes=args.n_processes) as pool:
                results = list(tqdm(
                    pool.imap_unordered(run_one, tasks),
                    total=len(tasks),
                    desc="SpatialMap",
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

        # Per-type breakdown
        stats_by_type = {}
        for r in results:
            qt = r["question_type"]
            s = stats_by_type.setdefault(qt, {"total": 0, "correct": 0})
            s["total"] += 1
            s["correct"] += int(r["is_correct"])

        with open(results_file, "w") as f:
            f.write("SpatialMap Evaluation Results\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Model: {main_model}\n")
            f.write(f"Number of Examples: {N}\n\n")
            f.write("Results:\n")
            f.write("---------\n")
            f.write(f"Correct: {num_correct}/{N}\n")
            f.write(f"Accuracy: {accuracy:.2%}\n")
            f.write(f"Soundness: {num_correct}/{num_attempted} = {soundness:.2%}\n")
            f.write(f"Excluded from soundness (no answer): {num_excluded}\n\n")
            f.write("Per-type Breakdown:\n")
            f.write("---------------------------\n")
            for qtype, stats in stats_by_type.items():
                if stats["total"] > 0:
                    acc = stats["correct"] / stats["total"]
                    f.write(f"  {qtype}: {acc:.2%} ({stats['correct']}/{stats['total']})\n")
            f.write("\nGenerated Token Statistics:\n")
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
