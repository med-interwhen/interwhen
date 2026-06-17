"""
VERINA Specification Generation Benchmark with Step Verification

This script evaluates LLM-generated specifications (preconditions and postconditions)
on the VERINA benchmark using soundness and completeness metrics, integrated with
the StepVerifierVerinaSpecMonitor for streaming verification.

Soundness: Tests that the spec correctly rejects invalid inputs/outputs
Completeness: Tests that the spec correctly accepts valid inputs/outputs

Usage:
    python verina_specgen.py --num_examples 50
    python verina_specgen.py --debug
"""

import argparse
import asyncio
import json
import logging
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

from interwhen.interject import stream_completion
from interwhen.monitors import StepVerifierVerinaSpecMonitor
from interwhen.utils.llm import init_llm_server, get_think_tags
from interwhen.utils.verina_spec_example_utils import *
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Model Config
MAIN_MODEL = "Qwen/QwQ-32B"
EARLYSTOP_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Walk up to find the repo root (contains pyproject.toml), output into it
_dir = Path(__file__).resolve().parent
while _dir != _dir.parent and not (_dir / "pyproject.toml").is_file():
    _dir = _dir.parent
_OUTPUT_ROOT = str(_dir)

# Module-level objects shared with worker processes (inherited via fork)
tokenizer = None
dataset = None
reason_dir = None

def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from the model path."""
    short_name = model_name.split("/")[-1]
    short_name = short_name.replace(" ", "_").replace(":", "-")
    return short_name


# Saving related utils

def save_reasoning_trace(idx: int, data_id: str, prompt_with_answer: str, reason_dir: str):
    """Save the full reasoning trace to a file"""
    filename = os.path.join(reason_dir, f"reason_{idx}_{data_id}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_with_answer)


def count_tokens(text, tokenizer):
    """Count the total number of tokens in the generated text using the tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def save_results_csv(results: list, output_path: str):
    """Save results to CSV file"""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx", "data_id", "compiles",
            "precond_sound_pass", "precond_sound_total",
            "precond_complete_pass", "precond_complete_total",
            "postcond_sound_pass", "postcond_sound_total",
            "postcond_complete_pass", "postcond_complete_total",
            "precond_correct", "postcond_correct",
            "spec_sound", "spec_complete", "full_spec_correct",
            "generated_tokens", "precond", "postcond", "num_times_forced","finally_wrong"
        ])
        for r in results:
            precond_escaped = r.get("precond", "").replace("\n", "\\n")
            postcond_escaped = r.get("postcond", "").replace("\n", "\\n")
            writer.writerow([
                r["idx"], 
                r["data_id"], 
                r["compiles"],
                r["precond_sound_pass"],
                r["precond_sound_total"],
                r["precond_complete_pass"],
                r["precond_complete_total"],
                r["postcond_sound_pass"],
                r["postcond_sound_total"],
                r["postcond_complete_pass"],
                r["postcond_complete_total"],
                r.get("precond_correct", False),
                r.get("postcond_correct", False),
                r.get("spec_sound", False),
                r.get("spec_complete", False),
                r.get("full_spec_correct", False),
                r.get("generated_tokens", 0),
                precond_escaped,
                postcond_escaped,
                r.get("num_times_forced", 0),
                r.get("finally_wrong", False)
            ])


# Sanity check to test if lean compilation works correctly
def test_lean_compile():
    """Test if Lean compile check works with valid and invalid code."""
    print("Testing Lean compile check...")
    clean_playground()
    
    # Test 1: Valid Lean code
    valid_code = """
def hello : Nat := 42
#check hello
theorem one_eq_one : 1 = 1 := rfl
"""
    lean_file = create_lean_file("test_valid", valid_code)
    success, output = check_lean_compile(lean_file)
    print(f"\n[Test 1] Valid code:")
    print(f"  Compiled successfully: {success}")
    if not success:
        print(f"  Error: {output[:300]}")
    
    # Test 2: Invalid Lean code (should fail)
    invalid_code = """
def broken : Nat := "not a nat"
"""
    lean_file2 = create_lean_file("test_invalid", invalid_code)
    success2, output2 = check_lean_compile(lean_file2)
    print(f"\n[Test 2] Invalid code:")
    print(f"  Compiled successfully: {success2} (expected: False)")
    
    # Summary
    print(f"\n" + "="*50)
    if success and not success2:
        print("Lean compile check is working correctly!")
    else:
        print("Lean compile check may have issues.")
        if not success:
            print("  - Valid code failed to compile")
        if success2:
            print("  - Invalid code unexpectedly compiled")
    print("="*50)
    
    return success and not success2


def run_one(task):
    """Process a single Verina spec example in a worker process.

    ``task`` is ``(args, idx)``.  Returns a result dict.  Uses the module-level
    ``tokenizer``, ``dataset`` and ``reason_dir`` globals (inherited via fork).
    """
    args, idx = task
    main_model = args.main_model
    think_tags = get_think_tags(main_model)

    llm_server = init_llm_server(main_model, context_length=20480, port=args.port)

    data = dataset[idx]
    prompt = build_full_prompt(data, tokenizer)

    # Convert BenchmarkData to dict for the monitor
    task_data = {
        "data_id": data.data_id,
        "description": data.description,
        "signature": data.signature,
        "lean_data": data.lean_data,
        "spec_desc": data.spec_desc,
        "tests": data.tests,
        "reject_inputs": data.reject_inputs,
        "metadata": data.metadata,
    }

    # Setup monitors
    if args.monitor:
        monitors = [
            StepVerifierVerinaSpecMonitor(
                name="VerinaSpecStepVerifier",
                task_data=task_data,
                llm_server=llm_server,
                prompt=prompt,
                k_steps=args.k_steps,
                compile_timeout=120,
                max_corrections=args.max_corrections,
                open_think=think_tags['open'],
                close_think=think_tags['close'],
                tokenizer=tokenizer,
            ),
        ]
    else:
        monitors = []

    try:
        answer = asyncio.run(
            stream_completion(
                prompt,
                prev_text="",
                llm_server=llm_server,
                monitors=monitors,
                add_delay=False,
                num_calls_index=0,
                async_execution=True,
                tokenizer=tokenizer,
            )
        )
        prompt_with_answer = prompt + answer
    except Exception as e:
        logger.error(f"Error during LLM generation for example {idx}: {e}")
        return {
            "idx": int(idx),
            "data_id": data.data_id,
            "compiles": False,
            "precond_sound_pass": 0,
            "precond_sound_total": 0,
            "precond_complete_pass": 0,
            "precond_complete_total": 0,
            "postcond_sound_pass": 0,
            "postcond_sound_total": 0,
            "postcond_complete_pass": 0,
            "postcond_complete_total": 0,
            "precond_correct": False,
            "postcond_correct": False,
            "spec_sound": False,
            "spec_complete": False,
            "full_spec_correct": False,
            "generated_tokens": 0,
            "precond": "",
            "postcond": "",
            "num_times_forced": 0,
            "finally_wrong": True,
            "output_text": "",
        }

    save_reasoning_trace(int(idx), data.data_id, prompt_with_answer, reason_dir)

    generated_tokens = count_tokens(answer, tokenizer)

    generated_spec = extract_spec_from_response(answer, think_tags['open'], think_tags['close'])
    logger.info(f"[{data.data_id}] Extracted precond: {generated_spec['precond'][:100]}...")
    logger.info(f"[{data.data_id}] Extracted postcond: {generated_spec['postcond'][:100]}...")

    # Final spec verification loop; retry if compilation fails
    if args.monitor and monitors and generated_spec.get("postcond") != "":
        final_spec, final_compiles, final_output, num_final_retries = asyncio.run(
            monitors[0].verify_final_spec(
                spec=generated_spec,
                prompt_with_answer=prompt_with_answer,
                max_retries=1
            )
        )
        if final_spec != generated_spec:
            logger.info(f"[Final verification] Spec fixed after {num_final_retries} retries")
            generated_spec = final_spec

    # check for soundness
    if args.monitor and monitors and generated_spec.get("postcond") != "":
        compiled, _ = monitors[0].sync_verify_compilation(generated_spec)
    else:
        compiled = True

    # Evaluate
    eval_result = evaluate_generated_spec(data, generated_spec, int(idx))

    if eval_result["compiles"]:
        logger.info(f"[{data.data_id}] Compiles | Full spec correct: {eval_result['full_spec_correct']}")
    else:
        logger.info(f"[{data.data_id}] FAIL - Compilation error")

    return {
        "idx": int(idx),
        "data_id": data.data_id,
        "compiles": eval_result["compiles"],
        "precond_sound_pass": eval_result["precond_sound_pass"],
        "precond_sound_total": eval_result["precond_sound_total"],
        "precond_complete_pass": eval_result["precond_complete_pass"],
        "precond_complete_total": eval_result["precond_complete_total"],
        "postcond_sound_pass": eval_result["postcond_sound_pass"],
        "postcond_sound_total": eval_result["postcond_sound_total"],
        "postcond_complete_pass": eval_result["postcond_complete_pass"],
        "postcond_complete_total": eval_result["postcond_complete_total"],
        "precond_correct": eval_result.get("precond_correct", False),
        "postcond_correct": eval_result.get("postcond_correct", False),
        "spec_sound": eval_result.get("spec_sound", False),
        "spec_complete": eval_result.get("spec_complete", False),
        "full_spec_correct": eval_result.get("full_spec_correct", False),
        "generated_tokens": generated_tokens,
        "precond": generated_spec["precond"],
        "postcond": generated_spec["postcond"],
        "num_times_forced": monitors[0].get_force_count() if monitors else 0,
        "finally_wrong": not compiled,
        "output_text": answer,
    }


# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verina spec generation benchmark with step verification")
    parser.add_argument("--monitor", "-m", action="store_true", default=False, help="Enable monitors")
    parser.add_argument("--num_examples", "-n", type=int, default=189, help="Number of examples to run")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logs and single-process mode")
    parser.add_argument("--port", type=int, default=8000, help="LLM server port")
    parser.add_argument("--main_model", type=str, default=MAIN_MODEL, help="Main model to use for generation")
    parser.add_argument("--earlystop_model", type=str, default=EARLYSTOP_MODEL, help="Model to use for early stopping")
    parser.add_argument("--k_steps", "-k", type=int, default=40, help="Newlines threshold for forcing spec output")
    parser.add_argument("--max_corrections", type=int, default=3,
                        help="Maximum number of correction attempts per example")
    parser.add_argument("--n_processes", "-p", type=int, default=16, help="Number of parallel worker processes")
    parser.add_argument("--n_exps", type=int, default=1, help="Number of independent sampling runs (produces outputs_solver_{i}.jsonl)")
    parser.add_argument("--extra", type=str, default="", help="Extra text description for the output directory")
    args = parser.parse_args()

    main_model = args.main_model
    earlystop_model = args.earlystop_model
    tokenizer = AutoTokenizer.from_pretrained(main_model, trust_remote_code=True)

    # ---- Unique, timestamped run directory ----
    model_short = get_model_short_name(main_model)
    mode = "monitor" if args.monitor else "solveronly"
    run_name = f"{model_short}_{mode}"
    if args.monitor:
        run_name += f"_maxcorr{args.max_corrections}_k{args.k_steps}"
    run_name += f"_nexps{args.n_exps}"
    if args.debug:
        run_name += "_debug"

    output_dir = os.path.join(
        _OUTPUT_ROOT, "Outputs_TTS", "VerinaSpecResults",
        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}-{run_name}',
    )
    if args.extra:
        output_dir += f"-{args.extra}"
    os.makedirs(output_dir, exist_ok=True)

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
    logger.info(f"Early stop model: {earlystop_model}")
    logger.info(f"Output directory: {output_dir}")

    logger.info("Loading verina dataset...")
    dataset = load_verina_dataset()
    logger.info(f"Loaded {len(dataset)} tasks")

    print("=============testing lean compile=================")
    test_lean_compile()

    N = args.num_examples if args.num_examples > 0 else len(dataset)
    indices = [i for i in range(N)]

    logger.info(f"Running on {N} examples...")
    logger.info(f"Monitor: {args.monitor} | Examples: {N} | Processes: {args.n_processes} | Runs: {args.n_exps}")

    for exp_i in range(args.n_exps):
        if args.n_exps > 1:
            print(f"\n=== Run {exp_i + 1}/{args.n_exps} ===")

        reason_dir = os.path.join(output_dir, "Reasoning_output", f"solver_{exp_i}")
        os.makedirs(reason_dir, exist_ok=True)
        outputs_file = os.path.join(output_dir, f"outputs_solver_{exp_i}.jsonl")
        results_csv = os.path.join(output_dir, f"results_solver_{exp_i}.csv")
        summary_file = os.path.join(output_dir, f"summary_solver_{exp_i}.json")

        tasks = [(args, int(idx)) for idx in indices]

        if args.debug:
            results = [run_one(t) for t in tqdm(tasks, desc="VerinaSpec")]
        else:
            with Pool(processes=args.n_processes) as pool:
                results = list(tqdm(
                    pool.imap_unordered(run_one, tasks),
                    total=len(tasks),
                    desc="VerinaSpec",
                ))

        results = [r for r in results if r is not None]

        # Save raw outputs
        with open(outputs_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        # Save CSV
        save_results_csv(results, results_csv)

        # Compute statistics
        num_compile = sum(1 for r in results if r["compiles"])

        total_precond_sound_pass = sum(r["precond_sound_pass"] for r in results)
        total_precond_sound_total = sum(r["precond_sound_total"] for r in results)
        total_precond_complete_pass = sum(r["precond_complete_pass"] for r in results)
        total_precond_complete_total = sum(r["precond_complete_total"] for r in results)
        total_postcond_sound_pass = sum(r["postcond_sound_pass"] for r in results)
        total_postcond_sound_total = sum(r["postcond_sound_total"] for r in results)
        total_postcond_complete_pass = sum(r["postcond_complete_pass"] for r in results)
        total_postcond_complete_total = sum(r["postcond_complete_total"] for r in results)

        compile_rate = num_compile / N if N > 0 else 0
        precond_sound_rate = total_precond_sound_pass / max(1, total_precond_sound_total)
        precond_complete_rate = total_precond_complete_pass / max(1, total_precond_complete_total)
        postcond_sound_rate = total_postcond_sound_pass / max(1, total_postcond_sound_total)
        postcond_complete_rate = total_postcond_complete_pass / max(1, total_postcond_complete_total)

        num_precond_correct = sum(1 for r in results if r.get("precond_correct", False))
        num_postcond_correct = sum(1 for r in results if r.get("postcond_correct", False))
        num_spec_sound = sum(1 for r in results if r.get("spec_sound", False))
        num_spec_complete = sum(1 for r in results if r.get("spec_complete", False))
        num_full_spec_correct = sum(1 for r in results if r.get("full_spec_correct", False))

        precond_correct_rate = num_precond_correct / N if N > 0 else 0
        postcond_correct_rate = num_postcond_correct / N if N > 0 else 0
        spec_sound_rate = num_spec_sound / N if N > 0 else 0
        spec_complete_rate = num_spec_complete / N if N > 0 else 0
        full_spec_correct_rate = num_full_spec_correct / N if N > 0 else 0

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS - SPECIFICATION GENERATION (run {exp_i})")
        print(f"{'='*60}")
        print(f"Model: {main_model}")
        print(f"Total examples: {N}")
        print(f"Successful compilations: {num_compile} ({compile_rate:.2%})")
        print(f"\n--- Individual Metrics (test-level) ---")
        print(f"Precondition Soundness: {total_precond_sound_pass}/{total_precond_sound_total} ({precond_sound_rate:.2%})")
        print(f"Precondition Completeness: {total_precond_complete_pass}/{total_precond_complete_total} ({precond_complete_rate:.2%})")
        print(f"Postcondition Soundness: {total_postcond_sound_pass}/{total_postcond_sound_total} ({postcond_sound_rate:.2%})")
        print(f"Postcondition Completeness: {total_postcond_complete_pass}/{total_postcond_complete_total} ({postcond_complete_rate:.2%})")
        print(f"\n--- Combined Metrics (task-level) ---")
        print(f"Precond Fully Correct (sound+complete): {num_precond_correct}/{N} ({precond_correct_rate:.2%})")
        print(f"Postcond Fully Correct (sound+complete): {num_postcond_correct}/{N} ({postcond_correct_rate:.2%})")
        print(f"Spec Sound (precond+postcond sound): {num_spec_sound}/{N} ({spec_sound_rate:.2%})")
        print(f"Spec Complete (precond+postcond complete): {num_spec_complete}/{N} ({spec_complete_rate:.2%})")
        print(f"\nFULL SPEC CORRECT (all sound+complete): {num_full_spec_correct}/{N} ({full_spec_correct_rate:.2%})")
        print(f"Results saved to: {results_csv}")

        # Save summary
        with open(summary_file, "w") as f:
            json.dump({
                "model": main_model,
                "earlystop_model": earlystop_model,
                "total_examples": N,
                "num_compile": num_compile,
                "compile_rate": compile_rate,
                "precond_sound_pass": total_precond_sound_pass,
                "precond_sound_total": total_precond_sound_total,
                "precond_sound_rate": precond_sound_rate,
                "precond_complete_pass": total_precond_complete_pass,
                "precond_complete_total": total_precond_complete_total,
                "precond_complete_rate": precond_complete_rate,
                "postcond_sound_pass": total_postcond_sound_pass,
                "postcond_sound_total": total_postcond_sound_total,
                "postcond_sound_rate": postcond_sound_rate,
                "postcond_complete_pass": total_postcond_complete_pass,
                "postcond_complete_total": total_postcond_complete_total,
                "postcond_complete_rate": postcond_complete_rate,
                "num_precond_correct": num_precond_correct,
                "precond_correct_rate": precond_correct_rate,
                "num_postcond_correct": num_postcond_correct,
                "postcond_correct_rate": postcond_correct_rate,
                "num_spec_sound": num_spec_sound,
                "spec_sound_rate": spec_sound_rate,
                "num_spec_complete": num_spec_complete,
                "spec_complete_rate": spec_complete_rate,
                "num_full_spec_correct": num_full_spec_correct,
                "full_spec_correct_rate": full_spec_correct_rate,
            }, f, indent=2)

        logger.info(
            f"Run {exp_i}: FullSpecCorrect={full_spec_correct_rate:.2%} "
            f"CompileRate={compile_rate:.2%} Summary saved to {summary_file}"
        )
