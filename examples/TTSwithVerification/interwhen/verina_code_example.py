import argparse
import asyncio
import json
import logging
import os
import numpy as np
import csv
import asyncio
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from interwhen.interject import stream_completion
from interwhen.monitors import StepVerifierVerinaMonitor
from interwhen.utils.llm import init_llm_server, get_think_tags
from interwhen.utils.verina_code_example_utils import *
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Model config
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
    # Get the last part after '/' and replace any problematic characters
    short_name = model_name.split("/")[-1]
    short_name = short_name.replace(" ", "_").replace(":", "-")
    return short_name


# Saving Utils
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
        writer.writerow(["idx", "data_id", "compiles", "all_tests_pass", "num_tests", "num_tests_passed", "generated_tokens", "generated_code","finally_wrong"])
        for r in results:
            # Escape newlines in generated_code for CSV compatibility
            code_escaped = r["generated_code"].replace("\n", "\\n") if r["generated_code"] else ""
            writer.writerow([
                r["idx"], 
                r["data_id"], 
                r["compiles"], 
                r["all_tests_pass"], 
                r["num_tests"],
                r["num_tests_passed"],
                r["generated_tokens"], 
                code_escaped,
                r['finally_wrong']
            ])


def plot_entropy_ewma(monitors, save_path):
    """Plot entropy and EWMA metrics."""
    entropy = monitors[0].entropy
    ema_mean = monitors[0].ema_means
    ema_var = monitors[0].ema_vars

    chunks_no = list(range(1, len(entropy) + 1))

    if monitors[0].exit_point is None:
        exit_point = len(entropy) - 1
    else:
        exit_point = monitors[0].exit_point - 1
    plt.figure(figsize=(12, 7))
    plt.plot(chunks_no, entropy, label="Entropy", linewidth=1.8)
    plt.plot(chunks_no, ema_mean, label="EWMA Mean", linewidth=1.8)
    plt.plot(chunks_no, ema_var, label="EWMA Variance", linewidth=1.8)

    plt.axvline(exit_point, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

    # Star markers on each curve
    plt.plot(exit_point, entropy[exit_point], "r*", markersize=14)
    plt.plot(exit_point, ema_mean[exit_point], "r*", markersize=14)
    plt.plot(exit_point, ema_var[exit_point], "r*", markersize=14)

    # Label the exit point
    plt.text(exit_point + 0.3, entropy[exit_point],
             f" Exit @ {exit_point}", color="red", fontsize=10)

    plt.xlabel("Chunk Index")
    plt.ylabel("Value")
    plt.title("EAT per Chunk")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_entropy_ewma2(monitors, save_path):
    """Plot EWMA variance."""
    chunks_no = list(range(1, len(monitors[0].entropy) + 1))
    plt.figure(figsize=(12, 7))
    plt.plot(chunks_no, monitors[0].ema_vars, label="EWMA Variance", linewidth=1.8)
    if monitors[0].exit_point is None:
        exit_point = len(monitors[0].ema_vars) - 1
    else:
        exit_point = monitors[0].exit_point - 1
    plt.axvline(exit_point, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    plt.plot(exit_point, monitors[0].ema_vars[exit_point], "r*", markersize=14)

    plt.xlabel("Chunk Index")
    plt.ylabel("Value")
    plt.title("EWMA Variance per Chunk")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_entropy_ewma3(monitors, save_path):
    """Plot DEER confidence."""
    chunks_no = list(range(1, len(monitors[0].confidence) + 1))
    plt.figure(figsize=(12, 7))
    plt.plot(chunks_no, monitors[0].confidence, label="DEER Confidence", linewidth=1.8)

    plt.xlabel("Chunk Index")
    plt.ylabel("Value")
    plt.title("DEER Confidence per Chunk")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_one(task):
    """Process a single Verina code example in a worker process.

    ``task`` is ``(args, idx)``.  Returns a result dict that the main process
    aggregates.  Uses the module-level ``tokenizer``, ``dataset`` and
    ``reason_dir`` globals (inherited via fork).
    """
    args, idx = task
    main_model = args.main_model
    think_tags = get_think_tags(main_model)

    llm_server = init_llm_server(main_model, context_length=20480, port=args.port)

    data = dataset[idx]

    # Build prompt
    prompt = build_full_prompt(data, tokenizer)

    # Convert BenchmarkData to dict for the monitor
    task_data = {
        "data_id": data.data_id,
        "description": data.description,
        "signature": data.signature,
        "lean_data": data.lean_data,
        "spec_desc": data.spec_desc,
        "tests": data.tests,
        "metadata": data.metadata,
    }

    # Setup monitors
    if args.monitor:
        monitors = [
            StepVerifierVerinaMonitor(
                name="VerinaStepVerifier",
                task_data=task_data,
                llm_server=llm_server,
                prompt=prompt,
                k_steps=40,  # Force code after every K newlines
                compile_timeout=120,
                max_corrections=args.max_corrections,
                open_think=think_tags['open'],
                close_think=think_tags['close'],
                tokenizer=tokenizer,
            ),
        ]
    else:
        monitors = []

    # Run LLM with streaming + monitor
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
            "all_tests_pass": False,
            "num_tests": len(data.tests) if data.tests else 0,
            "num_tests_passed": 0,
            "generated_tokens": 0,
            "generated_code": "",
            "num_times_code_forced": 0,
            "finally_wrong": True,
            "output_text": "",
        }

    # Save reasoning trace
    save_reasoning_trace(int(idx), data.data_id, prompt_with_answer, reason_dir)

    generated_tokens = count_tokens(answer, tokenizer)

    generated_code = extract_code_from_response(answer, think_tags['open'], think_tags['close'])
    old_code = generated_code

    # Final code verification loop - retry if compilation fails
    if args.monitor and monitors and generated_code:
        final_code, final_compiles, final_output, num_final_retries = asyncio.run(
            monitors[0].verify_final_code(
                code=generated_code,
                prompt_with_answer=prompt_with_answer,
                max_retries=1
            )
        )
        if final_code != generated_code:
            logger.info(f"[Final verification] Code fixed after {num_final_retries} retries")
            generated_code = final_code

    # check for soundness
    if args.monitor and monitors and generated_code:
        compiled, _ = monitors[0].sync_verify_compilation(generated_code)
    else:
        compiled = True

    # Evaluate - now includes unit tests
    compiles, all_tests_pass, compile_output, test_results = evaluate_generated_code(data, generated_code, int(idx))

    num_tests = len(data.tests) if data.tests else 0
    num_tests_passed = sum(1 for v in test_results.values() if v == "pass")

    if compiles and all_tests_pass:
        logger.info(f"[{data.data_id}] PASS - compiles and all {num_tests} tests pass")
    elif compiles:
        logger.info(f"[{data.data_id}] PARTIAL - compiles but {num_tests - num_tests_passed}/{num_tests} tests failed")
    else:
        logger.info(f"[{data.data_id}] FAIL - compilation error")
    logger.info(f"[{data.data_id}] Code Generated: {True if old_code.strip() else False}")

    return {
        "idx": int(idx),
        "data_id": data.data_id,
        "compiles": compiles,
        "all_tests_pass": all_tests_pass,
        "num_tests": num_tests,
        "num_tests_passed": num_tests_passed,
        "generated_tokens": generated_tokens,
        "generated_code": generated_code,
        "num_times_code_forced": monitors[0].get_force_count() if monitors else 0,
        "finally_wrong": not compiled,
        "output_text": answer,
    }


# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verina benchmark solver with LLM and monitors")
    parser.add_argument("--monitor", "-m", action="store_true", default=False, help="Enable monitors")
    parser.add_argument("--num_examples", "-n", type=int, default=189, help="Number of examples to run")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logs and single-process mode")
    parser.add_argument("--port", type=int, default=8000, help="LLM server port")
    parser.add_argument("--main_model", type=str, default=MAIN_MODEL, help="Main model to use for generation")
    parser.add_argument("--earlystop_model", type=str, default=EARLYSTOP_MODEL, help="Model to use for early stopping")
    parser.add_argument("--k_steps", "-k", type=int, default=75, help="Newlines threshold for forcing code output")
    parser.add_argument("--tasks", "-t", type=str, default=None, help="Comma-separated list of task IDs to run (e.g., verina_advanced_10,verina_basic_2)")
    parser.add_argument("--max_corrections", type=int, default=5, help="Maximum number of correction attempts per example")
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
        _OUTPUT_ROOT, "Outputs_TTS", "VerinaCodeResults",
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

    # Load dataset
    logger.info("Loading verina dataset...")
    dataset = load_verina_dataset()
    logger.info(f"Loaded {len(dataset)} tasks")

    print("=============testing lean compile=================")
    test_lean_compile()

    # Filter tasks if --tasks is specified
    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",")]
        dataset = [d for d in dataset if d.data_id in task_ids]
        logger.info(f"Filtered to {len(dataset)} tasks: {task_ids}")
        N = len(dataset)
        indices = list(range(N))
    else:
        # Select examples
        N = args.num_examples if args.num_examples > 0 else len(dataset)
        total = len(dataset)
        indices = np.linspace(0, total - 1, N, dtype=int)

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
            results = [run_one(t) for t in tqdm(tasks, desc="Verina")]
        else:
            with Pool(processes=args.n_processes) as pool:
                results = list(tqdm(
                    pool.imap_unordered(run_one, tasks),
                    total=len(tasks),
                    desc="Verina",
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
        num_all_tests_pass = sum(1 for r in results if r["all_tests_pass"])

        compile_rate = num_compile / N if N > 0 else 0
        accuracy = num_all_tests_pass / N if N > 0 else 0

        print(f"\n{'='*50}")
        print(f"FINAL RESULTS (run {exp_i})")
        print(f"{'='*50}")
        print(f"Model: {main_model}")
        print(f"Total examples: {N}")
        print(f"Successful compilations: {num_compile} ({compile_rate:.2%})")
        print(f"All tests pass: {num_all_tests_pass} ({accuracy:.2%})")
        print(f"Results saved to: {results_csv}")

        # Save summary
        with open(summary_file, "w") as f:
            json.dump({
                "model": main_model,
                "earlystop_model": earlystop_model,
                "total_examples": N,
                "num_compile": num_compile,
                "compile_rate": compile_rate,
                "num_all_tests_pass": num_all_tests_pass,
                "accuracy": accuracy,
            }, f, indent=2)

        logger.info(
            f"Run {exp_i}: Accuracy={accuracy:.2%} CompileRate={compile_rate:.2%} "
            f"Summary saved to {summary_file}"
        )
