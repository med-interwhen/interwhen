"""
Medical Reasoning (MedReason) experiment runner.

This is the general entry point: it loads a question from the MedReason
HuggingFace dataset, sends it as a prompt to the solver LLM, and traces the
reasoning via stream_completion() + MedicalMonitor.

The model is NOT told to use any particular reasoning structure. The only
structural requirement, added at the prompt level (see medical_helper.py), is
that reasoning happens inside a single <think>...</think> block — matching
every other InterWhen dataset integration (Game24, Maze, SpatialMap,
ZebraLogic, Verina).

MedicalMonitor currently:
  - step_extractor: fires every K non-empty lines generated inside <think>.
  - verify: stub, always passes (no real verifier yet).
  - fix: no-op (never invoked while verify() is a stub).

Usage (local vLLM server required):
    python examples/TTSwithVerification/interwhen/medical_example.py --solver_lm Qwen/QwQ-32B --port 8000 --monitor --line_interval 5 --num_examples 10 --debug

Without the monitor (plain baseline generation):
    python examples/TTSwithVerification/interwhen/medical_example.py --solver_lm Qwen/QwQ-32B --port 8000 --num_examples 10
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from multiprocessing import Pool

from tqdm import tqdm
from transformers import AutoTokenizer

from interwhen import stream_completion
from interwhen.monitors import MedicalMonitor
from interwhen.utils.medical_helper import (
    get_medical_dataset,
    build_prompt,
    extract_post_think_answer,
)

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Module-level tokenizer — initialised in __main__ for multiprocessing safety
tokenizer = None


# ---------------------------------------------------------------------------
# LLM server configuration
# ---------------------------------------------------------------------------

def init_llm_server(model_name: str, max_tokens: int = 32768, port: int = 8000) -> dict:
    """Return an llm_server config dict expected by stream_completion()."""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "temperature": 0.6,
        "stream": True,
        "logprobs": 20,
        "use_beam_search": False,
        "prompt_cache": True,
        "seed": 42,
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}


# ---------------------------------------------------------------------------
# Single-problem runner
# ---------------------------------------------------------------------------

def run(args, problem: dict) -> dict:
    """Send one MedReason question to the solver LLM and trace its reasoning.

    Args:
        args:    Parsed argparse namespace.
        problem: Problem dict from get_medical_dataset() — contains
                 question, options, answer, reasoning, id, dataset_name.

    Returns:
        Result dict written to the JSONL output file.
    """
    global tokenizer

    problem_id = problem["id"]
    output_file = os.path.join(args.output_dir, "outputs_medical.jsonl")

    llm_server = init_llm_server(args.solver_lm, max_tokens=args.max_tokens, port=args.port)
    prompt = build_prompt(problem, tokenizer)

    if args.monitor:
        monitors = [
            MedicalMonitor(
                name="MedicalMonitor",
                instance=problem,
                line_interval=args.line_interval,
                max_corrections=args.monitor_max_corrections,
            )
        ]
    else:
        monitors = []

    output_text = asyncio.run(
        stream_completion(
            prompt,
            llm_server=llm_server,
            monitors=tuple(monitors) if monitors else [],
            add_delay=False,
            async_execution=not args.debug,
        )
    )

    predicted_answer = extract_post_think_answer(output_text)

    output = {
        "problem_id": problem_id,
        "dataset_name": problem.get("dataset_name", ""),
        "question": problem["question"],
        "options": problem.get("options", ""),
        "reference_answer": problem.get("answer", ""),
        "output_text": output_text,
        "predicted_answer": predicted_answer,
    }

    with open(output_file, "a") as f:
        f.write(json.dumps(output, default=str) + "\n")

    return output


def _run_wrapper(args_problem):
    """Multiprocessing-safe wrapper."""
    return run(*args_problem)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medical Reasoning (MedReason) LLM Solver with MedicalMonitor"
    )
    parser.add_argument(
        "--solver_lm", type=str, required=True,
        help="Solver LLM model name (e.g. Qwen/QwQ-32B)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="vLLM server port (default: 8000)",
    )
    parser.add_argument(
        "--monitor", "-m", action="store_true",
        help="Enable MedicalMonitor (reasoning tracing during generation)",
    )
    parser.add_argument(
        "--monitor_max_corrections", type=int, default=50,
        help="Maximum feedback injections per problem (default: 50)",
    )
    parser.add_argument(
        "--line_interval", type=int, default=5,
        help="Trigger monitor verification every N non-empty reasoning lines "
             "inside <think> (default: 5)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=16384,
        help="Max tokens for generation (default: 16384)",
    )
    parser.add_argument(
        "--dataset_name_filter", type=str, default=None,
        help="Optional: restrict to one MedReason source dataset "
             "(e.g. medmcqa, pubmedqa, etc.)",
    )
    parser.add_argument(
        "--num_examples", type=int, default=10,
        help="Number of MedReason questions to run (default: 10)",
    )
    parser.add_argument(
        "--n_processes", "-p", type=int, default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Debug mode: single process, verbose logging, run 1 example",
    )
    parser.add_argument(
        "--continue_from", "-c", type=str, default=None,
        help="Continue from a previous output directory",
    )
    parser.add_argument(
        "--extra", type=str, default="",
        help="Extra label appended to the output directory name",
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )

    # Module-level tokenizer (must be set before multiprocessing)
    tokenizer = AutoTokenizer.from_pretrained(args.solver_lm)

    # Load dataset from HuggingFace (UCSC-VLAA/MedReason)
    logger.info("Loading MedReason dataset from HuggingFace...")
    ds = get_medical_dataset(
        dataset_name_filter=args.dataset_name_filter,
        limit=args.num_examples,
    )

    # Output directory setup
    if args.continue_from:
        output_dir = f"Outputs_TTS/medical/{args.continue_from}"
    else:
        output_dir = f"Outputs_TTS/medical/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if args.extra:
            output_dir += f"-{args.extra}"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    args.output_dir = output_dir
    output_file = os.path.join(output_dir, "outputs_medical.jsonl")

    # Resume support 
    if args.continue_from and os.path.exists(output_file):
        with open(output_file) as f:
            completed_ids = {json.loads(line)["problem_id"] for line in f}
        ds_run = [p for p in ds if p["id"] not in completed_ids]
        logger.warning(
            "Continuing from %s — skipping %d completed problems.",
            args.continue_from, len(completed_ids),
        )
    else:
        with open(output_file, "w") as f:
            f.write("")
        ds_run = ds

    # Run
    if not args.debug:
        with Pool(processes=args.n_processes) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_run_wrapper, [(args, p) for p in ds_run]),
                    total=len(ds_run),
                    desc="Medical problems",
                )
            )
    else:
        # Single-process debug mode — run only the first problem
        _run_wrapper((args, ds_run[0]))

    print(f"\nDone. Results written to {output_file}")