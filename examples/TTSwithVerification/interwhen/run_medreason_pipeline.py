"""
run_medreason_pipeline.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import datetime
from multiprocessing import Pool

from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from interwhen import stream_completion
from interwhen.monitors.medical_monitor import MedicalMonitor
from interwhen.utils.medical_prompts import SYSTEM_PROMPT_MEDICAL, USER_PROMPT_TEMPLATE

# Module-level tokenizer, set in __main__ (multiprocessing-safe).
tokenizer = None


# ══════════════════════════════════════════════════════════════════════════════
# MEDREASON LOADING
# ══════════════════════════════════════════════════════════════════════════════

class MedReasonLoader:
    """Loads UCSC-VLAA/MedReason into {id, question, options, answer, reasoning} dicts."""

    HF_PATH = "UCSC-VLAA/MedReason"
    _OPT_RE = re.compile(r"([A-E])\.\s*(.*?)(?=\n[A-E]\.|$)", re.DOTALL)

    def load(self, split: str, start_idx: int, end_idx: int, max_samples: int) -> list:
        print(f"Loading {self.HF_PATH}  split={split} ...")
        raw = load_dataset(self.HF_PATH, split=split)

        end = min(end_idx, len(raw)) if end_idx > 0 else len(raw)
        raw = raw.select(range(start_idx, end))
        if max_samples > 0:
            raw = raw.select(range(min(max_samples, len(raw))))

        rows = [self._normalise(i, item) for i, item in enumerate(raw)]
        print(f"  -> {len(rows)} samples ready.\n")
        return rows

    def _normalise(self, idx: int, item: dict) -> dict:
        options = {k: v.strip() for k, v in self._OPT_RE.findall(str(item.get("options", "")))}
        raw_ans = str(item.get("answer", "")).split("Explanation:")[0].strip().rstrip(". ")
        return {
            "id":        str(item.get("id_in_dataset", idx)),
            "question":  item.get("question", ""),
            "options":   options,
            "answer":    self._match_answer(raw_ans, options),
            "reasoning": item.get("reasoning") or "",
        }

    @staticmethod
    def _match_answer(clean: str, options: dict) -> str:
        cl = clean.lower()
        for letter, text in options.items():
            if cl == text.lower() or cl in text.lower() or text.lower() in cl:
                return letter
        words = set(cl.split())
        best_l, best_n = "?", 0
        for letter, text in options.items():
            n = len(words & set(text.lower().split()))
            if n > best_n:
                best_n, best_l = n, letter
        return best_l if best_n > 0 else "?"


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def init_llm_server(model_name: str, max_tokens: int = 16 * 1024, port: int = 8000) -> dict:
    """Solver vLLM server config."""
    return {
        "url": f"http://localhost:{port}/v1/completions",
        "payload": {
            "model": model_name, "max_tokens": max_tokens,
            "top_k": 20, "top_p": 0.95, "min_p": 0.0, "temperature": 0.6,
            "stream": True, "logprobs": 20, "use_beam_search": False,
            "prompt_cache": True, "seed": 42,
        },
        "headers": {"Content-Type": "application/json"},
    }


def build_prompt(sample: dict, tok) -> str:
    case_text = sample["question"]
    if sample["options"]:
        case_text += "\n\nOptions:\n" + "\n".join(f"{k}. {v}" for k, v in sample["options"].items())
    user_prompt = USER_PROMPT_TEMPLATE.format(case_text=case_text)
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT_MEDICAL},
         {"role": "user",   "content": user_prompt}],
        tokenize=False, add_generation_prompt=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def exact_correctness_check(output_text: str, sample: dict) -> "bool | None":
    """
    Primary scorer: extract 'Selected Option: X' from [FINAL ANSWER] block.

    Qwen3 thinking models write a draft [FINAL ANSWER] inside the <think>
    block 77% of the time, then produce the real response after </think>.
    We look in the response section first; fall back to the think section if
    the response has no block (23% of outputs).
    Returns None if no block is found or it contains no option letter.
    """
    think_close = "</think>"
    if think_close in output_text:
        response_section = output_text.split(think_close, 1)[1]
    else:
        response_section = output_text

    def _extract(text: str) -> "str | None":
        m = re.search(r"\[FINAL ANSWER\](.*?)\[/FINAL ANSWER\]", text, re.DOTALL)
        if not m:
            return None
        opt = re.search(r"Selected Option:\s*([A-E])", m.group(1), re.IGNORECASE)
        return opt.group(1).strip().upper() if opt else None

    letter = _extract(response_section) or _extract(output_text)
    if letter is None:
        return None
    return letter == sample["answer"].strip().upper()


def rough_correctness_check(output_text: str, sample: dict) -> "bool | None":
    """
    Fallback scorer (word-overlap) when [FINAL ANSWER] block is absent.
    Not authoritative — use only as a fallback signal.
    """
    m     = re.search(r"\[FINAL ANSWER\](.*?)\[/FINAL ANSWER\]", output_text, re.DOTALL)
    block = (m.group(1) if m else output_text[-600:]).lower()
    gt    = sample["options"].get(sample["answer"], "").lower()
    if not gt:
        return None
    gt_words = set(gt.split())
    overlap  = len(gt_words & set(block.split()))
    return overlap >= max(1, len(gt_words) // 2)


def check_correctness(output_text: str, sample: dict) -> "bool | None":
    """Try exact match first; fall back to word overlap."""
    result = exact_correctness_check(output_text, sample)
    if result is not None:
        return result
    return rough_correctness_check(output_text, sample)


# ══════════════════════════════════════════════════════════════════════════════
# PER-SAMPLE RUN
# ══════════════════════════════════════════════════════════════════════════════

def run(args, sample: dict) -> dict:
    global tokenizer
    llm_server = init_llm_server(args.solver_lm, port=args.port)
    prompt     = build_prompt(sample, tokenizer)

    monitors = []
    if args.monitor:
        monitors = [MedicalMonitor(
            name             = "MedicalVerifier",
            instance         = sample,
            max_corrections  = args.monitor_max_corrections,
            verifier_port    = args.verifier_port,
            verifier_model   = args.verifier_model,
            run_snomed       = not args.no_snomed,
        )]

    output_text = asyncio.run(stream_completion(
        prompt,
        llm_server     = llm_server,
        monitors       = tuple(monitors),
        async_execution= not args.debug,
    ))

    correct        = check_correctness(output_text, sample)
    exact_matched  = exact_correctness_check(output_text, sample)
    decision_log   = monitors[0].verifier.decision_log if monitors else []

    result = {
        "sample_id":       sample["id"],
        "question":        sample["question"],
        "ground_truth":    sample["answer"],
        "output_text":     output_text,
        "correct":         correct,
        "exact_matched":   exact_matched is not None,  # whether [FINAL ANSWER] block was found
        "decision_log":    decision_log,
    }
    with open(f"{args.output_dir}/outputs.jsonl", "a") as f:
        f.write(json.dumps(result, default=str) + "\n")
    return result


def _run_wrapper(args_sample):
    return run(*args_sample)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--solver_lm",              type=str,  required=True)
    ap.add_argument("--port",                   type=int,  default=8000)
    ap.add_argument("--monitor",   "-m",        action="store_true")
    ap.add_argument("--verifier_port",          type=int,  default=8001)
    ap.add_argument("--verifier_model",         type=str,  default="medverifier")
    ap.add_argument("--monitor_max_corrections",type=int,  default=5)
    ap.add_argument("--no_snomed",              action="store_true")
    ap.add_argument("--split",                  type=str,  default="train")
    ap.add_argument("--max_samples",            type=int,  default=20)
    ap.add_argument("--start_idx",              type=int,  default=0)
    ap.add_argument("--end_idx",                type=int,  default=-1)
    ap.add_argument("--n_processes", "-p",      type=int,  default=8)
    ap.add_argument("--debug",       "-d",      action="store_true")
    ap.add_argument("--continue_from", "-c",    type=str,  default=None)
    ap.add_argument("--extra",                  type=str,  default="")

    return ap.parse_args()


def main():
    global tokenizer
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.solver_lm)
    samples   = MedReasonLoader().load(args.split, args.start_idx, args.end_idx, args.max_samples)

    if args.continue_from:
        output_dir = f"Outputs_TTS/medreason/{args.continue_from}"
    else:
        output_dir = f"Outputs_TTS/medreason/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if args.extra:
            output_dir += f"-{args.extra}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)
    args.output_dir = output_dir

    output_file = f"{output_dir}/outputs.jsonl"
    if args.continue_from:
        with open(output_file, "r") as f:
            done_ids = {json.loads(line)["sample_id"] for line in f}
        samples = [s for s in samples if s["id"] not in done_ids]
        print(f"Continuing from {args.continue_from}, skipping {len(done_ids)} completed.")
    else:
        open(output_file, "w").close()

    if not args.debug:
        with Pool(processes=args.n_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(_run_wrapper, [(args, s) for s in samples]),
                total=len(samples),
            ))
    else:
        results = [run(args, samples[0])] if samples else []

    scored = [r["correct"] for r in results if r.get("correct") is not None]
    exact  = [r["exact_matched"] for r in results]
    if scored:
        print(f"\nAccuracy:       {sum(scored)}/{len(scored)} = {sum(scored)/len(scored):.2%}")
        print(f"[FINAL ANSWER] block found in {sum(exact)}/{len(exact)} outputs")
    print(f"Output -> {output_file}")


if __name__ == "__main__":
    main()
